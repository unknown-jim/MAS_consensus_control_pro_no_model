"""Phase 2 的生成图、局部观测与多规模训练契约。"""
import math
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

import mas_cc.agent as agents
from mas_cc import config
from mas_cc.agent import ActorOnlyPolicy, CTDEMAPPOAgent, CTDESACAgent
from mas_cc.config import DEVICE, STATE_DIM
from mas_cc.environment import BatchedModelFreeEnv, ModelFreeEnv
from mas_cc.networks import DecentralizedActor, LightweightAttentionEncoder, _parse_flat_state
from mas_cc.topology import CommunicationTopology
from mas_cc.utils import collect_trajectory, evaluate_agent


def chain_scenario(n):
    adj = np.eye(n + 1, k=-1, dtype=np.float32)
    return dict(adj_matrix=adj, pinned_ids=[1], trajectory_type="cosine",
                leader_amplitude=2., leader_omega=.5, leader_phase=0.,
                initial_positions=np.arange(n, dtype=np.float32) / 4,
                initial_velocities=np.arange(n, dtype=np.float32) / 8)


class TestGraphAndEnvironment(unittest.TestCase):
    def test_config_widths(self):
        self.assertEqual(config.OBS_VERSION, 2)
        self.assertEqual(config.OBS_LAYOUT, "v2_rel_10_7k")
        self.assertEqual(config.OBS_AGE_CAP, 300)
        self.assertEqual(config.SELF_STATE_DIM, 10)
        self.assertEqual(config.NEIGHBOR_FEAT_DIM, 7)
        self.assertEqual(config.STATE_DIM, 10 + 7 * config.MAX_NEIGHBORS)
        for n in (1, 10, 20, 30):
            self.assertEqual(config.global_state_dim_for(n), 4 * (n + 1) + 6)

    def test_local_rng_degrees_and_ratio_rounding(self):
        np.random.seed(913)
        expected = np.random.random(8)
        np.random.seed(913)
        means = []
        max_degrees = []
        for seed in range(40):
            n = 50
            mode = "low" if seed % 2 else "normal"
            lo, hi = (.02, .06) if mode == "low" else (.15, .30)
            graph = CommunicationTopology(n, seed=seed, pinned_mode=mode)
            ratio = np.random.default_rng(seed).uniform(lo, hi)
            self.assertEqual(len(graph.pinned_ids), max(1, math.floor(n * ratio + .5)))
            self.assertAlmostEqual(graph.mean_in_degree, graph.adj.sum().item() / n)
            means.append(graph.mean_in_degree)
            max_degrees.append(graph.max_in_degree)
        np.testing.assert_equal(np.random.random(8), expected)
        self.assertLess(abs(np.mean(means) - 3.), .12)
        self.assertGreater(max(max_degrees), 4)
        with self.assertRaisesRegex(ValueError, "Generator"):
            CommunicationTopology(5, rng=np.random)
        for invalid in (0, 6, True, 1.5):
            with self.assertRaises(ValueError):
                CommunicationTopology(5, num_pinned=invalid)

    def test_bfs_matches_final_directed_graph(self):
        for n, p in ((1, 1), (5, 5), (10, 9), (20, 1), (100, 4)):
            for mode in ("deep", "shallow", "random"):
                with self.subTest(n=n, p=p, mode=mode):
                    graph = CommunicationTopology(n, p, 7, depth_mode=mode)
                    network = nx.from_numpy_array(graph.adj.cpu().numpy().T,
                                                  create_using=nx.DiGraph)
                    depths = nx.single_source_shortest_path_length(network, 0)
                    self.assertTrue(graph.is_connected)
                    self.assertEqual(graph.reachable_count, n + 1)
                    self.assertEqual(list(graph.node_depths), [depths[i] for i in range(n+1)])
                    counts = np.bincount(list(depths.values())).tolist()
                    self.assertEqual(list(graph.layer_counts), counts)
                    self.assertEqual(counts[1], p)
                    if mode != "random":
                        h = 2 if mode == "shallow" else max(4, math.ceil(math.sqrt(n)))
                        self.assertEqual(graph.max_depth, min(n - p + 1, h))
                        if len(counts) > 2:
                            self.assertLessEqual(max(counts[2:]) - min(counts[2:]), 1)

    def test_frozen_graph_validates_and_owns_input(self):
        adj = chain_scenario(4)["adj_matrix"]
        frozen = CommunicationTopology.from_frozen(adj, [1])
        adj[:] = 0
        frozen.adj.zero_()
        self.assertEqual(frozen.get_receivers(0), [1])
        with self.assertRaises(ValueError):
            frozen.randomize()
        good = chain_scenario(4)["adj_matrix"]
        bad_cases = [(good, [2]), (good, [1, 1]), (good[:, :3], [1]),
                     (good + np.eye(5), [1]), (good * 0, []), (good * .5, [1])]
        disconnected = good.copy()
        disconnected[3, 2] = 0
        bad_cases.append((disconnected, [1]))
        for matrix, pins in bad_cases:
            with self.assertRaises(ValueError):
                CommunicationTopology.from_frozen(matrix, pins)

    def test_reset_subset_preserves_rows_clears_estimates(self):
        env = BatchedModelFreeEnv(CommunicationTopology(8, 2), 2)
        env.step(torch.ones((2, 8, 2), device=config.DEVICE))
        for prefix in ("leader_est_", "last_broadcast_leader_"):
            for field in ("pos", "vel", "seq"):
                getattr(env, prefix + field).fill_(39)
        env._get_state_optimized()
        other = {key: val[1].clone() for key, val in vars(env).items()
                 if torch.is_tensor(val) and val.ndim and val.shape[0] == 2}
        gen, rev = env.episode_gen, env.graph_rev
        env.reset([0])
        self.assertEqual((env.episode_gen, env.graph_rev), (gen, rev))
        for key, before in other.items():
            self.assertTrue(torch.equal(getattr(env, key)[1], before), key)
        for ids in ([0], [1, 0]):
            env.reset(ids)
            for row in ids:
                known = [0, *env.topology.pinned_ids]
                unknown = sorted(set(range(9)) - set(known))
                for prefix in ("leader_est_", "last_broadcast_leader_"):
                    for field, init in (("pos", env.positions[row, 0]), ("vel", env.velocities[row, 0]), ("seq", 0)):
                        value = getattr(env, prefix + field)[row]
                        self.assertTrue((value[known] == init).all())
                        self.assertTrue((value[unknown] == (-1 if field == "seq" else 0)).all())

    def test_reset_validation_and_one_draw(self):
        env = BatchedModelFreeEnv(CommunicationTopology(5), 2, auto_reset=False)
        with patch.object(env, "_randomize_topology", wraps=env._randomize_topology) as draw:
            for ids in ([0, 0], [1., 0.], [-1], [2], [True], [False, 1], [[0]], "0"):
                with self.assertRaises(ValueError, msg=str(ids)):
                    env.reset(ids)
            self.assertEqual(draw.call_count, 0)
            self.assertEqual(env.episode_gen, 0)
            rev = env.graph_rev
            env.reset([1, 0])
            self.assertEqual((draw.call_count, env.episode_gen, env.graph_rev), (1, 1, rev+1))
            env.topology.randomize()
            self.assertGreater(env.graph_rev, rev+1)

    def test_scenario_first_frame_and_reuse(self):
        scenario = chain_scenario(5)
        env = BatchedModelFreeEnv(CommunicationTopology(5), 2, fixed_scenario=scenario)
        x = env.positions.clone()
        self.assertEqual(x[0].tolist(), [2., 0., .25, .5, .75, 1.])
        self.assertEqual(env.velocities[1].tolist(), [0., 0., .125, .25, .375, .5])
        self.assertTrue(torch.equal(x, env.last_broadcast_pos))
        first = env._get_state_optimized()
        rev = env.graph_rev
        scenario["initial_positions"][:] = 8.
        env.step(torch.ones(2, 5, 2, device=config.DEVICE))
        self.assertTrue(torch.equal(first, env.reset()))
        self.assertEqual(env.graph_rev, rev)
        self.assertEqual(env.episode_gen, 2)

    def test_install_full_frame_rejects_inconsistent_leader_atomically(self):
        env = BatchedModelFreeEnv(CommunicationTopology(3), 2)
        env.step(torch.zeros(2, 3, 2, device=config.DEVICE))
        graph = env.topology
        counts = (env.graph_rev, env.frame_rev, env.episode_gen)
        before = env.get_global_state()
        scenario = chain_scenario(3)
        scenario["initial_positions"] = [9., 0., .25, .5]
        with self.assertRaisesRegex(ValueError, "leader"):
            env.reset(scenario=scenario)
        self.assertIs(env.topology, graph)
        self.assertEqual(counts, (env.graph_rev, env.frame_rev, env.episode_gen))
        self.assertTrue(torch.equal(env.get_global_state(), before))
        scenario["initial_positions"][0] = 2.
        state = env.reset(scenario=scenario)
        self.assertEqual(env.positions[1].tolist(), scenario["initial_positions"])
        self.assertTrue(torch.equal(state, env._get_state_optimized()))

    def test_v2_features_age_mask_and_translation(self):
        env = BatchedModelFreeEnv(CommunicationTopology(4), 1, fixed_scenario=chain_scenario(4))
        env.positions[0, 3], env.velocities[0, 3] = 4., -2.
        env.leader_seq.fill_(400)
        state = env._get_state_optimized()
        expect = [-2/config.VEL_LIMIT, (4.-.5)/20, (-2.-.25)/20, 0., 0., 1., 0., 0., 0., 1.]
        torch.testing.assert_close(state[0, 3, :10], state.new_tensor(expect))
        slots = state[0, 3, 10:].view(-1, 7)
        torch.testing.assert_close(slots[0], state.new_tensor([(.25-4)/20, (.125+2)/20, 0, 0, 1, 0, 1]))
        self.assertTrue((slots[1:] == 0).all())
        env.last_broadcast_leader_seq[0, 2] = 390
        known = env._get_state_optimized()
        self.assertAlmostEqual(known[0, 3, 14].item(), math.log1p(10)/math.log1p(300), places=6)
        self.assertEqual(known[0, 1, 5].item(), 1.)
        for field in ("positions", "last_broadcast_pos", "leader_est_pos", "last_broadcast_leader_pos"):
            getattr(env, field).add_(5.)
        torch.testing.assert_close(env._get_state_optimized(), known)

    def test_renumber_before_topk_and_gossip_all_neighbors(self):
        k = config.MAX_NEIGHBORS
        n = k+3
        spec = chain_scenario(n)
        adj = spec["adj_matrix"]
        adj[1:n, 0] = 1
        adj[n, 1:n] = 1
        permutation = np.r_[0, np.arange(n-1, 0, -1), n]
        selected = []
        for perm in (np.arange(n+1), permutation):
            scenario = dict(spec, adj_matrix=adj[perm][:, perm], pinned_ids=list(range(1,n)),
                            initial_positions=spec["initial_positions"][perm[1:]-1],
                            initial_velocities=spec["initial_velocities"][perm[1:]-1])
            env = BatchedModelFreeEnv(CommunicationTopology(n), 1, fixed_scenario=scenario)
            env.leader_seq.fill_(1000)
            env.last_broadcast_leader_seq[0, 1:n] = -1
            unknown = env._get_state_optimized()[0, n].clone()
            env.last_broadcast_leader_seq[0, 1:n] = 1
            # 同龄跨越 K 边界，且 1 与 2 的编码 age 均已饱和。
            freshest = int(np.where(perm == n-1)[0][0])
            env.last_broadcast_leader_seq[0, freshest] = 2
            known = env._get_state_optimized()[0, n]
            dx = (env.last_broadcast_pos[0,freshest]-env.positions[0,n]) / (2*config.POS_LIMIT)
            self.assertEqual(known[10].item(), dx.item())
            selected.append((unknown, known))
            # 唯一新广播在原始候选槽 K 之外，关闭触发确保缓存可控。
            env.last_broadcast_leader_seq[0, n-1] = 7
            env._cooldown_counter.fill_(10)
            env.step(torch.zeros(1, n, 2, device=config.DEVICE))
            self.assertEqual(env.leader_est_seq[0,n].item(), 7)
        for a, b in zip(*selected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class TestPhase2Actor(unittest.TestCase):
    def test_parser_uses_only_valid_bit_and_actor_permutation(self):
        k = config.MAX_NEIGHBORS
        actor = DecentralizedActor(hidden_dim=16).eval()
        flat = torch.zeros(3, 10 + 7 * k)
        slots = flat[:, 10:].reshape(3, k, 7)
        slots[:, :, :6].normal_()
        slots[:, 0] = 0
        slots[:, 0, 6] = 1  # 有效的全零载荷
        slots[:, 2::2, 6] = 1
        local, role, data, mask = _parse_flat_state(flat)
        self.assertEqual((local.shape[-1], role.shape[-1], data.shape[-1]), (7, 3, 6))
        self.assertTrue(torch.equal(mask, slots[:, :, 6] <= 0.5))
        flipped = torch.cat((flat[:, :10], slots.flip(1).flatten(1)), dim=1)
        for a, b in zip(actor.compute_action_params(flat), actor.compute_action_params(flipped)):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        self.assertFalse(any('pos_embed' in key for key in actor.state_dict()))

    def test_all_empty_projection_and_gradients(self):
        encoder = LightweightAttentionEncoder(6, 16, 2, 2, dropout=0.)
        with torch.no_grad():
            encoder.output_proj[-1].bias.fill_(1.5)
        x = torch.randn(2, config.MAX_NEIGHBORS, 6, requires_grad=True)
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        mask[1, 0] = False
        softmax = torch.nn.functional.softmax

        def finite_softmax(*args, **kwargs):
            weights = softmax(*args, **kwargs)
            self.assertTrue(torch.isfinite(weights).all())
            return weights

        with patch('mas_cc.networks.F.softmax', side_effect=finite_softmax):
            out = encoder(x, mask)
        self.assertTrue(torch.equal(out[0], torch.zeros_like(out[0])))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.equal(x.grad[0], torch.zeros_like(x.grad[0])))
        for p in encoder.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_checkpoint_roundtrip_and_schema_rejection(self):
        topology = CommunicationTopology(config.NUM_FOLLOWERS)
        inputs = torch.randn(1, 4, config.STATE_DIM, device=config.DEVICE)
        inputs[..., 16::7] = 1
        with TemporaryDirectory() as tmp, patch('mas_cc.agent.CTDEReplayBuffer'):
            path = Path(tmp) / 'ckpt.pt'
            for algorithm, cls in [('MAPPO', CTDEMAPPOAgent), ('MASAC', CTDESACAgent)]:
                with self.subTest(algorithm=algorithm):
                    agent = cls(topology, use_amp=False)
                    expected = ActorOnlyPolicy(agent.actor).select_action(inputs)
                    agent.save(path)
                    full = torch.load(path, map_location='cpu')
                    for name, value in {
                        'algorithm': algorithm, 'obs_version': 2, 'top_k': config.MAX_NEIGHBORS,
                        'state_dim': config.STATE_DIM, 'hidden_dim': config.HIDDEN_DIM,
                        'num_heads': config.NUM_ATTENTION_HEADS, 'num_layers': config.NUM_TRANSFORMER_LAYERS,
                    }.items():
                        self.assertEqual(full[name], value)
                    with torch.no_grad():
                        agent.actor.v_mean.bias.add_(1.)
                    agent.load(path)
                    torch.testing.assert_close(ActorOnlyPolicy(agent.actor).select_action(inputs), expected)
                    for name, bad in [('top_k', 0), ('obs_version', 3), ('algorithm', 'unknown'),
                                      ('algorithm', 'MAPPO' if algorithm == 'MASAC' else 'MASAC'),
                                      ('hidden_dim', 1), ('num_heads', 1), ('num_layers', 0),
                                      ('state_dim', 1)]:
                        torch.save(dict(full, **{name: bad}), path)
                        for loader in [ActorOnlyPolicy.from_checkpoint, agent.load]:
                            with self.subTest(field=name, loader=loader), self.assertRaisesRegex(ValueError, name):
                                loader(path)
                    agent.save_actor(path)
                    wrapped = torch.load(path, map_location='cpu')
                    self.assertEqual(wrapped['algorithm'], algorithm)
                    self.assertEqual(set(wrapped), set(full) & {
                        'algorithm', 'obs_version', 'top_k', 'state_dim', 'hidden_dim',
                        'num_heads', 'num_layers', 'actor'})
                    for payload in [full, wrapped, agent.actor.state_dict()]:
                        torch.save(payload, path)
                        torch.testing.assert_close(ActorOnlyPolicy.from_checkpoint(path).select_action(inputs), expected)

    def test_rejects_old_and_corrupted_actor_weights(self):
        weights = DecentralizedActor().state_dict()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.pt'
            for bad in [
                {k: v for k, v in weights.items() if k != '_obs_version'},
                dict(weights, _top_k=torch.tensor(config.MAX_NEIGHBORS + 1)),
                dict(weights, _obs_version=torch.tensor(7)),
                dict(weights, **{'neighbor_encoder.pos_embed': torch.ones(1)}),
                {'actor': weights},  # 版本化的封装必须有完整 schema
            ]:
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    ActorOnlyPolicy.from_checkpoint(path)
            torch.save(weights, path)
            with self.assertRaises(ValueError):
                ActorOnlyPolicy.from_checkpoint(path, strict=False)


class TestPhase2Collection(unittest.TestCase):
    def test_default_300_frame_boundary_and_rollout_flushes(self):
        env = BatchedModelFreeEnv(CommunicationTopology(2, num_pinned=1), 1)
        agent = CTDEMAPPOAgent(env.topology)
        session = agent.start_episode(env)
        flushed = []
        for frame in range(1, 301):
            session.step_collect()
            if frame in (1, 300):
                size = len(session.buffer)
                with patch.object(agent.actor_optimizer, "step") as actor_step, patch.object(
                    agent.value_optimizer, "step"
                ) as value_step:
                    with self.assertRaises((ValueError, RuntimeError)):
                        session.flush_update(is_episode_end=frame == 1)
                    if frame == 300:
                        with self.assertRaises((ValueError, RuntimeError)):
                            session.step_collect()
                    self.assertEqual(len(session.buffer), size)
                    actor_step.assert_not_called()
                    value_step.assert_not_called()
            if session.buffer.is_full() or frame == 300:
                flushed.append((frame, len(session.buffer)))
                session.flush_update(is_episode_end=frame == 300)
        self.assertEqual(flushed, [(128, 128), (256, 128), (300, 44)])
        session.close()

    def test_simultaneous_bind_to_different_environments(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        env_a, env_b = [BatchedModelFreeEnv(CommunicationTopology(10), 2, max_steps=1) for _ in range(2)]
        agent = CTDEMAPPOAgent(env_a.topology)
        entered, release = Event(), Event()
        snapshot = env_a._get_state_optimized

        def hold_first_snapshot():
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("Timed out while binding the first episode")
            return snapshot()

        with patch.object(env_a, "_get_state_optimized", side_effect=hold_first_snapshot):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(agent.start_episode, env_a)
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    second = pool.submit(agent.start_episode, env_b)
                    with self.assertRaises(RuntimeError):
                        second.result(timeout=1)
                finally:
                    release.set()
                session = first.result(timeout=5)
        session.step_collect()
        session.flush_update(is_episode_end=True)
        session.close()

    def test_session_starts_only_on_full_reset_and_empty_fixed_buffer(self):
        graph = CommunicationTopology(10, num_pinned=2)
        env = BatchedModelFreeEnv(graph, 2)
        agent = agents.CTDEMAPPOAgent(graph)
        env.reset([1])
        with self.assertRaises((RuntimeError, ValueError)):
            agent.start_episode(env)
        s = env.reset()
        g = env.get_global_state()
        a, lp, v = agent.act(s, g)
        z = torch.zeros(2, device=DEVICE)
        agent.store_rollout_step(s, g, a, lp, v, z, z.bool())
        with self.assertRaises((RuntimeError, ValueError)):
            agent.start_episode(env)
        self.assertEqual(len(agent.buffer), 1)
        agent.update(g, z.bool())
        session = agent.start_episode(env)
        with self.assertRaises(RuntimeError):
            agent.buffer.add(s[:, 1:], g, a, lp, v, z, z.bool())
        self.assertEqual(len(session.buffer), 0)

    def test_collection_holds_environment_and_graph_through_saving_and_update(self):

        for operation in ("collect", "flush"):
            for mutation in ("reset", "randomize_on_thread"):
                with self.subTest(operation=operation, mutation=mutation):
                    graph = CommunicationTopology(10, num_pinned=2)
                    env = BatchedModelFreeEnv(graph, 2, max_steps=2 if operation == "collect" else 1)
                    agent = agents.CTDEMAPPOAgent(graph)
                    session = agent.start_episode(env)
                    session.step_collect()
                    before = (env.frame_rev, env.episode_gen, env.graph_rev, graph.revision)

                    def mutate(mod, args):
                        if mutation == "reset":
                            env.reset()
                        else:
                            rejected = []
                            def change_graph():
                                try:
                                    graph.randomize()
                                except RuntimeError as err:
                                    rejected.append(err)
                            worker = Thread(target=change_graph)
                            worker.start()
                            worker.join(timeout=2)
                            self.assertFalse(worker.is_alive(), "graph writer must fail without blocking")
                            if rejected:
                                raise rejected[0]

                    with agent.value_net.register_forward_pre_hook(mutate), patch.object(
                        agent.actor_optimizer, "step"
                    ) as actor_step, patch.object(agent.value_optimizer, "step") as value_step:
                        with self.assertRaises(RuntimeError):
                            if operation == "collect":
                                session.step_collect()
                            else:
                                session.flush_update(is_episode_end=True)
                        actor_step.assert_not_called()
                        value_step.assert_not_called()
                    self.assertEqual(len(session.buffer), 1)
                    self.assertEqual((env.frame_rev, env.episode_gen, env.graph_rev, graph.revision), before)
                    # 拒绝发生在变更之前，去掉钩子后可以继续更新同一份 rollout。
                    if operation == "collect":
                        session.step_collect()
                    session.flush_update(is_episode_end=True)
                    session.close()

    def setUp(self):
        torch.manual_seed(21)
        self.addCleanup(patch.stopall)
        patch.object(agents, "HIDDEN_DIM", 16).start()
        patch.object(agents, "PPO_EPOCHS", 1).start()

    def test_scales_ownership_optimizer_continuity_and_successor_bootstrap(self):
        graphs = {n: CommunicationTopology(n, num_pinned=2) for n in (10, 20)}
        envs = {n: BatchedModelFreeEnv(g, 2, max_steps=2) for n, g in graphs.items()}
        agent = agents.CTDEMAPPOAgent(graphs[10])
        modules = (agent.actor, agent.value_net, agent.actor_optimizer, agent.value_optimizer)
        params = [agent.actor.v_mean.bias, agent.value_net.v_net[-1].bias]
        moments = []
        for episode, n in enumerate((10, 20, 10)):
            env = envs[n]
            env.reset()
            session = agent.start_episode(env)
            with self.assertRaises((RuntimeError, ValueError)):
                agent.start_episode(env)
            with self.assertRaises((RuntimeError, ValueError)):
                agents.CTDEMAPPOAgent(graphs[n]).start_episode(env)
            for end in (False, True):
                session.step_collect()
                with self.assertRaises((RuntimeError, ValueError)):
                    session.close()
                agent.value_net.eval()
                with torch.no_grad():
                    successor = agent.value_net(env.get_global_state()).flatten()
                with patch.object(agent, "_compute_gae", wraps=agent._compute_gae) as gae:
                    session.flush_update(is_episode_end=end)
                kwargs = gae.call_args.kwargs
                torch.testing.assert_close(kwargs["last_value"], successor)
                self.assertEqual(kwargs["truncated"][-1].tolist(), [end] * 2)
                if end:
                    torch.testing.assert_close(kwargs["truncated_next_values"][-1], successor)
                self.assertEqual(len(session.buffer), 0)
                if not end:
                    with self.assertRaises((RuntimeError, ValueError)):
                        session.close()
            session.close()
            self.assertEqual(modules, (agent.actor, agent.value_net, agent.actor_optimizer, agent.value_optimizer))
            states = [opt.state[p] for opt, p in zip(modules[2:], params)]
            self.assertEqual([s["step"].item() for s in states], [2 * (episode + 1)] * 2)
            if moments:
                for s, prev in zip(states, moments):
                    self.assertIs(s["exp_avg"], prev)
            moments = [s["exp_avg"] for s in states]

    def test_collection_rejects_stale_frame_or_graph_before_writing(self):
        for mutation in ("step", "partial_reset", "reset", "randomize", "precompute"):
            with self.subTest(mutation=mutation):
                graph = CommunicationTopology(10, num_pinned=2)
                env = BatchedModelFreeEnv(graph, 2)
                agent = agents.CTDEMAPPOAgent(graph)
                session = agent.start_episode(env)
                session.step_collect()
                with self.assertRaises((RuntimeError, ValueError)):
                    agent.update(next_global_states=env.get_global_state())
                s, g = env._get_state_optimized(), env.get_global_state()
                a, lp, v = agent.act(s, g)
                z = torch.zeros(2, device=DEVICE)
                with self.assertRaises((RuntimeError, ValueError)):
                    agent.store_rollout_step(s, g, a, lp, v, z, z.bool())
                if mutation == "step":
                    env.step(a)
                elif mutation == "partial_reset":
                    env.reset([1])
                elif mutation == "reset":
                    env.reset()
                elif mutation == "randomize":
                    graph.randomize()
                else:
                    env._precompute_neighbor_indices(verbose=False)
                with patch.object(agent.actor_optimizer, "step") as actor_step, patch.object(
                    agent.value_optimizer, "step"
                ) as value_step:
                    for attempt in (session.step_collect, session.flush_update):
                        with self.assertRaises((RuntimeError, ValueError)):
                            attempt()
                        self.assertEqual(len(session.buffer), 1)
                    actor_step.assert_not_called()
                    value_step.assert_not_called()

    def test_sac_rejects_multiscale_before_allocating_replay(self):
        with self.assertRaisesRegex(ValueError, "(?i)scale|fixed|规模"):
            agents.CTDESACAgent(CommunicationTopology(20), train_follower_scales=(10, 20))

    def test_actor_microbatch_matches_logical_minibatches_and_kl_stop(self):
        graph = CommunicationTopology(2, num_pinned=1)
        for drift, expected_batches in ((0.0, 3), (1.0, 1)):
            with self.subTest(drift=drift):
                full = agents.CTDEMAPPOAgent(graph, actor_microbatch_max_rows=None)
                split = agents.CTDEMAPPOAgent(graph, actor_microbatch_max_rows=380)
                split.actor.load_state_dict(full.actor.state_dict())
                split.value_net.load_state_dict(full.value_net.state_dict())
                # 2112 = 2*1024 + 64 条环境记录。只缩小网络宽度以限制 CPU 回归成本。
                e = 2112
                s = torch.randn(e, 3, STATE_DIM, device=DEVICE)
                s[..., 16::7] = (s[..., 16::7] > 0).float()
                g = torch.randn(e, full.expected_global_state_dim, device=DEVICE)
                a, lp, v, pre = full.act(s, g, return_pre_squash=True)
                r = torch.randn(e, device=DEVICE)
                done = torch.zeros(e, device=DEVICE, dtype=torch.bool)
                snapshots = []
                reports = []
                for learner, cap in ((full, None), (split, 380)):
                    learner.store_rollout_step(s, g, a, lp + drift, v, r, done, pre_squash_v=pre)
                    batches = []
                    first_step = []
                    def record_gradient(opt, args, kwargs):
                        if not first_step:
                            first_step.append([p.grad.clone() for p in learner.actor.parameters()])
                    def record_weights(opt, args, kwargs):
                        if len(first_step) == 1:
                            first_step.append([p.detach().clone() for p in learner.actor.parameters()])
                    learner.actor_optimizer.register_step_pre_hook(record_gradient)
                    learner.actor_optimizer.register_step_post_hook(record_weights)
                    learner.value_net.register_forward_pre_hook(lambda mod, args: batches.append(len(args[0])))
                    torch.manual_seed(567)
                    with patch.object(learner.actor, "evaluate_actions", wraps=learner.actor.evaluate_actions) as ev:
                        report = learner.update(g, done)
                    row_counts = [len(c.args[0]) for c in ev.call_args_list]
                    if cap:
                        self.assertLessEqual(max(row_counts), cap)
                        self.assertTrue(all(rows % 2 == 0 for rows in row_counts))
                    self.assertEqual(batches, [2112] + [1024, 1024, 64][:expected_batches])
                    self.assertEqual(sum(row_counts), sum(batches[1:]) * 2)
                    self.assertEqual(report["update_minibatches"], expected_batches)
                    self.assertEqual(report["early_stopped"], bool(drift))
                    snapshots.append(first_step)
                    reports.append(report)
                for ref_group, actual_group in zip(*snapshots):
                    for ref, actual in zip(ref_group, actual_group):
                        torch.testing.assert_close(ref, actual, atol=2e-6, rtol=2e-4)
                for key in ("kl", "entropy", "clipfrac", "policy", "value"):
                    self.assertAlmostEqual(reports[0][key], reports[1][key], delta=3e-5)


class TestTrainingAndTrajectory(unittest.TestCase):
    def test_actor_only_history_and_graph_snapshots_across_scales(self):
        policy = ActorOnlyPolicy(DecentralizedActor(hidden_dim=16))
        for n in (5, 11):
            env = ModelFreeEnv(CommunicationTopology(n))
            positions, velocities = [], []
            def snapshot():
                positions.append(env.positions[1:].detach().cpu().numpy().copy())
                velocities.append(env.velocities[1:].detach().cpu().numpy().copy())

            reset, step = env.reset, env.step
            def recorded_reset():
                state = reset()
                snapshot()
                return state

            def recorded_step(action):
                result = step(action)
                snapshot()
                return result

            with patch.object(env, 'reset', recorded_reset), patch.object(env, 'step', recorded_step):
                traj = collect_trajectory(policy, env, 4)
            np.testing.assert_array_equal(traj['follower_pos'], positions)
            np.testing.assert_array_equal(traj['follower_vel'], velocities)
            self.assertEqual(traj['num_followers'], n)
            self.assertEqual(traj['pinned_ids'], list(env.topology.pinned_followers))
            self.assertEqual(traj['topology_stats']['max_depth'], env.topology.max_depth)
            old_depths = list(traj['topology_stats']['node_depths'])
            with patch('mas_cc.utils.MAX_STEPS', 3):
                metrics = evaluate_agent(policy, env, num_episodes=2)
            self.assertTrue(np.isfinite(metrics['mean_reward']))
            self.assertEqual(old_depths, list(traj['topology_stats']['node_depths']))

    def test_train_reuses_three_envs_and_isolates_evaluation_graphs(self):
        import train as training

        starts, eval_graphs = [], []
        collect = training.collect_trajectory
        def record_trajectory(policy, env, max_steps):
            self.assertIsInstance(policy, agents.ActorOnlyPolicy)
            eval_graphs.append(env.topology)
            return collect(policy, env, max_steps)

        start = getattr(agents.CTDEMAPPOAgent, 'start_episode', None)
        def record_start(agent, env):
            starts.append((env, env.episode_gen, env.topology.pinned_mode, env.topology.depth_mode))
            return start(agent, env)

        with TemporaryDirectory() as directory, ExitStack() as stack:
            best = Path(directory) / 'best.pt'
            stack.enter_context(patch.object(training, 'SAVE_MODEL_PATH', str(best)))
            stack.enter_context(patch.object(training, 'NUM_PARALLEL_ENVS', 2))
            stack.enter_context(patch.object(training, 'MAX_STEPS', 2))
            stack.enter_context(patch.object(agents, 'PPO_EPOCHS', 1))
            stack.enter_context(patch.object(agents, 'HIDDEN_DIM', 16))
            stack.enter_context(patch.object(agents.CTDEMAPPOAgent, 'start_episode', record_start, create=True))
            stack.enter_context(patch.object(training, 'collect_trajectory', record_trajectory))
            ui = stack.enter_context(patch('mas_cc.dashboard.TrainingDashboard'))
            agent, topology, dashboard = training.train(num_episodes=6, train_follower_scales=(10, 20, 30),
                topology_mix=(0., 0., 1.), vis_interval=1, show_dashboard=True)
            self.assertIs(dashboard, ui.return_value)
            dash_graph = ui.call_args.kwargs['topology']
            self.assertEqual(len(starts), 6)
            self.assertEqual(len(set(item[0] for item in starts)), 3)
            for offset in (0, 3):
                self.assertEqual(sorted(env.num_followers for env, *_ in starts[offset:offset + 3]), [10, 20, 30])
                self.assertTrue(all(gen == offset // 3 + 1 for _, gen, *_ in starts[offset:offset + 3]))
            self.assertTrue(all((p, d) == ('low', 'deep') for _, _, p, d in starts))
            train_graphs = {env.topology for env, *_ in starts}
            self.assertTrue(eval_graphs)
            self.assertTrue(train_graphs.isdisjoint(eval_graphs))
            self.assertNotIn(dash_graph, train_graphs)
            self.assertNotIn(dash_graph, eval_graphs)
            self.assertEqual(topology.num_followers, starts[-1][0].num_followers)
            self.assertTrue(best.is_file())
            final = Path(directory) / 'final_actor.pt'
            agents.ActorOnlyPolicy.from_checkpoint(str(best))
            loaded = agents.ActorOnlyPolicy.from_checkpoint(str(final))
            for key, value in agent.actor.state_dict().items():
                torch.testing.assert_close(value, loaded.actor.state_dict()[key], rtol=0, atol=0)
