"""针对已知问题修复的回归测试套件（覆盖算法不变量、边界条件与工程闭环）。"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import torch

from mas_cc.agent import ActorOnlyPolicy, CTDEMAPPOAgent, CTDESACAgent
from mas_cc.config import (
    ACTION_DIM,
    DEVICE,
    GAMMA,
    NUM_FOLLOWERS,
    NUM_PINNED,
    PPO_EPOCHS,
    STATE_DIM,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    set_seed,
)
from mas_cc.environment import BatchedModelFreeEnv, ModelFreeEnv
from mas_cc.networks import CentralizedCritic, CentralizedValue, DecentralizedActor
from mas_cc.topology import CommunicationTopology
from mas_cc.utils import collect_trajectory, evaluate_agent
from test_env import _raw_threshold_from_env_threshold


class TestKnownFixes(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_actor_init_weights_not_collapsed(self):
        actor = DecentralizedActor().to(DEVICE)
        # 隐藏层权重采用 sqrt(2) 正交初始化，不应被压到 0.01
        hidden_std = float(actor.local_encoder[0].weight.std().item())
        self.assertGreater(hidden_std, 0.05)

        # 速度头仍保持小增益 0.01
        v_head_norm = float(actor.v_mean.weight.norm().item())
        self.assertLess(v_head_norm, 0.05)

        # 通信头初始均值约 0.70（高通信率起步）
        dummy_state = torch.zeros(4, STATE_DIM, device=DEVICE)
        _, _, alpha, beta = actor.compute_action_params(dummy_state)
        init_mean = float((alpha / (alpha + beta)).mean().item())
        self.assertAlmostEqual(init_mean, 0.70, delta=0.05)

    def test_evaluate_actions_return_entropy_and_independent_gradients(self):
        actor = DecentralizedActor().to(DEVICE)
        actor.eval()
        states = torch.randn(16, STATE_DIM, device=DEVICE) * 0.2
        actions, logp_act = actor(states, deterministic=False)

        # 默认返回签名兼容
        logp_only = actor.evaluate_actions(states, actions)
        self.assertEqual(tuple(logp_only.shape), (16, 1))
        self.assertTrue(torch.allclose(logp_only, logp_act, atol=1e-4))

        # 单独对解析熵反向传播，验证 Normal 速度头与 Beta 通信头均收到非零梯度
        actor.zero_grad(set_to_none=True)
        logp, entropy = actor.evaluate_actions(states, actions, return_entropy=True)
        self.assertEqual(tuple(logp.shape), (16, 1))
        self.assertEqual(tuple(entropy.shape), (16, 1))
        self.assertTrue(torch.isfinite(entropy).all())

        (-entropy.mean()).backward()
        self.assertIsNotNone(actor.v_log_std.weight.grad)
        self.assertGreater(float(actor.v_log_std.weight.grad.norm().item()), 0.0)
        self.assertIsNotNone(actor.comm_alpha.weight.grad)
        self.assertGreater(float(actor.comm_alpha.weight.grad.norm().item()), 0.0)
        self.assertIsNotNone(actor.comm_beta.weight.grad)
        self.assertGreater(float(actor.comm_beta.weight.grad.norm().item()), 0.0)

    def test_mappo_dropout_and_saturation_preserve_ratio_one_and_kl_zero(self):
        """默认方差与高方差饱和(log_std=1.5)下，真实训练模式+非空邻居输入跑真实 update()，参数不变时 ratio==1、KL==0、ClipFrac==0。"""
        for log_std_bias in (0.0, 1.5):
            with self.subTest(log_std_bias=log_std_bias):
                set_seed(42)
                topo = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
                env = BatchedModelFreeEnv(topo, num_envs=4)
                agent = CTDEMAPPOAgent(topo)
                if log_std_bias != 0.0:
                    with torch.no_grad():
                        agent.actor.v_log_std.bias.fill_(log_std_bias)
                agent.actor.train()

                local_s = env.reset()
                global_s = env.get_global_state()
                self.assertGreater(float(local_s[:, 1:, 9:].abs().sum().item()), 0.0)

                actions, old_logp, values, pre_v = agent.act(
                    local_s, global_s, deterministic=False, return_pre_squash=True
                )
                self.assertIsNotNone(pre_v)
                if log_std_bias > 0.0:
                    # 确认高方差下确实产生了接近 tanh 饱和区的速度动作
                    self.assertGreater(float((actions[:, :, 0].abs() > 0.999).float().mean().item()), 0.1)

                next_local_s, rewards, dones, _ = env.step(actions)
                next_global_s = env.get_global_state()
                agent.store_rollout_step(
                    local_s, global_s, actions, old_logp, values, rewards, dones, pre_squash_v=pre_v
                )

                # 先直接核对逐元素 ratio 精度
                fs_flat = local_s[:, 1:, :].reshape(-1, STATE_DIM)
                act_flat = actions.reshape(-1, ACTION_DIM)
                agent._disable_actor_dropout()
                new_logp_flat, _ = agent.actor.evaluate_actions(
                    fs_flat, act_flat, return_entropy=True, pre_squash_v=pre_v.reshape(-1, 1)
                )
                ratio = torch.exp(new_logp_flat.view(4, NUM_FOLLOWERS, 1) - old_logp)
                self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4))

                # 再走真实 update() 公开入口（临时 lr=0 锁定参数不变，验证首 minibatch KL==0 且 ClipFrac==0）
                orig_lr = agent.actor_optimizer.param_groups[0]["lr"]
                agent.actor_optimizer.param_groups[0]["lr"] = 0.0
                try:
                    losses = agent.update(next_global_states=next_global_s, next_dones=dones)
                finally:
                    agent.actor_optimizer.param_groups[0]["lr"] = orig_lr

                self.assertLess(abs(losses["kl"]), 1e-6)
                self.assertEqual(losses["clipfrac"], 0.0)
                self.assertFalse(losses["early_stopped"])

                # 确认关闭 Dropout 后参数梯度仍正常回传
                agent.actor_optimizer.zero_grad(set_to_none=True)
                new_logp_flat2, _ = agent.actor.evaluate_actions(
                    fs_flat, act_flat, return_entropy=True, pre_squash_v=pre_v.reshape(-1, 1)
                )
                new_logp_flat2.mean().backward()
                grad_norm = sum(p.grad.norm().item() for p in agent.actor.parameters() if p.grad is not None)
                self.assertGreater(grad_norm, 0.0)

    def test_mappo_pre_squash_mixed_batch_and_interleaved_equal_actions(self):
        """验证：(1) 混合批次中缺值步不会破坏有值步的精确 ratio；(2) 同值不同采样的交错调用不会串用 pre_squash_v。"""
        # --- (1) 混合批次 + 中间穿插确定性 select_action ---
        set_seed(42)
        topo = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
        env = BatchedModelFreeEnv(topo, num_envs=4)
        agent = CTDEMAPPOAgent(topo)
        with torch.no_grad():
            agent.actor.v_log_std.bias.fill_(1.5)
        agent.actor.train()

        s1 = env.reset()
        g1 = env.get_global_state()
        a1, lp1, v1 = agent.act(s1, g1, deterministic=False)
        # 在 act() 与 store_rollout_step() 之间插入确定性推理，不应破坏 a1 的绑定
        _ = agent.select_action(s1, deterministic=True)
        ns1, r1, d1, _ = env.step(a1)
        ng1 = env.get_global_state()
        agent.store_rollout_step(s1, g1, a1, lp1, v1, r1, d1)

        # 第二步故意传入 clone()（失去对象身份且不传 pre_squash_v），构造混合 [True, False] 批次
        a2, lp2, v2 = agent.act(ns1, ng1, deterministic=False)
        ns2, r2, d2, _ = env.step(a2)
        ng2 = env.get_global_state()
        agent.store_rollout_step(ns1, ng1, a2.clone(), lp2, v2, r2, d2)

        self.assertTrue(bool(agent.buffer.has_pre_squash_v[0].all().item()))
        self.assertFalse(bool(agent.buffer.has_pre_squash_v[1].any().item()))

        # 按记录逐条验证：第一步记录经 evaluate_actions(pre_squash_v, has_pre_squash_v) 必须精确 ratio==1
        fs_all = torch.stack(agent.buffer.follower_states, dim=0).reshape(-1, STATE_DIM)
        act_all = torch.stack(agent.buffer.actions, dim=0).reshape(-1, ACTION_DIM)
        pre_all = torch.stack(agent.buffer.pre_squash_v, dim=0).reshape(-1, 1)
        has_all = torch.stack(agent.buffer.has_pre_squash_v, dim=0).reshape(-1, 1)
        old_lp_all = torch.stack(agent.buffer.logp, dim=0).reshape(-1, 1)

        agent._disable_actor_dropout()
        new_lp_all, _ = agent.actor.evaluate_actions(
            fs_all, act_all, return_entropy=True, pre_squash_v=pre_all, has_pre_squash_v=has_all
        )
        n_step1 = 4 * NUM_FOLLOWERS
        ratio_step1 = torch.exp(new_lp_all[:n_step1] - old_lp_all[:n_step1])
        self.assertTrue(torch.allclose(ratio_step1, torch.ones_like(ratio_step1), atol=1e-4))
        agent.buffer.clear()

        # --- (2) 交错采样产生完全相同动作值但不同原始速度：对象身份绑定绝不串用 ---
        topo_s = CommunicationTopology(1, num_pinned=1)
        agent_s = CTDEMAPPOAgent(topo_s)
        with torch.no_grad():
            for p in agent_s.actor.parameters():
                p.zero_()
            agent_s.actor.v_log_std.bias.fill_(1.5)
            agent_s.actor.comm_alpha.bias.fill_(2.0)
            agent_s.actor.comm_beta.bias.fill_(-1.0)
            # 让 v_mean 对 state 线性敏感，从而不同状态产生不同 pre-squash 速度但 tanh 后均饱和到 +1.0
            agent_s.actor.local_encoder[0].weight.fill_(1.0)
            agent_s.actor.fusion[0].weight.fill_(1.0)
            agent_s.actor.shared[0].weight.fill_(1.0)
            agent_s.actor.v_mean.weight.fill_(1.0)

        st_a = torch.zeros(1, 2, STATE_DIM, device=DEVICE)
        st_b = torch.zeros(1, 2, STATE_DIM, device=DEVICE)
        st_b[:, 1, 0] = 0.5
        gs_dummy = torch.zeros(1, agent_s.value_net.ctx_dim + 2 * agent_s.value_net.agent_feat_dim, device=DEVICE)

        # 两次高均值采样：动作速度均饱和到 1.0，但 v_pre 不同
        with torch.no_grad():
            agent_s.actor.v_mean.bias.fill_(11.5)
        set_seed(777)
        act_a, lp_a, val_a = agent_s.act(st_a, gs_dummy, deterministic=False)

        with torch.no_grad():
            agent_s.actor.v_mean.bias.fill_(18.5)
        set_seed(777)
        act_b, lp_b, val_b = agent_s.act(st_b, gs_dummy, deterministic=False)

        # 确认两份动作值完全相等（复现审计反例）
        self.assertTrue(torch.equal(act_a, act_b))

        # 恢复第一份的参数并存入 act_a（此时 act_b 是最近一次采样），验证未串用 act_b 的 18.5
        with torch.no_grad():
            agent_s.actor.v_mean.bias.fill_(11.5)
        agent_s.store_rollout_step(
            st_a, gs_dummy, act_a, lp_a, val_a,
            torch.zeros(1, device=DEVICE), torch.zeros(1, dtype=torch.bool, device=DEVICE),
        )
        stored_pre = float(agent_s.buffer.pre_squash_v[0].item())
        self.assertLess(abs(stored_pre - 11.5), 5.0)
        self.assertNotAlmostEqual(stored_pre, 18.5, delta=2.0)

        orig_lr = agent_s.actor_optimizer.param_groups[0]["lr"]
        agent_s.actor_optimizer.param_groups[0]["lr"] = 0.0
        try:
            losses_s = agent_s.update(next_global_states=gs_dummy, next_dones=torch.zeros(1, dtype=torch.bool, device=DEVICE))
        finally:
            agent_s.actor_optimizer.param_groups[0]["lr"] = orig_lr
        self.assertLess(abs(losses_s["kl"]), 1e-6)
        self.assertEqual(losses_s["clipfrac"], 0.0)

    def test_mappo_per_agent_logp_and_update(self):
        topo = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
        env = BatchedModelFreeEnv(topo, num_envs=4)
        agent = CTDEMAPPOAgent(topo)

        local_s = env.reset()
        global_s = env.get_global_state()
        for step in range(8):
            actions, logp, values = agent.act(local_s, global_s, deterministic=False)
            self.assertEqual(tuple(logp.shape), (4, NUM_FOLLOWERS, 1))
            next_local_s, rewards, dones, _ = env.step(actions)
            next_global_s = env.get_global_state()
            truncated = torch.full_like(dones, step == 7, dtype=torch.bool)
            agent.store_rollout_step(
                local_s,
                global_s,
                actions,
                logp,
                values,
                rewards,
                dones,
                truncated=truncated,
                next_global_states=next_global_s if truncated.any() else None,
            )
            local_s, global_s = next_local_s, next_global_s

        losses = agent.update(next_global_states=global_s, next_dones=dones)
        for key in ("policy", "value", "entropy", "entropy_joint", "kl", "clipfrac"):
            self.assertIn(key, losses)
            self.assertTrue(math.isfinite(losses[key]), f"{key} is not finite: {losses[key]}")
        self.assertGreaterEqual(losses["kl"], -1e-6)
        self.assertAlmostEqual(losses["entropy_joint"], losses["entropy"] * NUM_FOLLOWERS, places=4)

    def test_mappo_kl_early_stop_breaks_outer_epoch_loop(self):
        topo = CommunicationTopology(6, num_pinned=2)
        env = BatchedModelFreeEnv(topo, num_envs=4)
        agent = CTDEMAPPOAgent(topo)

        local_s = env.reset()
        global_s = env.get_global_state()
        for _ in range(16):
            actions, logp, values = agent.act(local_s, global_s, deterministic=False)
            next_local_s, rewards, dones, _ = env.step(actions)
            next_global_s = env.get_global_state()
            agent.store_rollout_step(local_s, global_s, actions, logp, values, rewards, dones)
            local_s, global_s = next_local_s, next_global_s

        # 真实扰动策略参数制造策略偏移，触发 k3 KL 早停并中断内外双层循环
        with torch.no_grad():
            agent.actor.v_mean.bias.add_(2.0)

        losses = agent.update(next_global_states=global_s, next_dones=dones)
        self.assertTrue(losses["early_stopped"])
        self.assertEqual(losses["update_minibatches"], 1)
        self.assertLess(losses["update_minibatches"], PPO_EPOCHS)

    def test_gae_truncation_mid_rollout_single_step_and_done_priority(self):
        topo = CommunicationTopology(6, num_pinned=2)
        agent = CTDEMAPPOAgent(topo)

        # 1) 跨 episode 异步截断：env0 在 t=1 截断，env1 在 t=3 截断；
        #    截断前 V(s_t)=2.0，截断后真实 V(s_{t+1})=10.0，下一 episode 初态 V=99.0
        T, E = 4, 2
        rewards = torch.zeros(T, E, device=DEVICE)
        values = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 3.0],  # env0 在 t=1 截断
                [99.0, 4.0],  # env0 进入新 episode
                [5.0, 2.0],  # env1 在 t=3 截断
            ],
            device=DEVICE,
        )
        dones = torch.zeros(T, E, dtype=torch.bool, device=DEVICE)
        truncated = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, False],
                [False, True],
            ],
            dtype=torch.bool,
            device=DEVICE,
        )
        trunc_nv = torch.zeros(T, E, device=DEVICE)
        trunc_nv[1, 0] = 10.0
        trunc_nv[3, 1] = 12.0
        has_trunc_nv = truncated.clone()
        last_value = torch.tensor([6.0, 88.0], device=DEVICE)

        adv, ret = agent._compute_gae(
            rewards,
            dones,
            values,
            last_value,
            truncated=truncated,
            truncated_next_values=trunc_nv,
            has_truncated_next_values=has_trunc_nv,
        )
        # env0 在 t=1 应使用 trunc_nv=10.0 而非 values[1]=2.0 或新 episode 的 99.0
        self.assertAlmostEqual(float(ret[1, 0].item()), GAMMA * 10.0, places=4)
        # env1 在 t=3 应使用存储的 trunc_nv=12.0（即使 last_value 被重置后状态污染成 88.0）
        self.assertAlmostEqual(float(ret[3, 1].item()), GAMMA * 12.0, places=4)
        # 确认 env0 新 episode(t=2,3) 的 GAE 未泄露到旧 episode(t=1)
        expected_delta_t1 = 0.0 + GAMMA * 10.0 - 2.0
        self.assertAlmostEqual(float(adv[1, 0].item()), expected_delta_t1, places=4)

        # 2) 中间截断缺少 next_value 时应显式报错，而不是静默使用错误值
        with self.assertRaises(ValueError):
            agent._compute_gae(rewards, dones, values, last_value, truncated=truncated)

        # 3) 单步 rollout (T=1) 且 done 与 truncated 同时为真：真实终止优先，不 bootstrap
        r1 = torch.tensor([[1.5, 2.0]], device=DEVICE)
        v1 = torch.tensor([[5.0, 5.0]], device=DEVICE)
        d1 = torch.tensor([[True, False]], dtype=torch.bool, device=DEVICE)
        tr1 = torch.tensor([[True, True]], dtype=torch.bool, device=DEVICE)
        lv1 = torch.tensor([10.0, 10.0], device=DEVICE)
        _, ret1 = agent._compute_gae(r1, d1, v1, lv1, truncated=tr1)
        self.assertAlmostEqual(float(ret1[0, 0].item()), 1.5, places=5)
        self.assertAlmostEqual(float(ret1[0, 1].item()), 2.0 + GAMMA * 10.0, places=5)

    def test_update_public_entry_final_truncation_missing_and_three_controls(self):
        """公开入口 store_rollout_step() -> update()：末步截断缺后继价值必抛 ValueError；提供采集态或更新态后继状态均精确 bootstrap。"""
        topo = CommunicationTopology(6, num_pinned=2)
        env = BatchedModelFreeEnv(topo, num_envs=2)
        agent = CTDEMAPPOAgent(topo)

        # 将 value_net 固定为恒等输出 10.0，便于精确校验 return = r + GAMMA * 10.0
        with torch.no_grad():
            for p in agent.value_net.parameters():
                p.zero_()
            agent.value_net.v_net[-1].bias.fill_(10.0)

        local_s = env.reset()
        global_s = env.get_global_state()
        actions, logp, values = agent.act(local_s, global_s, deterministic=False)
        next_local_s, _, _, _ = env.step(actions)
        next_global_s = env.get_global_state()
        rewards = torch.full((2,), 2.0, device=DEVICE)
        # env0: done & truncated 同真（真实终止优先）；env1: 仅 truncated（必须 bootstrap）
        dones = torch.tensor([True, False], dtype=torch.bool, device=DEVICE)
        truncated = torch.tensor([True, True], dtype=torch.bool, device=DEVICE)

        # 对照 1：采集和更新均未提供后继状态/价值 -> 必须抛 ValueError
        agent.store_rollout_step(local_s, global_s, actions, logp, values, rewards, dones, truncated=truncated)
        with self.assertRaises(ValueError):
            agent.update(next_global_states=None, next_dones=None)
        agent.buffer.clear()

        # 截获 _compute_gae 的 return 验证两个合法对照
        captured_rets = []
        orig_gae = agent._compute_gae

        def _spy_gae(*args, **kwargs):
            adv, ret = orig_gae(*args, **kwargs)
            captured_rets.append(ret.detach().clone())
            return adv, ret

        agent._compute_gae = _spy_gae
        try:
            # 对照 2：采集时提供 next_global_states，update 不传
            agent.store_rollout_step(
                local_s, global_s, actions, logp, values, rewards, dones,
                truncated=truncated, next_global_states=next_global_s,
            )
            agent.update(next_global_states=None, next_dones=None)
            ret_store = captured_rets[-1]
            self.assertAlmostEqual(float(ret_store[0, 0].item()), 2.0, places=4)
            self.assertAlmostEqual(float(ret_store[0, 1].item()), 2.0 + GAMMA * 10.0, places=4)

            # 重新锁定 value_net 输出为 10.0 再测对照 3
            with torch.no_grad():
                for p in agent.value_net.parameters():
                    p.zero_()
                agent.value_net.v_net[-1].bias.fill_(10.0)

            # 对照 3：采集时不传，update 传入 next_global_states
            agent.store_rollout_step(local_s, global_s, actions, logp, values, rewards, dones, truncated=truncated)
            agent.update(next_global_states=next_global_s, next_dones=dones)
            ret_upd = captured_rets[-1]
            self.assertAlmostEqual(float(ret_upd[0, 0].item()), 2.0, places=4)
            self.assertAlmostEqual(float(ret_upd[0, 1].item()), 2.0 + GAMMA * 10.0, places=4)
        finally:
            agent._compute_gae = orig_gae

    def test_deepsets_value_and_critic_dynamic_num_followers_and_errors(self):
        value_net = CentralizedValue().to(DEVICE)
        critic_net = CentralizedCritic().to(DEVICE)

        for n_followers in (6, 20, 50):
            topo = CommunicationTopology(n_followers, num_pinned=min(3, n_followers))
            env = BatchedModelFreeEnv(topo, num_envs=3)
            env.reset()
            gs = env.get_global_state()
            v = value_net(gs)
            self.assertEqual(tuple(v.shape), (3, 1))
            self.assertTrue(torch.isfinite(v).all())

            joint_act = torch.zeros(3, n_followers * ACTION_DIM, device=DEVICE)
            q = critic_net(gs, joint_act)
            self.assertEqual(tuple(q.shape), (3, 1))
            self.assertTrue(torch.isfinite(q).all())

            (v.mean() + q.mean()).backward()

        # 非法维度输入应明确抛出 ValueError
        bad_gs = torch.zeros(2, value_net.ctx_dim + 1, device=DEVICE)
        with self.assertRaises(ValueError):
            value_net(bad_gs)

        # 仅 1 个 agent（无 follower）应抛出 ValueError
        one_agent_gs = torch.zeros(2, value_net.ctx_dim + value_net.agent_feat_dim, device=DEVICE)
        with self.assertRaises(ValueError):
            value_net(one_agent_gs)

        # joint_action 维度与 global_state 推导的 num_followers 不匹配时应抛 ValueError
        topo6 = CommunicationTopology(6, num_pinned=2)
        env6 = BatchedModelFreeEnv(topo6, num_envs=2)
        env6.reset()
        with self.assertRaises(ValueError):
            critic_net(env6.get_global_state(), torch.zeros(2, 10 * ACTION_DIM, device=DEVICE))

    def test_actor_only_policy_evaluation_and_checkpoint_roundtrip(self):
        actor = DecentralizedActor().to(DEVICE)
        policy = ActorOnlyPolicy(actor)
        self.assertIsNone(policy.value_net)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_mappo.pt")
            torch.save({"actor": actor.state_dict()}, ckpt_path)
            reloaded = ActorOnlyPolicy.from_checkpoint(ckpt_path)
            self.assertIsNone(reloaded.value_net)

            for n_followers in (10, 40):
                topo = CommunicationTopology(n_followers, num_pinned=3)
                env = ModelFreeEnv(topo)
                traj = collect_trajectory(reloaded, env, max_steps=10)
                self.assertEqual(traj["follower_pos"].shape, (11, n_followers))

                stats = evaluate_agent(reloaded, env, num_episodes=1)
                self.assertTrue(math.isfinite(stats["mean_tracking_error"]))

    def test_env_threshold_inverse_mapping_direction(self):
        self.assertAlmostEqual(_raw_threshold_from_env_threshold(THRESHOLD_MIN), 1.0, places=6)
        self.assertAlmostEqual(_raw_threshold_from_env_threshold(THRESHOLD_MAX), 0.0, places=6)

        def _measure_comm_rate(th_env: float) -> float:
            set_seed(123)
            topo = CommunicationTopology(12, num_pinned=3, seed=123)
            env = ModelFreeEnv(topo)
            env.reset()
            theta_norm = _raw_threshold_from_env_threshold(th_env)
            rates = []
            for _ in range(40):
                action = torch.zeros(env.num_followers, 2, device=DEVICE)
                action[:, 1] = theta_norm
                _, _, _, info = env.step(action)
                rates.append(info["comm_rate"])
            return sum(rates) / len(rates)

        comm_low_th = _measure_comm_rate(THRESHOLD_MIN)
        comm_high_th = _measure_comm_rate(THRESHOLD_MAX)
        self.assertGreaterEqual(comm_low_th, comm_high_th)

    def test_masac_smoke_compatible(self):
        topo = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
        env = BatchedModelFreeEnv(topo, num_envs=4)
        agent = CTDESACAgent(topo, use_amp=False)
        local_s = env.reset()
        global_s = env.get_global_state()
        for _ in range(10):
            actions = agent.select_action(local_s, deterministic=False)
            next_local_s, rewards, dones, _ = env.step(actions)
            next_global_s = env.get_global_state()
            agent.store_transitions_batch(local_s, global_s, actions, rewards, next_local_s, next_global_s, dones)
            local_s, global_s = next_local_s, next_global_s
        losses = agent.update(batch_size=8, gradient_steps=1)
        self.assertIn("q1", losses)
        self.assertTrue(math.isfinite(losses["q1"]))

    def test_dashboard_entropy_fallback_html(self):
        from mas_cc.dashboard import TrainingDashboard

        dash = TrainingDashboard.__new__(TrainingDashboard)
        dash.total_episodes = 10
        dash.reward_good_threshold = -10.0
        dash.reward_poor_threshold = -100.0
        dash.error_good_threshold = 0.5
        dash.error_poor_threshold = 2.0
        dash.comm_good_threshold = 0.3
        dash.comm_poor_threshold = 0.7

        html_new = dash._generate_stats_html(
            1, -20.0, 0.4, 0.2, -20.0, {"policy": 0.1, "value": 0.2, "entropy": 1.23, "kl": 0.005, "clipfrac": 0.05}, 5.0
        )
        self.assertIn("H(agent)", html_new)
        self.assertIn("1.23", html_new)

        html_old = dash._generate_stats_html(
            1, -20.0, 0.4, 0.2, -20.0, {"policy": 0.1, "value": 0.2, "entropy_joint": 9.87, "kl": 0.005, "clipfrac": 0.05}, 5.0
        )
        self.assertIn("H(joint)", html_old)
        self.assertIn("9.87", html_old)


if __name__ == "__main__":
    unittest.main()
