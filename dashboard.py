"""
训练可视化仪表盘 - 动态阈值版本
"""
import time
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.max_open_warning'] = 50
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

# 🔧 导入 MAX_STEPS 用于动态计算阈值
from config import MAX_STEPS


class TrainingDashboard:
    """训练仪表盘 - 动态阈值版本"""
    
    def __init__(self, total_episodes, vis_interval=10, topology=None):
        self.total_episodes = total_episodes
        self.vis_interval = vis_interval
        self.start_time = None
        self.max_steps = MAX_STEPS
        
        # 🔧 保存拓扑信息用于区分角色
        self.topology = topology
        self.pinned_followers = topology.pinned_followers if topology else []
        
        # 🔧 动态计算阈值（基于 MAX_STEPS）
        # 每步奖励范围约 [-1.3, 0.5]（经过 soft scaling）
        # 好的奖励：误差小，每步约 -0.17（对应 tanh(0.1*2)*1 ≈ 0.2 的惩罚 + 0.03 的改进奖励）
        # 差的奖励：误差大，每步约 -0.67（对应 tanh(0.5*2)*1 ≈ 0.76 的惩罚）
        self.reward_good_threshold = -0.17 * self.max_steps  # 好：> -51 (for 300 steps)
        self.reward_poor_threshold = -0.67 * self.max_steps  # 差：< -201 (for 300 steps)
        
        # 跟踪误差阈值（每步平均值，与 MAX_STEPS 无关）
        self.error_good_threshold = 0.3
        self.error_poor_threshold = 1.0
        
        # 通信率阈值（比例值，与 MAX_STEPS 无关）
        self.comm_good_threshold = 0.3
        self.comm_poor_threshold = 0.7
        
        # 历史记录
        self.reward_history = []
        self.tracking_error_history = []
        self.comm_history = []
        self.best_reward = -float('inf')
        self.best_trajectory = None
        
        self.use_widgets = HAS_WIDGETS and HAS_MATPLOTLIB
        
        if self.use_widgets:
            self._create_widgets()
        
        # 打印阈值信息
        print(f"📊 Dashboard thresholds (based on MAX_STEPS={self.max_steps}):")
        print(f"   Reward: Good > {self.reward_good_threshold:.1f}, Poor < {self.reward_poor_threshold:.1f}")
        print(f"   Error:  Good < {self.error_good_threshold}, Poor > {self.error_poor_threshold}")
        print(f"   Comm:   Good < {self.comm_good_threshold*100:.0f}%, Poor > {self.comm_poor_threshold*100:.0f}%")
    
    def _create_widgets(self):
        """创建 UI 组件"""
        self.title_html = widgets.HTML(value="""
            <div style="background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="color: white; margin: 0; text-align: center;">
                    🎯 Leader-Follower MAS Consensus Control
                </h2>
            </div>
        """)
        
        self.main_progress = widgets.FloatProgress(
            value=0, min=0, max=100, description='Total:',
            bar_style='info', style={'bar_color': '#11998e', 'description_width': '60px'},
            layout=widgets.Layout(width='100%', height='30px')
        )
        
        self.step_progress = widgets.FloatProgress(
            value=0, min=0, max=100, description='Episode:',
            bar_style='success', style={'bar_color': '#38ef7d', 'description_width': '60px'},
            layout=widgets.Layout(width='100%', height='20px')
        )
        
        self.progress_text = widgets.HTML(value="<p>Initializing...</p>")
        self.stats_html = widgets.HTML(value="")
        self.plot_output = widgets.Output()
        self.log_output = widgets.Output(layout=widgets.Layout(
            height='150px', overflow='auto', border='1px solid #ddd', padding='10px'
        ))
    
    def _format_time(self, seconds):
        """格式化时间"""
        if seconds is None or seconds < 0:
            return "N/A"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"
    
    def _get_elapsed(self):
        """获取已用时间"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def _estimate_remaining(self, episode, elapsed):
        """估计剩余时间"""
        if episode == 0 or elapsed is None or elapsed <= 0:
            return "..."
        return self._format_time((elapsed / episode) * (self.total_episodes - episode))
    
    def _get_reward_color(self, reward):
        """根据奖励值返回颜色"""
        if reward > self.reward_good_threshold:
            return "#48bb78"  # 绿色
        elif reward < self.reward_poor_threshold:
            return "#f56565"  # 红色
        else:
            return "#ed8936"  # 橙色
    
    def _get_error_color(self, error):
        """根据误差值返回颜色"""
        if error < self.error_good_threshold:
            return "#48bb78"
        elif error > self.error_poor_threshold:
            return "#f56565"
        else:
            return "#ed8936"
    
    def _get_comm_color(self, comm):
        """根据通信率返回颜色"""
        if comm < self.comm_good_threshold:
            return "#48bb78"
        elif comm > self.comm_poor_threshold:
            return "#f56565"
        else:
            return "#ed8936"
    
    def _generate_stats_html(self, episode, reward, tracking_err, comm, best, losses, elapsed):
        """生成统计信息 HTML（使用动态阈值）"""
        r_color = self._get_reward_color(reward)
        e_color = self._get_error_color(tracking_err)
        c_color = self._get_comm_color(comm)
        
        return f"""
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;">
            <div style="flex:1;min-width:100px;background:linear-gradient(135deg,#11998e,#38ef7d);padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:11px;">📍 Episode</div>
                <div style="font-size:18px;font-weight:bold;">{episode}/{self.total_episodes}</div>
            </div>
            <div style="flex:1;min-width:100px;background:{r_color};padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:11px;">🏆 Reward</div>
                <div style="font-size:18px;font-weight:bold;">{reward:.2f}</div>
                <div style="font-size:9px;">Best: {best:.2f}</div>
            </div>
            <div style="flex:1;min-width:100px;background:{e_color};padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:11px;">🎯 Error</div>
                <div style="font-size:18px;font-weight:bold;">{tracking_err:.4f}</div>
            </div>
            <div style="flex:1;min-width:100px;background:{c_color};padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:11px;">📡 Comm</div>
                <div style="font-size:18px;font-weight:bold;">{comm*100:.1f}%</div>
            </div>
            <div style="flex:1;min-width:100px;background:#4a5568;padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:11px;">⏱️ Time</div>
                <div style="font-size:18px;font-weight:bold;">{self._format_time(elapsed)}</div>
                <div style="font-size:9px;">ETA: {self._estimate_remaining(episode, elapsed)}</div>
            </div>
        </div>
        <div style="background:#f7fafc;padding:6px;border-radius:6px;font-size:11px;">
            Q1: <b>{losses.get('q1',0):.4f}</b> | Q2: <b>{losses.get('q2',0):.4f}</b> | 
            Actor: <b>{losses.get('actor',0):.4f}</b> | α: <b>{losses.get('alpha',0.2):.4f}</b>
        </div>
        """
    
    def display(self):
        """显示仪表盘"""
        self.start_time = time.time()
        if self.use_widgets:
            dashboard = widgets.VBox([
                self.title_html, self.main_progress, self.step_progress,
                self.progress_text, self.stats_html,
                widgets.HTML("<h4>📈 Training Progress</h4>"),
                self.plot_output,
                widgets.HTML("<h4>📝 Log</h4>"),
                self.log_output
            ])
            display(dashboard)
        else:
            print("Dashboard requires ipywidgets in Jupyter environment")
            print("Falling back to console output...")
    
    def update_step(self, step, max_steps):
        """更新步数进度"""
        if self.use_widgets:
            self.step_progress.value = (step / max_steps) * 100
    
    def update_episode(self, episode, reward, tracking_err, comm, losses, trajectory_data=None):
        """更新回合信息"""
        elapsed = self._get_elapsed()
        
        # 记录历史
        self.reward_history.append(reward)
        self.tracking_error_history.append(tracking_err)
        self.comm_history.append(comm)
        
        # 更新最佳记录
        if reward > self.best_reward:
            self.best_reward = reward
            if trajectory_data is not None:
                self.best_trajectory = trajectory_data
        
        if self.use_widgets:
            # 更新进度条
            self.main_progress.value = (episode / self.total_episodes) * 100
            self.step_progress.value = 0
            
            # 更新文本
            speed = episode / elapsed if elapsed > 0 else 0
            self.progress_text.value = f"<p>🚀 <b>Ep {episode}</b> | {speed:.2f} ep/s</p>"
            self.stats_html.value = self._generate_stats_html(
                episode, reward, tracking_err, comm, self.best_reward, losses, elapsed
            )
            
            # 更新日志（使用动态阈值）
            with self.log_output:
                ts = time.strftime("%H:%M:%S")
                if reward >= self.best_reward - 0.1:
                    st = "🏆"
                elif reward > self.reward_good_threshold:
                    st = "✅"
                elif reward > self.reward_poor_threshold:
                    st = "📊"
                else:
                    st = "⚠️"
                print(f"[{ts}] {st} Ep {episode:4d} | R:{reward:7.2f} | Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")
            
            # 更新图表
            if episode % self.vis_interval == 0 or episode == 1:
                self._update_plots()
        else:
            if episode % 20 == 0:
                print(f"Ep {episode:4d} | R:{reward:7.2f} | Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")
    
    def _update_plots(self):
        """更新训练图表（2×3 布局，含通信分析）"""
        if not HAS_MATPLOTLIB:
            return
        
        with self.plot_output:
            clear_output(wait=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
            
            # 颜色定义
            leader_color = '#e74c3c'
            raw_color = '#95a5a6'
            smooth_color = '#11998e'
            error_color = '#f39c12'
            comm_color = '#e74c3c'
            
            # ========== 子图 1: 位置跟踪 ==========
            ax1 = axes[0, 0]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                fp = self.best_trajectory['follower_pos']
                lp = self.best_trajectory['leader_pos']
                num_followers = fp.shape[1]
                
                # 🔧 区分 Pinned 和 Normal Follower
                pinned_indices = [i for i in range(num_followers) if (i + 1) in self.pinned_followers]
                normal_indices = [i for i in range(num_followers) if (i + 1) not in self.pinned_followers]
                
                # 绘制 Normal Followers（蓝色系）
                if normal_indices:
                    colors_normal = plt.cm.Blues(np.linspace(0.4, 0.8, len(normal_indices)))
                    for idx, i in enumerate(normal_indices):
                        label = 'Normal Followers' if idx == 0 else None
                        ax1.plot(t, fp[:, i], color=colors_normal[idx], alpha=0.6, lw=1.0, label=label)
                
                # 绘制 Pinned Followers（绿色系，更粗）
                if pinned_indices:
                    colors_pinned = plt.cm.Greens(np.linspace(0.5, 0.9, len(pinned_indices)))
                    for idx, i in enumerate(pinned_indices):
                        label = 'Pinned Followers' if idx == 0 else None
                        ax1.plot(t, fp[:, i], color=colors_pinned[idx], alpha=0.8, lw=1.8, 
                                linestyle='-', label=label)
                
                # 领导者（红色）
                ax1.plot(t, lp, color=leader_color, lw=2.5, label='Leader', zorder=10)
                
                # 平均值
                avg_fp = fp.mean(axis=1)
                ax1.plot(t, avg_fp, color='#9b59b6', lw=2, linestyle='--', 
                        label='Avg Follower', alpha=0.8, zorder=9)
            
            ax1.set_title(f'Position Tracking (Best R={self.best_reward:.2f})', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Position', fontsize=10)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # ========== 子图 2: 速度跟踪 ==========
            ax2 = axes[0, 1]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                fv = self.best_trajectory['follower_vel']
                lv = self.best_trajectory['leader_vel']
                num_followers = fv.shape[1]
                
                # 🔧 区分 Pinned 和 Normal Follower
                pinned_indices = [i for i in range(num_followers) if (i + 1) in self.pinned_followers]
                normal_indices = [i for i in range(num_followers) if (i + 1) not in self.pinned_followers]
                
                # 绘制 Normal Followers（蓝色系）
                if normal_indices:
                    colors_normal = plt.cm.Blues(np.linspace(0.4, 0.8, len(normal_indices)))
                    for idx, i in enumerate(normal_indices):
                        label = 'Normal Followers' if idx == 0 else None
                        ax2.plot(t, fv[:, i], color=colors_normal[idx], alpha=0.6, lw=1.0, label=label)
                
                # 绘制 Pinned Followers（绿色系，更粗）
                if pinned_indices:
                    colors_pinned = plt.cm.Greens(np.linspace(0.5, 0.9, len(pinned_indices)))
                    for idx, i in enumerate(pinned_indices):
                        label = 'Pinned Followers' if idx == 0 else None
                        ax2.plot(t, fv[:, i], color=colors_pinned[idx], alpha=0.8, lw=1.8, 
                                linestyle='-', label=label)
                
                # 领导者（红色）
                ax2.plot(t, lv, color=leader_color, lw=2.5, label='Leader', zorder=10)
                
                # 平均值
                avg_fv = fv.mean(axis=1)
                ax2.plot(t, avg_fv, color='#9b59b6', lw=2, linestyle='--', 
                        label='Avg Follower', alpha=0.8, zorder=9)
            
            ax2.set_title('Velocity Tracking', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (s)', fontsize=10)
            ax2.set_ylabel('Velocity', fontsize=10)
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # ========== 子图 3: 通信分析（新增）==========
            ax3 = axes[0, 2]
            if self.best_trajectory is not None and 'comm_rates' in self.best_trajectory:
                t_comm = self.best_trajectory['times'][1:]  # 通信数据从 step 1 开始
                comm_rates = self.best_trajectory['comm_rates']
                thresholds = self.best_trajectory['thresholds']
                num_followers = thresholds.shape[1]
                
                # 区分 Pinned 和 Normal
                pinned_indices = [i for i in range(num_followers) if (i + 1) in self.pinned_followers]
                normal_indices = [i for i in range(num_followers) if (i + 1) not in self.pinned_followers]
                
                # 滑动窗口平滑通信率
                window = min(20, len(comm_rates) // 5) if len(comm_rates) > 20 else 5
                if window >= 2:
                    comm_smooth = np.convolve(comm_rates, np.ones(window)/window, mode='valid')
                    t_smooth = t_comm[window-1:]
                else:
                    comm_smooth = comm_rates
                    t_smooth = t_comm
                
                # 绘制平滑通信率
                ax3.plot(t_smooth, comm_smooth * 100, color=comm_color, lw=2.5, 
                        label=f'Comm Rate (smooth w={window})')
                ax3.fill_between(t_smooth, 0, comm_smooth * 100, color=comm_color, alpha=0.2)
                
                # 绘制阈值（副轴）
                ax3t = ax3.twinx()
                
                # Pinned followers 平均阈值
                if pinned_indices:
                    pinned_th = thresholds[:, pinned_indices].mean(axis=1)
                    ax3t.plot(t_comm, pinned_th, color='#27ae60', lw=1.5, linestyle='--',
                             label='Pinned Threshold', alpha=0.8)
                
                # Normal followers 平均阈值
                if normal_indices:
                    normal_th = thresholds[:, normal_indices].mean(axis=1)
                    ax3t.plot(t_comm, normal_th, color='#3498db', lw=1.5, linestyle='--',
                             label='Normal Threshold', alpha=0.8)
                
                # 所有 followers 平均阈值
                avg_th = thresholds.mean(axis=1)
                ax3t.plot(t_comm, avg_th, color='#8e44ad', lw=2, linestyle='-',
                         label='Avg Threshold', alpha=0.9)
                
                ax3.set_xlabel('Time (s)', fontsize=10)
                ax3.set_ylabel('Comm Rate (%)', color=comm_color, fontsize=10)
                ax3.set_ylim(0, 100)
                ax3.tick_params(axis='y', labelcolor=comm_color)
                
                ax3t.set_ylabel('Threshold', color='#8e44ad', fontsize=10)
                ax3t.tick_params(axis='y', labelcolor='#8e44ad')
                
                # 合并图例
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3t.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
                
                # 计算统计信息
                avg_comm = np.mean(comm_rates) * 100
                ax3.set_title(f'Communication Analysis (Avg: {avg_comm:.1f}%)', 
                             fontsize=12, fontweight='bold')
            else:
                ax3.set_title('Communication Analysis', fontsize=12, fontweight='bold')
                ax3.text(0.5, 0.5, 'No data yet', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12, color='gray')
            
            ax3.grid(True, alpha=0.3)
            
            # ========== 子图 4: 奖励曲线 ==========
            ax4 = axes[1, 0]
            num_eps = len(self.reward_history)
            
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                
                ax4.plot(eps, self.reward_history, color=raw_color, alpha=0.5, lw=1, 
                        label='Raw Reward')
                
                if num_eps >= 10:
                    w = min(20, num_eps // 2)
                    if w >= 2:
                        sm = np.convolve(self.reward_history, np.ones(w)/w, mode='valid')
                        sm_eps = np.arange(w, num_eps + 1)
                        ax4.plot(sm_eps, sm, color=smooth_color, lw=2.5, label=f'Smoothed (w={w})')
                
                best_idx = np.argmax(self.reward_history)
                ax4.scatter([best_idx + 1], [self.reward_history[best_idx]], 
                           color='gold', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5, label=f'Best: {self.best_reward:.2f}')
                
                # 🔧 使用动态阈值绘制参考线
                ax4.axhline(y=self.reward_good_threshold, color='green', linestyle='--', 
                           alpha=0.5, label=f'Good ({self.reward_good_threshold:.0f})')
                ax4.axhline(y=self.reward_poor_threshold, color='red', linestyle='--', 
                           alpha=0.5, label=f'Poor ({self.reward_poor_threshold:.0f})')
                
                ax4.set_xlim(0, max(num_eps + 1, 10))
                ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax4.set_title('Episode Reward', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Episode', fontsize=10)
            ax4.set_ylabel('Reward', fontsize=10)
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # ========== 子图 5: 跟踪误差 ==========
            ax5 = axes[1, 1]
            
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                
                ax5.plot(eps, self.tracking_error_history, color=error_color, alpha=0.5, lw=1, 
                        label='Raw Error')
                
                if num_eps >= 10:
                    w = min(20, num_eps // 2)
                    if w >= 2:
                        sme = np.convolve(self.tracking_error_history, np.ones(w)/w, mode='valid')
                        sme_eps = np.arange(w, num_eps + 1)
                        ax5.plot(sme_eps, sme, color='#38ef7d', lw=2.5, label=f'Smoothed (w={w})')
                
                min_idx = np.argmin(self.tracking_error_history)
                min_err = self.tracking_error_history[min_idx]
                ax5.scatter([min_idx + 1], [min_err], 
                           color='lime', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5, label=f'Min: {min_err:.4f}')
                
                # 🔧 使用动态阈值绘制参考线
                ax5.axhline(y=self.error_good_threshold, color='green', linestyle='--', 
                           alpha=0.5, label=f'Good ({self.error_good_threshold})')
                ax5.axhline(y=self.error_poor_threshold, color='red', linestyle='--', 
                           alpha=0.5, label=f'Poor ({self.error_poor_threshold})')
                
                ax5.set_xlim(0, max(num_eps + 1, 10))
                ax5.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax5.set_title('Tracking Error', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Episode', fontsize=10)
            ax5.set_ylabel('Error', fontsize=10)
            ax5.legend(loc='best', fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # ========== 子图 6: 通信率趋势（跨 Episode）==========
            ax6 = axes[1, 2]
            
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                
                # 原始通信率
                ax6.plot(eps, [c*100 for c in self.comm_history], color=comm_color, 
                        alpha=0.5, lw=1, label='Raw Comm Rate')
                
                # 平滑通信率
                if num_eps >= 10:
                    w = min(20, num_eps // 2)
                    if w >= 2:
                        smc = np.convolve(self.comm_history, np.ones(w)/w, mode='valid')
                        smc_eps = np.arange(w, num_eps + 1)
                        ax6.plot(smc_eps, smc * 100, color='#9b59b6', lw=2.5, 
                                label=f'Smoothed (w={w})')
                
                # 参考线
                ax6.axhline(y=self.comm_good_threshold * 100, color='green', linestyle='--', 
                           alpha=0.5, label=f'Good (<{self.comm_good_threshold*100:.0f}%)')
                ax6.axhline(y=self.comm_poor_threshold * 100, color='red', linestyle='--', 
                           alpha=0.5, label=f'Poor (>{self.comm_poor_threshold*100:.0f}%)')
                
                # 最低通信率标记
                min_comm_idx = np.argmin(self.comm_history)
                min_comm = self.comm_history[min_comm_idx]
                ax6.scatter([min_comm_idx + 1], [min_comm * 100], 
                           color='cyan', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5, label=f'Min: {min_comm*100:.1f}%')
                
                ax6.set_xlim(0, max(num_eps + 1, 10))
                ax6.set_ylim(0, 100)
                ax6.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax6.set_title('Communication Rate Trend', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Episode', fontsize=10)
            ax6.set_ylabel('Comm Rate (%)', fontsize=10)
            ax6.legend(loc='best', fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            plt.show()
    
    def finish(self):
        """训练完成"""
        elapsed = self._get_elapsed()
        if self.use_widgets:
            self.main_progress.value = 100
            self.main_progress.bar_style = 'success'
            with self.log_output:
                print("=" * 50)
                print(f"✅ Training Complete!")
                print(f"   Total Time: {self._format_time(elapsed)}")
                print(f"   Best Reward: {self.best_reward:.2f}")
                if self.tracking_error_history:
                    print(f"   Final Tracking Error: {self.tracking_error_history[-1]:.4f}")
                if self.comm_history:
                    print(f"   Final Comm Rate: {self.comm_history[-1]*100:.1f}%")
                print("=" * 50)
        else:
            print(f"\n✅ Training complete!")
            print(f"   Best reward: {self.best_reward:.2f}")
            print(f"   Time: {self._format_time(elapsed)}")
    
    def get_summary(self):
        """获取训练摘要"""
        return {
            'best_reward': self.best_reward,
            'final_reward': self.reward_history[-1] if self.reward_history else None,
            'final_tracking_error': self.tracking_error_history[-1] if self.tracking_error_history else None,
            'final_comm_rate': self.comm_history[-1] if self.comm_history else None,
            'total_episodes': len(self.reward_history),
            'elapsed_time': self._get_elapsed(),
            'max_steps': self.max_steps,
            'thresholds': {
                'reward_good': self.reward_good_threshold,
                'reward_poor': self.reward_poor_threshold,
                'error_good': self.error_good_threshold,
                'error_poor': self.error_poor_threshold,
            }
        }