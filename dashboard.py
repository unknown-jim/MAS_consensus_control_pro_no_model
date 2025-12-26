"""
训练可视化仪表盘 - 修复版
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


class TrainingDashboard:
    """训练仪表盘 - 修复版"""
    
    def __init__(self, total_episodes, vis_interval=10):
        self.total_episodes = total_episodes
        self.vis_interval = vis_interval
        self.start_time = None
        
        # 历史记录
        self.reward_history = []
        self.tracking_error_history = []
        self.comm_history = []
        self.best_reward = -float('inf')
        self.best_trajectory = None
        
        self.use_widgets = HAS_WIDGETS and HAS_MATPLOTLIB
        
        if self.use_widgets:
            self._create_widgets()
    
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
    
    def _generate_stats_html(self, episode, reward, tracking_err, comm, best, losses, elapsed):
        """生成统计信息 HTML"""
        r_color = "#48bb78" if reward > -500 else "#f56565" if reward < -1500 else "#ed8936"
        e_color = "#48bb78" if tracking_err < 0.5 else "#f56565" if tracking_err > 2 else "#ed8936"
        c_color = "#48bb78" if comm < 0.3 else "#f56565" if comm > 0.6 else "#ed8936"
        
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
            
            # 更新日志
            with self.log_output:
                ts = time.strftime("%H:%M:%S")
                st = "🏆" if reward >= self.best_reward - 0.1 else "✅" if reward > -10 else "⚠️"
                print(f"[{ts}] {st} Ep {episode:4d} | R:{reward:7.2f} | Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")
            
            # 更新图表
            if episode % self.vis_interval == 0 or episode == 1:
                self._update_plots()
        else:
            if episode % 20 == 0:
                print(f"Ep {episode:4d} | R:{reward:7.2f} | Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")
    
    def _update_plots(self):
        """更新训练图表 - 修复版"""
        if not HAS_MATPLOTLIB:
            return
        
        with self.plot_output:
            clear_output(wait=True)
            
            # 🔧 使用 constrained_layout 替代 tight_layout
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
            
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
                
                # 绘制 followers
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_followers))
                for i in range(num_followers):
                    label = 'Followers' if i == 0 else None
                    ax1.plot(t, fp[:, i], color=colors[i], alpha=0.6, lw=1.0, label=label)
                
                # 绘制 leader
                ax1.plot(t, lp, color=leader_color, lw=2.5, label='Leader', zorder=10)
                
                # 绘制平均 follower
                avg_fp = fp.mean(axis=1)
                ax1.plot(t, avg_fp, color='#2ecc71', lw=2, linestyle='--', 
                        label='Avg Follower', alpha=0.8, zorder=9)
            
            ax1.set_title(f'Position Tracking (Best R={self.best_reward:.2f})', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Position', fontsize=10)
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # ========== 子图 2: 速度跟踪 ==========
            ax2 = axes[0, 1]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                fv = self.best_trajectory['follower_vel']
                lv = self.best_trajectory['leader_vel']
                num_followers = fv.shape[1]
                
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_followers))
                for i in range(num_followers):
                    label = 'Followers' if i == 0 else None
                    ax2.plot(t, fv[:, i], color=colors[i], alpha=0.6, lw=1.0, label=label)
                
                ax2.plot(t, lv, color=leader_color, lw=2.5, label='Leader', zorder=10)
                
                avg_fv = fv.mean(axis=1)
                ax2.plot(t, avg_fv, color='#2ecc71', lw=2, linestyle='--', 
                        label='Avg Follower', alpha=0.8, zorder=9)
            
            ax2.set_title('Velocity Tracking', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (s)', fontsize=10)
            ax2.set_ylabel('Velocity', fontsize=10)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # ========== 子图 3: 奖励曲线 ==========
            ax3 = axes[1, 0]
            num_eps = len(self.reward_history)
            
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                
                # 绘制原始奖励
                ax3.plot(eps, self.reward_history, color=raw_color, alpha=0.5, lw=1, 
                        label='Raw Reward')
                
                # 绘制平滑奖励
                if num_eps >= 10:
                    w = min(20, num_eps // 2)
                    if w >= 2:
                        sm = np.convolve(self.reward_history, np.ones(w)/w, mode='valid')
                        sm_eps = np.arange(w, num_eps + 1)
                        ax3.plot(sm_eps, sm, color=smooth_color, lw=2.5, label=f'Smoothed (w={w})')
                
                # 标记最佳
                best_idx = np.argmax(self.reward_history)
                ax3.scatter([best_idx + 1], [self.reward_history[best_idx]], 
                           color='gold', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5, label=f'Best: {self.best_reward:.2f}')
                
                # 🔧 修复 x 轴刻度
                ax3.set_xlim(0, max(num_eps + 1, 10))
                ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax3.set_title('Episode Reward', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Episode', fontsize=10)
            ax3.set_ylabel('Reward', fontsize=10)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # 添加通信率副轴
            if num_eps > 0:
                ax3t = ax3.twinx()
                ax3t.plot(eps, [c*100 for c in self.comm_history], color=comm_color, 
                         linestyle=':', lw=1.5, alpha=0.7)
                ax3t.set_ylabel('Comm Rate (%)', color=comm_color, fontsize=10)
                ax3t.set_ylim(0, 100)
                ax3t.tick_params(axis='y', labelcolor=comm_color)
            
            # ========== 子图 4: 跟踪误差 ==========
            ax4 = axes[1, 1]
            
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                
                ax4.plot(eps, self.tracking_error_history, color=error_color, alpha=0.5, lw=1, 
                        label='Raw Error')
                
                if num_eps >= 10:
                    w = min(20, num_eps // 2)
                    if w >= 2:
                        sme = np.convolve(self.tracking_error_history, np.ones(w)/w, mode='valid')
                        sme_eps = np.arange(w, num_eps + 1)
                        ax4.plot(sme_eps, sme, color='#38ef7d', lw=2.5, label=f'Smoothed (w={w})')
                
                # 标记最小误差
                min_idx = np.argmin(self.tracking_error_history)
                min_err = self.tracking_error_history[min_idx]
                ax4.scatter([min_idx + 1], [min_err], 
                           color='lime', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5, label=f'Min: {min_err:.4f}')
                
                # 🔧 修复 x 轴刻度
                ax4.set_xlim(0, max(num_eps + 1, 10))
                ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            ax4.set_title('Tracking Error', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Episode', fontsize=10)
            ax4.set_ylabel('Error', fontsize=10)
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
            
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
            'elapsed_time': self._get_elapsed()
        }