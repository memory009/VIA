#!/usr/bin/env python3
"""
CVaR-based Safe RL 数据处理模块
在 Rollout 阶段结束后，处理轨迹数据并进行 relabeling
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
import time


class CVaRDataProcessor:
    """CVaR 数据处理器 - 用于安全强化学习"""
    
    def __init__(self, cvar_alpha=0.1, penalty_scale=50.0):
        """
        Args:
            cvar_alpha: CVaR 阈值 (0.1 = worst 10%)
            penalty_scale: 惩罚项缩放系数（与 goal_reward=100 同量级）
        """
        self.cvar_alpha = cvar_alpha
        self.penalty_scale = penalty_scale
        
        # POLAR 验证参数（从 rotation_only_map_base.py 复制）
        self.collision_delta = 0.4
        self.safety_margin = 0.05
        self.dt = 0.1
        self.robot_radius = 0.17
        self.n_action_samples = 3
        self.n_trajectory_samples = 8
        self.total_sample_points = self.n_action_samples ** 2 * self.n_trajectory_samples * 4  # 3x3x8x4=288
    
    def process_epoch_trajectories(
        self,
        model,
        epoch,
        traj_dir="training_trajectories_8_polar",
        scenario_dir="train_scenario_8_polar",
        observation_error=0.01
    ):
        """
        处理整个 epoch 的轨迹数据
        
        Args:
            model: 训练的模型（用于 POLAR 验证）
            epoch: 当前 epoch 编号
            traj_dir: 轨迹保存目录
            scenario_dir: 场景保存目录
            observation_error: POLAR 观测误差
        
        Returns:
            list: 处理后的轨迹数据，每条包含 (s, a, R', C, done, s')
        """
        print("\n" + "="*80)
        print(f"🔄 Phase 2: CVaR 数据处理 - Epoch {epoch}")
        print("="*80)
        
        base_dir = Path("src/drl_navigation_ros2/assets")
        epoch_traj_dir = base_dir / traj_dir / f"epoch_{epoch:03d}"
        epoch_scenario_dir = base_dir / scenario_dir / f"epoch_{epoch:03d}"
        
        # Step 1: 加载所有轨迹
        print(f"\n[1/5] 加载轨迹数据...")
        trajectories = self._load_all_trajectories(epoch_traj_dir)
        print(f"   ✅ 加载了 {len(trajectories)} 条轨迹")
        
        # Step 2: CVaR 选择 worst-case
        print(f"\n[2/5] CVaR 选择 (α={self.cvar_alpha})...")
        worst_indices, worst_episodes = self._select_worst_trajectories(trajectories)
        print(f"   ✅ 选出 worst {len(worst_indices)} 条轨迹")
        print(f"   Episodes: {worst_episodes}")
        
        # Step 3: POLAR 验证 worst-case
        print(f"\n[3/5] POLAR 验证 worst-case 轨迹...")
        safety_violations = self._verify_worst_cases(
            model, trajectories, worst_indices,
            epoch_scenario_dir, observation_error
        )
        print(f"   ✅ 验证完成")
        
        # Step 4: Relabel rewards & costs
        print(f"\n[4/5] Relabeling rewards 和 costs...")
        processed_data = self._relabel_trajectories(
            trajectories, worst_indices, safety_violations
        )
        
        # 统计信息
        total_transitions = sum(len(traj['transitions']) for traj in processed_data)
        worst_transitions = sum(
            len(traj['transitions']) for traj in processed_data if traj['is_worst_case']
        )
        avg_penalty = np.mean([
            np.mean([t[5] for t in traj['transitions']])  # penalty per step
            for traj in processed_data if traj['is_worst_case']
        ])
        
        print(f"   ✅ 处理完成")
        print(f"   - 总 transitions: {total_transitions}")
        print(f"   - Worst-case transitions: {worst_transitions}")
        print(f"   - 平均 penalty: {avg_penalty:.2f}")
        
        # Step 5: 保存统计报告
        print(f"\n[5/5] 保存处理报告...")
        self._save_processing_report(
            epoch, trajectories, worst_indices, 
            safety_violations, processed_data, base_dir
        )
        
        print("="*80)
        
        return processed_data
    
    def _load_all_trajectories(self, epoch_traj_dir):
        """加载 epoch 内所有轨迹"""
        trajectories = []
        pkl_files = sorted(epoch_traj_dir.glob("episode_*.pkl"))
        
        for pkl_file in pkl_files:
            episode_num = int(pkl_file.stem.split('_')[1])
            
            try:
                with open(pkl_file, 'rb') as f:
                    traj_data = pickle.load(f)
                
                trajectories.append({
                    'episode': episode_num,
                    'data': traj_data,
                    'total_reward': traj_data['info']['total_reward'],
                    'steps': traj_data['info']['steps'],
                })
            except Exception as e:
                print(f"   ⚠️  加载失败 {pkl_file.name}: {e}")
        
        return trajectories
    
    def _select_worst_trajectories(self, trajectories):
        """CVaR 选择 worst-case 轨迹"""
        rewards = [traj['total_reward'] for traj in trajectories]
        n_worst = max(1, int(len(trajectories) * self.cvar_alpha))
        
        worst_indices = np.argsort(rewards)[:n_worst]
        worst_episodes = [trajectories[i]['episode'] for i in worst_indices]
        
        worst_rewards = [rewards[i] for i in worst_indices]
        print(f"   Worst rewards: {[f'{r:.1f}' for r in worst_rewards]}")
        print(f"   CVaR 平均: {np.mean(worst_rewards):.2f}")
        
        return worst_indices.tolist(), worst_episodes
    
    def _verify_worst_cases(self, model, trajectories, worst_indices, 
                           scenario_dir, observation_error):
        """POLAR 验证 worst-case 轨迹的安全性"""
        from reachability_verifier import (
            compute_reachable_set_pure_polar,
            check_action_safety_geometric_complete
        )
        
        safety_violations = {}
        
        start_time = time.time()
        
        for idx in worst_indices:
            traj = trajectories[idx]
            episode_num = traj['episode']
            
            # 加载场景
            scenario_path = scenario_dir / f"episode_{episode_num:03d}.json"
            if not scenario_path.exists():
                print(f"   ⚠️  场景不存在: episode {episode_num}")
                continue
            
            with open(scenario_path, 'r') as f:
                obstacle_map = json.load(f)
            
            print(f"   验证 episode {episode_num}...", end=" ")
            
            # 逐步验证
            states = traj['data']['states']
            observations = traj['data']['observations']
            
            step_violations = []
            
            for step_idx, state in enumerate(states):
                obs = observations[step_idx]
                
                # 计算可达集
                action_ranges = compute_reachable_set_pure_polar(
                    model.actor, state, observation_error, 1, 4000, 1.0
                )
                
                # 位姿（暂时用默认值，后续可以从 odometry 提取）
                pose = [0.0, 0.0, 0.0]
                
                # 检查安全性（带详细信息）
                is_safe, overlap_info = self._check_safety_with_details(
                    action_ranges, state, pose, obstacle_map
                )
                
                step_violations.append({
                    'step': step_idx,
                    'is_safe': is_safe,
                    'overlap_rate': overlap_info['overlap_rate'],
                    'unsafe_points': overlap_info['unsafe_points'],
                })
            
            safety_violations[episode_num] = step_violations
            
            # 统计
            unsafe_steps = sum(1 for v in step_violations if not v['is_safe'])
            avg_overlap = np.mean([v['overlap_rate'] for v in step_violations])
            
            print(f"✅ Unsafe: {unsafe_steps}/{len(step_violations)} steps, "
                  f"Avg overlap: {avg_overlap:.3f}")
        
        elapsed = time.time() - start_time
        print(f"   ⏱️  验证耗时: {elapsed:.1f}秒")
        
        return safety_violations
    
    def _check_safety_with_details(self, action_ranges, state, pose, obstacle_map):
        """
        检查安全性并返回详细的重叠信息
        
        Returns:
            is_safe: bool
            overlap_info: dict with 'overlap_rate', 'unsafe_points'
        """
        safe_threshold = self.collision_delta + self.safety_margin
        x, y, theta = pose
        
        v_min = (action_ranges[0][0] + 1) / 2
        v_max = (action_ranges[0][1] + 1) / 2
        omega_min, omega_max = action_ranges[1][0], action_ranges[1][1]
        
        v_samples = np.linspace(v_min, v_max, self.n_action_samples)
        omega_samples = np.linspace(omega_min, omega_max, self.n_action_samples)
        
        total_points = 0
        unsafe_points = 0
        
        for v_test in v_samples:
            for omega_test in omega_samples:
                swept_points = self._compute_robot_swept_area(
                    pose, v_test, abs(omega_test)
                )
                
                for point in swept_points:
                    total_points += 1
                    
                    # 检查与所有障碍物的碰撞
                    is_point_safe = True
                    for obs in obstacle_map['obstacles']:
                        if obs['type'] == 'boundary':
                            continue
                        
                        if obs['shape'] == 'box':
                            dist = self._point_to_box_distance(
                                point, obs['position'], obs['size'], obs.get('yaw', 0.0)
                            )
                        elif obs['shape'] == 'cylinder':
                            radius = obs.get('radius', obs['size'][0] / 2.0)
                            dist = self._point_to_circle_distance(
                                point, obs['position'], radius
                            )
                        else:
                            continue
                        
                        if dist < safe_threshold:
                            is_point_safe = False
                            break
                    
                    if not is_point_safe:
                        unsafe_points += 1
        
        overlap_rate = unsafe_points / total_points if total_points > 0 else 0.0
        is_safe = (unsafe_points == 0)
        
        return is_safe, {
            'overlap_rate': overlap_rate,
            'unsafe_points': unsafe_points,
            'total_points': total_points,
        }
    
    def _relabel_trajectories(self, trajectories, worst_indices, safety_violations):
        """
        Relabel rewards 和 costs
        
        对于 worst-case 轨迹：
        - R' = R - penalty (penalty = overlap_rate * penalty_scale)
        - C = overlap_rate
        
        对于 normal 轨迹：
        - R' = R (保持不变)
        - C = 0
        """
        processed_data = []
        
        for idx, traj in enumerate(trajectories):
            episode_num = traj['episode']
            is_worst = idx in worst_indices
            
            states = traj['data']['states']
            actions = traj['data']['actions']
            rewards = traj['data']['rewards']
            dones = traj['data']['dones']
            
            transitions = []
            
            for step_idx in range(len(states) - 1):  # -1 因为需要 next_state
                state = states[step_idx]
                action = actions[step_idx]
                reward_original = rewards[step_idx]
                done = dones[step_idx]
                next_state = states[step_idx + 1]
                
                if is_worst and episode_num in safety_violations:
                    # Worst-case: 计算 penalty 和 cost
                    violation = safety_violations[episode_num][step_idx]
                    overlap_rate = violation['overlap_rate']
                    
                    penalty = overlap_rate * self.penalty_scale
                    reward_modified = reward_original - penalty
                    cost = overlap_rate
                else:
                    # Normal case
                    penalty = 0.0
                    reward_modified = reward_original
                    cost = 0.0
                
                transitions.append((
                    state,           # s
                    action,          # a
                    reward_modified, # R'
                    cost,            # C
                    done,            # terminal
                    next_state,      # s'
                    penalty,         # 用于统计
                ))
            
            processed_data.append({
                'episode': episode_num,
                'is_worst_case': is_worst,
                'transitions': transitions,
            })
        
        return processed_data
    
    def _save_processing_report(self, epoch, trajectories, worst_indices,
                                safety_violations, processed_data, base_dir):
        """保存数据处理报告"""
        report_dir = base_dir / "cvar_processing_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"epoch_{epoch:03d}_report.json"
        
        # 统计信息
        worst_episodes = [trajectories[i]['episode'] for i in worst_indices]
        worst_rewards = [trajectories[i]['total_reward'] for i in worst_indices]
        
        total_transitions = sum(len(traj['transitions']) for traj in processed_data)
        worst_transitions = sum(
            len(traj['transitions']) for traj in processed_data if traj['is_worst_case']
        )
        
        # 计算平均 penalty 和 cost
        all_penalties = []
        all_costs = []
        for traj in processed_data:
            if traj['is_worst_case']:
                for t in traj['transitions']:
                    all_penalties.append(t[6])  # penalty
                    all_costs.append(t[3])      # cost
        
        report = {
            'epoch': epoch,
            'cvar_alpha': self.cvar_alpha,
            'penalty_scale': self.penalty_scale,
            'total_trajectories': len(trajectories),
            'worst_case_count': len(worst_indices),
            'worst_episodes': worst_episodes,
            'worst_rewards': worst_rewards,
            'cvar_mean_reward': float(np.mean(worst_rewards)),
            'total_transitions': total_transitions,
            'worst_transitions': worst_transitions,
            'avg_penalty': float(np.mean(all_penalties)) if all_penalties else 0.0,
            'avg_cost': float(np.mean(all_costs)) if all_costs else 0.0,
            'max_penalty': float(np.max(all_penalties)) if all_penalties else 0.0,
            'max_cost': float(np.max(all_costs)) if all_costs else 0.0,
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ 报告已保存: {report_path.name}")
    
    # ===== 几何计算辅助函数 =====
    
    def _compute_robot_swept_area(self, pose, v_max, omega_max):
        """计算机器人扫过的区域"""
        x0, y0, theta0 = pose
        critical_points = []
        angles_to_check = [0, np.pi/2, np.pi, -np.pi/2]
        
        for angle_offset in angles_to_check:
            edge_angle = theta0 + angle_offset
            critical_points.append((
                x0 + self.robot_radius * np.cos(edge_angle),
                y0 + self.robot_radius * np.sin(edge_angle)
            ))
        
        for t in np.linspace(0, self.dt, self.n_trajectory_samples):
            theta_t = theta0 + omega_max * t
            x_t = x0 + v_max * np.cos(theta0) * t
            y_t = y0 + v_max * np.sin(theta0) * t
            
            for angle_offset in angles_to_check:
                edge_angle = theta_t + angle_offset
                critical_points.append((
                    x_t + self.robot_radius * np.cos(edge_angle),
                    y_t + self.robot_radius * np.sin(edge_angle)
                ))
        
        theta_final = theta0 + omega_max * self.dt
        x_final = x0 + v_max * np.cos(theta0) * self.dt
        y_final = y0 + v_max * np.sin(theta0) * self.dt
        
        for angle_offset in angles_to_check:
            edge_angle = theta_final + angle_offset
            critical_points.append((
                x_final + self.robot_radius * np.cos(edge_angle),
                y_final + self.robot_radius * np.sin(edge_angle)
            ))
        
        return critical_points
    
    def _point_to_box_distance(self, point, box_center, box_size, box_yaw):
        """计算点到旋转矩形的距离"""
        px, py = point
        cx = box_center[0] if isinstance(box_center, (list, tuple)) else box_center
        cy = box_center[1] if isinstance(box_center, (list, tuple)) else box_center
        
        hw, hh = box_size[0] / 2, box_size[1] / 2
        dx, dy = px - cx, py - cy
        
        cos_theta, sin_theta = np.cos(-box_yaw), np.sin(-box_yaw)
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        if abs(local_x) <= hw and abs(local_y) <= hh:
            return -min(hw - abs(local_x), hh - abs(local_y))
        
        nearest_x = np.clip(local_x, -hw, hw)
        nearest_y = np.clip(local_y, -hh, hh)
        
        return np.sqrt((local_x - nearest_x)**2 + (local_y - nearest_y)**2)
    
    def _point_to_circle_distance(self, point, circle_center, radius):
        """计算点到圆的距离"""
        px, py = point
        cx = circle_center[0] if isinstance(circle_center, (list, tuple)) else circle_center
        cy = circle_center[1] if isinstance(circle_center, (list, tuple)) else circle_center
        
        return np.sqrt((px - cx)**2 + (py - cy)**2) - radius


if __name__ == "__main__":
    print("CVaR 数据处理模块")
    print("使用方法:")
    print("  from cvar_data_processor import CVaRDataProcessor")
    print("  processor = CVaRDataProcessor()")
    print("  processed_data = processor.process_epoch_trajectories(model, epoch)")