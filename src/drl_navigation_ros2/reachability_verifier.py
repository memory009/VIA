#!/usr/bin/env python3
"""
辅助函数：在训练评估阶段进行可达集验证
供 train.py 的 eval 函数调用
基于 rotation_only_map_base.py，但简化为串行版本（n_workers=1）
"""

import json
from pathlib import Path
import time
import numpy as np
import torch
import pickle


def verify_trajectories_reachability(
    model,
    trajectory_path,
    epoch,
    observation_error=0.01,
    sample_interval=1,
    save_dir="reachability_result_8_polar"
):
    """
    对收集的轨迹进行可达集验证（串行版本）
    
    Args:
        model: 训练的模型
        trajectory_path: 轨迹文件路径（.pkl）
        epoch: 当前epoch编号
        observation_error: 观测误差
        sample_interval: 采样间隔
        save_dir: 保存目录名称（相对于assets文件夹）
    
    Returns:
        str: 保存的JSON文件路径
    """
    # 设置保存路径
    base_dir = Path("src/drl_navigation_ros2/assets")
    result_dir = base_dir / save_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名格式：reachability_result_polar_epoch[i].json
    save_path = result_dir / f"{save_dir}_epoch{epoch:03d}.json"
    
    print(f"🔍 验证 Epoch {epoch} 的可达集安全性...")
    
    # 加载轨迹
    with open(trajectory_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"   加载了 {len(trajectories)} 条轨迹")
    
    # 加载障碍物地图
    project_root = Path("src/drl_navigation_ros2")
    
    # 验证所有轨迹
    start_time = time.time()
    all_results = []
    skipped_trajectories = []  # 记录跳过的轨迹
    
    for traj_idx, trajectory in enumerate(trajectories):
        # ✅ 检查是否超时（步数达到300）
        traj_steps = trajectory['info'].get('steps', 0)
        if traj_steps >= 300:
            scenario_id = trajectory.get('scenario_id', traj_idx)
            skipped_trajectories.append({
                'trajectory_idx': traj_idx,
                'scenario_id': scenario_id,
                'steps': traj_steps,
                'reason': 'timeout'
            })
            print(f"   ⏭️  跳过轨迹 {traj_idx + 1}/{len(trajectories)} (超时，步数={traj_steps})")
            continue
        
        print(f"   处理轨迹 {traj_idx + 1}/{len(trajectories)}...", end=" ")
        
        # 加载对应场景的障碍物地图
        scenario_id = trajectory.get('scenario_id', traj_idx)
        obstacle_map_path = (
            project_root / "assets" / "eval_scenarios_8_polar" / 
            f"obstacle_map_scenario_{scenario_id:02d}.json"
        )
        
        if not obstacle_map_path.exists():
            print(f"❌ 找不到障碍物地图: {obstacle_map_path.name}")
            continue
        
        with open(obstacle_map_path, 'r') as f:
            obstacle_map = json.load(f)
        
        # 提取状态和位姿数据（如果有）
        states = trajectory['states']
        observations = trajectory['observations']
        
        # 采样
        sampled_states = states[::sample_interval]
        n_samples = len(sampled_states)
        
        # 逐点验证
        results = []
        safe_count = 0
        
        for i, state in enumerate(sampled_states):
            step_idx = i * sample_interval
            
            # 获取对应的观测信息
            obs = observations[step_idx] if step_idx < len(observations) else observations[-1]
            
            # 计算可达集
            action_ranges = compute_reachable_set_pure_polar(
                model.actor,
                state,
                observation_error=observation_error,
                bern_order=1,
                error_steps=4000,
                max_action=1.0,
            )
            
            # 构造位姿（从observation中提取）
            # 注意：这里需要根据你的observation结构调整
            # 假设环境可以提供位姿信息
            laser_scan = obs.get('laser_scan', [])
            pose = [0.0, 0.0, 0.0]  # 默认值，需要根据实际情况调整
            
            # 检查安全性
            is_safe = check_action_safety_geometric_complete(
                action_ranges, state, pose, obstacle_map
            )
            
            # 获取确定性动作
            det_action = model.get_action(state, add_noise=False)
            
            # 计算可达集宽度
            width_v = action_ranges[0][1] - action_ranges[0][0]
            width_omega = action_ranges[1][1] - action_ranges[1][0]
            
            if is_safe:
                safe_count += 1
            
            # 保存结果
            result = {
                'step': step_idx,
                'pose': pose,
                'det_action': det_action.tolist() if hasattr(det_action, 'tolist') else list(det_action),
                'action_ranges': action_ranges,
                'is_safe': is_safe,
                'width_v': float(width_v),
                'width_omega': float(width_omega),
                'min_laser': float(np.min(laser_scan)) if len(laser_scan) > 0 else 3.5,
                'distance': float(obs.get('distance', 0.0)),
            }
            results.append(result)
        
        # 轨迹总结
        trajectory_summary = {
            'trajectory_idx': traj_idx,
            'scenario_id': scenario_id,
            'n_samples': n_samples,
            'safe_count': safe_count,
            'safety_rate': safe_count / n_samples if n_samples > 0 else 0,
            'collision': trajectory['info'].get('collision', False),
            'goal_reached': trajectory['info'].get('reached_goal', False),
            'steps': trajectory['info'].get('steps', 0),
            'results': results,
        }
        
        all_results.append(trajectory_summary)
        print(f"✅ 安全率: {trajectory_summary['safety_rate']*100:.1f}%")
    
    elapsed_time = time.time() - start_time
    
    # 计算整体统计
    total_samples = sum(r['n_samples'] for r in all_results)
    total_safe = sum(r['safe_count'] for r in all_results)
    overall_safety_rate = total_safe / total_samples if total_samples > 0 else 0
    
    # 保存结果
    output_data = {
        'metadata': {
            'epoch': epoch,
            'method': 'pure_polar_paper_aligned',
            'model': 'TD3_lightweight',
            'n_trajectories': len(trajectories),
            'n_verified': len(all_results),
            'n_skipped': len(skipped_trajectories),
            'total_samples': total_samples,
            'observation_error': observation_error,
            'sample_interval': sample_interval,
            'elapsed_time': elapsed_time,
        },
        'summary': {
            'overall_safety_rate': overall_safety_rate,
            'total_safe': total_safe,
            'total_samples': total_samples,
        },
        'trajectories': all_results,
        'skipped_trajectories': skipped_trajectories,
    }
    
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # ✅ 保存跳过轨迹的日志文件
    if skipped_trajectories:
        log_path = result_dir / f"{save_dir}_epoch{epoch:03d}_skipped.log"
        with open(log_path, 'w') as f:
            f.write(f"Epoch {epoch} - 跳过的轨迹日志\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"总轨迹数: {len(trajectories)}\n")
            f.write(f"已验证: {len(all_results)}\n")
            f.write(f"已跳过: {len(skipped_trajectories)}\n\n")
            f.write("跳过的轨迹详情:\n")
            f.write("-" * 70 + "\n")
            for skip in skipped_trajectories:
                f.write(f"轨迹索引: {skip['trajectory_idx']}, "
                       f"场景ID: {skip['scenario_id']}, "
                       f"步数: {skip['steps']}, "
                       f"原因: {skip['reason']}\n")
        print(f"   📝 日志已保存: {log_path.name}")
    
    print(f"   ✅ 结果已保存: {save_path.name}")
    print(f"   📊 整体安全率: {overall_safety_rate*100:.1f}% ({total_safe}/{total_samples})")
    print(f"   ⏱️  耗时: {elapsed_time:.1f}秒")
    
    return str(save_path)


def compute_reachable_set_pure_polar(
    actor,
    state,
    observation_error=0.01,
    bern_order=1,
    error_steps=4000,
    max_action=1.0,
):
    """
    纯POLAR可达性验证 - 从 rotation_only_map_base.py 复制
    """
    import sympy as sym
    from verification.taylor_model import (
        TaylorModel,
        TaylorArithmetic,
        BernsteinPolynomial,
        compute_tm_bounds,
        apply_activation,
    )
    
    # 1. 提取Actor权重
    weights = []
    biases = []
    
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if 'weight' in name:
                weights.append(param.cpu().numpy())
            elif 'bias' in name:
                biases.append(param.cpu().numpy())
    
    # 验证网络结构
    state_dim = len(state)
    
    # 2. 创建符号变量
    z_symbols = [sym.Symbol(f'z{i}') for i in range(state_dim)]
    
    # 3. 构造输入Taylor模型
    TM_state = []
    for i in range(state_dim):
        poly = sym.Poly(
            observation_error * z_symbols[i] + state[i], 
            *z_symbols
        )
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))
    
    # 4. 逐层传播
    TM_input = TM_state
    TA = TaylorArithmetic()
    BP = BernsteinPolynomial(error_steps=error_steps)
    
    num_layers = len(biases)
    
    for layer_idx in range(num_layers):
        TM_temp = []
        W = weights[layer_idx]
        b = biases[layer_idx]
        
        num_neurons = len(b)
        
        for neuron_idx in range(num_neurons):
            # 线性变换
            tm_neuron = TA.weighted_sumforall(
                TM_input,
                W[neuron_idx],
                b[neuron_idx]
            )
            
            # 激活函数
            is_hidden = (layer_idx < num_layers - 1)
            
            if is_hidden:
                # ReLU
                a, b_bound = compute_tm_bounds(tm_neuron)
                
                if a >= 0:
                    TM_after = tm_neuron
                elif b_bound <= 0:
                    zero_poly = sym.Poly(0, *z_symbols)
                    TM_after = TaylorModel(zero_poly, [0, 0])
                else:
                    bern_poly = BP.approximate(a, b_bound, bern_order, 'relu')
                    bern_error = BP.compute_error(a, b_bound, 'relu')
                    TM_after = apply_activation(
                        tm_neuron, bern_poly, bern_error, bern_order
                    )
            else:
                # 输出层: Tanh
                a, b_bound = compute_tm_bounds(tm_neuron)
                bern_poly = BP.approximate(a, b_bound, bern_order, 'tanh')
                bern_error = BP.compute_error(a, b_bound, 'tanh')
                TM_after = apply_activation(
                    tm_neuron, bern_poly, bern_error, bern_order
                )
                TM_after = TA.constant_product(TM_after, max_action)
            
            TM_temp.append(TM_after)
        
        TM_input = TM_temp
    
    # 5. 计算动作可达集
    action_ranges = []
    for tm in TM_input:
        a, b = compute_tm_bounds(tm)
        action_ranges.append([a, b])
    
    return action_ranges


def check_action_safety_geometric_complete(action_ranges, state, pose, obstacle_map):
    """
    检查可达集安全性 - 从 rotation_only_map_base.py 复制
    """
    COLLISION_DELTA = 0.4
    SAFETY_MARGIN = 0.05
    DT = 0.1
    ROBOT_RADIUS = 0.17
    N_ACTION_SAMPLES = 3
    N_TRAJECTORY_SAMPLES = 8
    
    safe_threshold = COLLISION_DELTA + SAFETY_MARGIN
    x, y, theta = pose
    
    v_interval = action_ranges[0]
    omega_interval = action_ranges[1]
    
    # 映射到实际控制空间
    v_min = (v_interval[0] + 1) / 2
    v_max = (v_interval[1] + 1) / 2
    omega_min = omega_interval[0]
    omega_max = omega_interval[1]
    
    # 在可达集中采样多个动作
    v_samples = np.linspace(v_min, v_max, N_ACTION_SAMPLES)
    omega_samples = np.linspace(omega_min, omega_max, N_ACTION_SAMPLES)
    
    # 检查每个采样动作
    for v_test in v_samples:
        for omega_test in omega_samples:
            # 计算这个动作的轨迹
            swept_area_points = compute_robot_swept_area(
                pose, v_test, abs(omega_test), DT, ROBOT_RADIUS, N_TRAJECTORY_SAMPLES
            )
            
            # 检查与障碍物的碰撞
            for obs in obstacle_map['obstacles']:
                if obs['type'] == 'boundary':
                    continue
                
                for point in swept_area_points:
                    if obs['shape'] == 'box':
                        dist = point_to_box_distance(
                            point, obs['position'], obs['size'], obs.get('yaw', 0.0)
                        )
                    elif obs['shape'] == 'cylinder':
                        radius = obs.get('radius', obs['size'][0] / 2.0)
                        dist = point_to_circle_distance(
                            point, obs['position'], radius
                        )
                    else:
                        continue
                    
                    if dist < safe_threshold:
                        return False
    
    return True


def compute_robot_swept_area(pose, v_max, omega_max, dt, robot_radius=0.17, n_samples=10):
    """计算机器人扫过的区域"""
    x0, y0, theta0 = pose
    critical_points = []
    
    angles_to_check = [0, np.pi/2, np.pi, -np.pi/2]
    
    for angle_offset in angles_to_check:
        edge_angle = theta0 + angle_offset
        edge_x = x0 + robot_radius * np.cos(edge_angle)
        edge_y = y0 + robot_radius * np.sin(edge_angle)
        critical_points.append((edge_x, edge_y))
    
    for t in np.linspace(0, dt, n_samples):
        theta_t = theta0 + omega_max * t
        x_t = x0 + v_max * np.cos(theta0) * t
        y_t = y0 + v_max * np.sin(theta0) * t
        
        for angle_offset in angles_to_check:
            edge_angle = theta_t + angle_offset
            edge_x = x_t + robot_radius * np.cos(edge_angle)
            edge_y = y_t + robot_radius * np.sin(edge_angle)
            critical_points.append((edge_x, edge_y))
    
    theta_final = theta0 + omega_max * dt
    x_final = x0 + v_max * np.cos(theta0) * dt
    y_final = y0 + v_max * np.sin(theta0) * dt
    
    for angle_offset in angles_to_check:
        edge_angle = theta_final + angle_offset
        edge_x = x_final + robot_radius * np.cos(edge_angle)
        edge_y = y_final + robot_radius * np.sin(edge_angle)
        critical_points.append((edge_x, edge_y))
    
    return critical_points


def point_to_box_distance(point, box_center, box_size, box_yaw=0.0):
    """计算点到旋转矩形的距离"""
    px, py = point
    if isinstance(box_center, (list, tuple)):
        cx, cy = box_center[0], box_center[1]
    else:
        cx, cy = box_center, box_center
    
    hw, hh = box_size[0] / 2, box_size[1] / 2
    
    dx = px - cx
    dy = py - cy
    
    cos_theta = np.cos(-box_yaw)
    sin_theta = np.sin(-box_yaw)
    local_x = dx * cos_theta - dy * sin_theta
    local_y = dx * sin_theta + dy * cos_theta
    
    if abs(local_x) <= hw and abs(local_y) <= hh:
        dist_to_edge_x = hw - abs(local_x)
        dist_to_edge_y = hh - abs(local_y)
        return -min(dist_to_edge_x, dist_to_edge_y)
    
    nearest_x = np.clip(local_x, -hw, hw)
    nearest_y = np.clip(local_y, -hh, hh)
    
    dx_out = local_x - nearest_x
    dy_out = local_y - nearest_y
    
    return np.sqrt(dx_out**2 + dy_out**2)


def point_to_circle_distance(point, circle_center, radius):
    """计算点到圆的距离"""
    px, py = point
    if isinstance(circle_center, (list, tuple)):
        cx, cy = circle_center[0], circle_center[1]
    else:
        cx, cy = circle_center, circle_center
    
    center_dist = np.sqrt((px - cx)**2 + (py - cy)**2)
    return center_dist - radius


if __name__ == "__main__":
    print("这是一个辅助模块，应该从 train.py 导入使用")
    print("示例用法:")
    print("  from reachability_verifier import verify_trajectories_reachability")
    print("  verify_trajectories_reachability(model, trajectory_path, epoch)")