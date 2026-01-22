#!/usr/bin/env python3
"""
并行可达性验证脚本 - 支持多种TD3模型
支持的模型类型:
- TD3_Lightweight: 基础轻量级TD3模型
- TD3_SafetyCritic: 安全批评家模型
- TD3_SafetyCritic_Freeze: 支持冻结Task Critic的安全批评家模型
- TD3_BFTQ: 支持budget参数的BFTQ模型
- TD3_CVaRCPO: CVaR-CPO安全模型

与训练代码完全对齐
"""

import sys
import argparse
try:
    import distutils.version
except AttributeError:
    import distutils
    from packaging import version as packaging_version
    distutils.version = type('version', (), {
        'LooseVersion': packaging_version.Version,
        'StrictVersion': packaging_version.Version
    })
from pathlib import Path
import numpy as np
import torch
import pickle
import json
import time
from multiprocessing import Pool, cpu_count

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from TD3.TD3_lightweight_safety_critic import TD3_SafetyCritic as TD3_SafetyCritic_Base
from TD3.TD3_lightweight_safety_critic_with_freeze import TD3_SafetyCritic as TD3_SafetyCritic_Freeze
from TD3.TD3_lightweight_BFTQ import TD3_BFTQ
from TD3.TD3_cvar_cpo import TD3_CVaRCPO

def point_to_box_distance(point, box_center, box_size, box_yaw=0.0):
    """
    计算点到矩形（可旋转）的最短距离
    
    Args:
        point: (x, y) 点坐标
        box_center: [x, y] or (x, y) 矩形中心
        box_size: [width, height, _] 矩形尺寸
        box_yaw: float, 矩形旋转角度（弧度）
    
    Returns:
        float: 最短距离（米），负值表示点在矩形内部
    """
    px, py = point
    # 确保 box_center 可以用索引访问
    if isinstance(box_center, (list, tuple)):
        cx, cy = box_center[0], box_center[1]
    else:
        cx, cy = box_center, box_center  # 异常情况处理
    
    hw, hh = box_size[0] / 2, box_size[1] / 2
    
    # 将点转换到矩形局部坐标系（考虑旋转）
    dx = px - cx
    dy = py - cy
    
    # 旋转变换（反向旋转，将旋转的矩形"转正"）
    cos_theta = np.cos(-box_yaw)
    sin_theta = np.sin(-box_yaw)
    local_x = dx * cos_theta - dy * sin_theta
    local_y = dx * sin_theta + dy * cos_theta
    
    # 在局部坐标系中，矩形是轴对齐的
    # 如果点在矩形内部
    if abs(local_x) <= hw and abs(local_y) <= hh:
        # 返回到最近边的距离（负值）
        dist_to_edge_x = hw - abs(local_x)
        dist_to_edge_y = hh - abs(local_y)
        return -min(dist_to_edge_x, dist_to_edge_y)
    
    # 点在矩形外部
    nearest_x = np.clip(local_x, -hw, hw)
    nearest_y = np.clip(local_y, -hh, hh)
    
    dx_out = local_x - nearest_x
    dy_out = local_y - nearest_y
    
    return np.sqrt(dx_out**2 + dy_out**2)


def point_to_circle_distance(point, circle_center, radius):
    """
    计算点到圆的最短距离
    
    Args:
        point: (x, y) 点坐标
        circle_center: [x, y] or (x, y) 圆心
        radius: float, 半径
    
    Returns:
        float: 最短距离（米），负值表示点在圆内部
    """
    px, py = point
    if isinstance(circle_center, (list, tuple)):
        cx, cy = circle_center[0], circle_center[1]
    else:
        cx, cy = circle_center, circle_center
    
    center_dist = np.sqrt((px - cx)**2 + (py - cy)**2)
    
    return center_dist - radius


def compute_robot_swept_area(pose, v_max, omega_max, dt, robot_radius=0.17, n_samples=10):
    """
    计算机器人在一个时间步内扫过的区域（考虑旋转）
    
    这是关键函数：不仅考虑直线前进，还考虑旋转时车体的摆动！
    
    Args:
        pose: (x, y, theta) 当前位姿
        v_max: float, 最大线速度（m/s）
        omega_max: float, 最大角速度（rad/s）
        dt: float, 时间步长（秒）
        robot_radius: float, 机器人半径（米）
        n_samples: int, 采样点数（越多越精确，但计算量更大）
    
    Returns:
        list of (x, y): 机器人边缘扫过的所有关键点
    """
    x0, y0, theta0 = pose
    
    # 收集所有可能接触障碍物的点
    critical_points = []
    
    # 1. 当前位置的机器人边缘点（圆形机器人的四个极值点）
    angles_to_check = [0, np.pi/2, np.pi, -np.pi/2]  # 前后左右
    for angle_offset in angles_to_check:
        edge_angle = theta0 + angle_offset
        edge_x = x0 + robot_radius * np.cos(edge_angle)
        edge_y = y0 + robot_radius * np.sin(edge_angle)
        critical_points.append((edge_x, edge_y))
    
    # 2. 模拟运动轨迹（关键！）
    # 同时考虑线速度和角速度的影响
    for t in np.linspace(0, dt, n_samples):
        # 当前时刻的位姿
        theta_t = theta0 + omega_max * t
        x_t = x0 + v_max * np.cos(theta0) * t  # 简化：假设小时间步内theta变化不大
        y_t = y0 + v_max * np.sin(theta0) * t
        
        # 该时刻的机器人边缘点
        for angle_offset in angles_to_check:
            edge_angle = theta_t + angle_offset
            edge_x = x_t + robot_radius * np.cos(edge_angle)
            edge_y = y_t + robot_radius * np.sin(edge_angle)
            critical_points.append((edge_x, edge_y))
    
    # 3. 最终位置
    theta_final = theta0 + omega_max * dt
    x_final = x0 + v_max * np.cos(theta0) * dt
    y_final = y0 + v_max * np.sin(theta0) * dt
    
    for angle_offset in angles_to_check:
        edge_angle = theta_final + angle_offset
        edge_x = x_final + robot_radius * np.cos(edge_angle)
        edge_y = y_final + robot_radius * np.sin(edge_angle)
        critical_points.append((edge_x, edge_y))
    
    return critical_points

def compute_reachable_set_pure_polar(
    actor,
    state,
    observation_error=0.01,
    bern_order=1,
    error_steps=4000,
    max_action=1.0,
    budget=None,  # 新增: 对于BFTQ模型，需要传入budget参数
    e_t=None,  # 新增: 对于CVaRCPO模型，需要传入e_t参数
):
    """
    纯POLAR可达性验证 - 与clearpath_rl_polar完全一致
    支持动态网络结构（自动适配隐藏层维度）
    支持BFTQ模型（budget作为额外输入）
    支持CVaRCPO模型（e_t作为额外输入）
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

    # 对于BFTQ模型，输入维度是 state_dim + 1 (budget)
    # 对于CVaRCPO模型，输入维度是 state_dim + 1 (e_t)
    if budget is not None:
        input_dim = state_dim + 1
    elif e_t is not None:
        input_dim = state_dim + 1
    else:
        input_dim = state_dim

    assert weights[0].shape[1] == input_dim, \
        f"输入维度不匹配: 期望 {input_dim}, 实际 {weights[0].shape[1]}"
    assert weights[-1].shape[0] == 2, \
        f"输出维度不匹配: 期望 2, 实际 {weights[-1].shape[0]}"

    # 自动检测隐藏层维度
    hidden_dim = weights[0].shape[0]

    # 2. 创建符号变量
    z_symbols = [sym.Symbol(f'z{i}') for i in range(input_dim)]

    # 3. 构造输入Taylor模型
    TM_state = []
    # 对于state部分，添加观测误差
    for i in range(state_dim):
        poly = sym.Poly(
            observation_error * z_symbols[i] + state[i],
            *z_symbols
        )
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))

    # 对于BFTQ模型，将budget作为确定性输入（无误差）
    if budget is not None:
        poly = sym.Poly(budget, *z_symbols)  # budget作为常数，无不确定性
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))

    # 对于CVaRCPO模型，将e_t作为确定性输入（无误差）
    if e_t is not None:
        poly = sym.Poly(e_t, *z_symbols)  # e_t作为常数，无不确定性
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
                # ReLU (使用论文Equation 8的优化)
                a, b_bound = compute_tm_bounds(tm_neuron)
                
                if a >= 0:
                    # 情况1: 完全激活
                    TM_after = tm_neuron
                elif b_bound <= 0:
                    # 情况2: 完全不激活
                    zero_poly = sym.Poly(0, *z_symbols)
                    TM_after = TaylorModel(zero_poly, [0, 0])
                else:
                    # 情况3: 跨越零点，使用Bernstein多项式
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
                # 缩放到动作空间
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
    ✅ 完整版本：采样可达集中的多个动作
    """
    COLLISION_DELTA = 0.4
    # SAFETY_MARGIN = 0.05
    SAFETY_MARGIN = 0.0  # 关闭额外安全边距以匹配训练时设置
    DT = 0.1
    ROBOT_RADIUS = 0.17
    N_ACTION_SAMPLES = 3        # 每维采样3个点（3×3=9个组合）
    N_TRAJECTORY_SAMPLES = 8    # 时间轴采样点数
    
    safe_threshold = COLLISION_DELTA + SAFETY_MARGIN
    x, y, theta = pose
    
    v_interval = action_ranges[0]
    omega_interval = action_ranges[1]
    
    # 映射到实际控制空间
    v_min = (v_interval[0] + 1) / 2
    v_max = (v_interval[1] + 1) / 2
    omega_min = omega_interval[0]
    omega_max = omega_interval[1]
    
    # ===== 关键改进：在可达集中采样多个动作 =====
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
                        return False  # 发现不安全的动作
    
    return True  # 所有采样动作都安全

def verify_single_trajectory_worker(args):
    """
    单个轨迹的验证函数（纯POLAR + 几何距离 + 旋转检查）

    关键改进：
    1. 加载真实障碍物地图
    2. 考虑旋转时的车体扫过区域
    3. 检查整个运动轨迹，不只是终点
    """
    trajectory_idx, trajectory_data, model_path, model_type, model_name, observation_error, sample_interval, budget, e_t = args

    # ===== 1. 加载模型（支持三种类型）=====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "TD3_SafetyCritic_Freeze":
        # 加载 TD3_SafetyCritic_Freeze 模型（支持冻结Task Critic）
        agent = TD3_SafetyCritic_Freeze(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,
            save_directory=model_path,
            model_name=model_name,
            run_id="polar_verification",
            lambda_safe=50.0,
            load_baseline=False,
            baseline_path=None,
            freeze_task_critic=False,
        )
        agent.load(filename=model_name, directory=str(model_path))

    elif model_type == "TD3_SafetyCritic":
        # 加载 TD3_SafetyCritic 模型（基础版）
        agent = TD3_SafetyCritic_Base(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,
            save_directory=model_path,
            model_name=model_name,
            run_id="polar_verification",
            lambda_safe=50.0,
        )
        agent.load(filename=model_name, directory=str(model_path))

    elif model_type == "TD3_Lightweight":
        # 加载 TD3_Lightweight 模型
        agent = TD3_Lightweight(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            hidden_dim=26,
            load_model=True,
            model_name=model_name,
            load_directory=model_path,
        )

    elif model_type == "TD3_BFTQ":
        # 加载 TD3_BFTQ 模型（支持budget参数）
        agent = TD3_BFTQ(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            hidden_dim=26,
            load_model=False,
            save_directory=model_path,
            model_name=model_name,
            run_id="polar_verification_bftq",
        )
        agent.load(filename=model_name, directory=str(model_path))

    elif model_type == "TD3_CVaRCPO":
        # 加载 TD3_CVaRCPO 模型（支持e_t参数）
        agent = TD3_CVaRCPO(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            hidden_dim=26,
            load_model=False,
            save_directory=model_path,
            model_name=model_name,
            run_id="polar_verification_cvar_cpo",
        )
        agent.load(filename=model_name, directory=str(model_path))

        # 从checkpoint中自动获取var_u作为e_t（如果命令行未指定）
        if e_t is None or e_t == 0.0:
            e_t = agent.var_u.item()
            print(f"[进程 {trajectory_idx+1}] 📊 从checkpoint自动读取 var_u 作为 e_t: {e_t:.4f}")

    else:
        raise ValueError(f"未知的模型类型: {model_type}。请选择: TD3_Lightweight, TD3_SafetyCritic, TD3_SafetyCritic_Freeze, TD3_BFTQ, 或 TD3_CVaRCPO")
    
    # ===== 2. 加载对应场景的障碍物地图 =====
    # 构造障碍物地图文件路径（根据模型类型选择场景目录）
    scenario_dir = "eval_scenarios_12" if model_type in ["TD3_Lightweight", "TD3_SafetyCritic", "TD3_SafetyCritic_Freeze", "TD3_BFTQ", "TD3_CVaRCPO"] else "eval_scenarios_12"
    obstacle_map_path = (
        project_root / "assets" / scenario_dir /
        f"obstacle_map_scenario_{trajectory_idx:02d}.json"
    )
    
    if obstacle_map_path.exists():
        with open(obstacle_map_path, 'r') as f:
            obstacle_map = json.load(f)
        print(f"[进程 {trajectory_idx+1}] ✅ 加载障碍物地图: {obstacle_map_path.name}")
        print(f"[进程 {trajectory_idx+1}] 障碍物数量: {obstacle_map['total_obstacles']}")
    else:
        print(f"[进程 {trajectory_idx+1}] ❌ 找不到障碍物地图: {obstacle_map_path}")
        print(f"[进程 {trajectory_idx+1}] 请先运行: python scripts/export_gazebo_map.py")
        raise FileNotFoundError(f"障碍物地图不存在: {obstacle_map_path}")
    
    # ===== 3. 提取轨迹数据并采样 =====
    states = trajectory_data['states']
    poses = trajectory_data['poses']
    
    sampled_states = states[::sample_interval]
    sampled_poses = poses[::sample_interval]
    n_samples = len(sampled_states)
    
    print(f"[进程 {trajectory_idx+1}] 开始验证 {n_samples} 个采样点（几何+旋转版本）...")
    
    # ===== 4. 逐点验证 =====
    results = []
    safe_count = 0
    start_time = time.time()
    
    for i, (state, pose) in enumerate(zip(sampled_states, sampled_poses)):
        step_idx = i * sample_interval
        
        # 进度显示
        if i % max(1, n_samples // 4) == 0:
            elapsed = time.time() - start_time
            print(f"[进程 {trajectory_idx+1}] 进度: {i+1}/{n_samples} "
                  f"({i/n_samples*100:.0f}%) | 已用时: {elapsed/60:.1f}分钟")
        
        # 计算可达集（对于BFTQ模型，传入budget参数；对于CVaRCPO模型，传入e_t参数）
        action_ranges = compute_reachable_set_pure_polar(
            agent.actor,
            state,
            observation_error=observation_error,
            bern_order=1,
            error_steps=4000,
            max_action=1.0,
            budget=budget if model_type == "TD3_BFTQ" else None,
            e_t=e_t if model_type == "TD3_CVaRCPO" else None,
        )

        is_safe = check_action_safety_geometric_complete(
            action_ranges, state, pose, obstacle_map
        )

        # 获取确定性动作（用于对比）
        # 对于BFTQ模型，需要传入budget参数
        # 对于CVaRCPO模型，需要传入e_t参数
        if model_type == "TD3_BFTQ":
            det_action = agent.get_action(state, budget, add_noise=False)
        elif model_type == "TD3_CVaRCPO":
            det_action = agent.get_action(state, e_t, add_noise=False)
        else:
            det_action = agent.get_action(state, add_noise=False)
        
        # 计算可达集宽度
        width_v = action_ranges[0][1] - action_ranges[0][0]
        width_omega = action_ranges[1][1] - action_ranges[1][0]
        
        if is_safe:
            safe_count += 1
        
        # 保存结果
        result = {
            'step': step_idx,
            'pose': pose.tolist(),
            'det_action': det_action.tolist(),
            'action_ranges': action_ranges,
            'is_safe': is_safe,
            'width_v': float(width_v),
            'width_omega': float(width_omega),
            'min_laser': float(np.min(state[0:20])),
            'distance': float(state[20]),
        }
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # ===== 5. 轨迹总结 =====
    trajectory_summary = {
        'trajectory_idx': trajectory_idx,
        'n_samples': n_samples,
        'safe_count': safe_count,
        'safety_rate': safe_count / n_samples if n_samples > 0 else 0,
        'collision': trajectory_data['collision'],
        'goal_reached': trajectory_data['goal_reached'],
        'steps': trajectory_data['steps'],
        'total_reward': float(trajectory_data['total_reward']),
        'compute_time': elapsed_time,
        'results': results,
        'obstacle_map_file': obstacle_map_path.name,
        'obstacle_count': obstacle_map['total_obstacles'],
        'e_t_used': e_t if model_type == "TD3_CVaRCPO" else None,  # 记录实际使用的e_t值
    }

    print(f"[进程 {trajectory_idx+1}] ✅ 完成！安全率: {trajectory_summary['safety_rate']*100:.1f}% | "
          f"耗时: {elapsed_time/60:.1f}分钟")

    return (trajectory_idx, trajectory_summary)


def load_trajectories(pkl_path):
    """
    加载保存的轨迹

    Args:
        pkl_path: 轨迹文件的完整路径

    Returns:
        有效的轨迹列表
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"轨迹文件不存在: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        trajectories = pickle.load(f)

    valid_trajectories = [t for t in trajectories if t is not None]
    return valid_trajectories


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='POLAR并行验证工具 (支持多种模型)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认budget=0.0
  python rotation_only_map_base.py

  # 指定budget=0.5
  python rotation_only_map_base.py --budget 0.5

  # 指定budget=1.0
  python rotation_only_map_base.py --budget 1.0

  # 同时指定版本和budget
  python rotation_only_map_base.py --version v2 --budget 0.5
        """
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=0.0,
        help='BFTQ模型的budget参数 (默认: 0.0，可选: 0.0, 0.5, 1.0等)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        help='轨迹版本号 (默认: v1，可选: v1-v9)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='TD3_BFTQ',
        choices=['TD3_BFTQ', 'TD3_SafetyCritic_Freeze', 'TD3_SafetyCritic', 'TD3_Lightweight', 'TD3_CVaRCPO'],
        help='模型类型 (默认: TD3_BFTQ)'
    )
    parser.add_argument(
        '--safety-epoch',
        type=str,
        default='010',
        help='SafetyCritic的epoch (默认: 010，仅对SafetyCritic模型有效)'
    )
    parser.add_argument(
        '--e-t',
        type=float,
        default=0.0,
        help='CVaRCPO模型的e_t参数 (默认: 0.0 表示自动从checkpoint读取var_u，仅对CVaRCPO模型有效)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 POLAR并行验证工具 (支持多种模型)")
    print("="*70)

    n_cores = cpu_count()
    print(f"\n检测到 CPU 核心数: {n_cores}")

    # ========== 配置区域 ==========
    # 从命令行参数获取配置
    model_type = args.model_type
    trajectory_version = args.version
    safety_critic_epoch = args.safety_epoch
    budget = args.budget
    e_t = args.e_t

    print(f"\n📋 运行配置:")
    print(f"  模型类型: {model_type}")
    print(f"  轨迹版本: {trajectory_version}")
    if model_type == "TD3_BFTQ":
        print(f"  Budget参数: {budget}")
    if model_type == "TD3_CVaRCPO":
        if e_t == 0.0:
            print(f"  e_t参数: 自动从checkpoint读取 (var_u)")
        else:
            print(f"  e_t参数: {e_t} (手动指定)")
    if model_type in ["TD3_SafetyCritic", "TD3_SafetyCritic_Freeze"]:
        print(f"  Safety Epoch: {safety_critic_epoch}")

    # 根据模型类型配置路径（自动匹配）
    if model_type in ["TD3_SafetyCritic", "TD3_SafetyCritic_Freeze"]:
        model_name = f"TD3_safety_epoch_{safety_critic_epoch}"
        model_path = project_root / "models" / "TD3_safety" / "Dec09_20-20-51_cheeson_from_baseline_frozen"
        trajectory_path = project_root / "assets" / f"trajectories_lightweight_8_polar_freeze_{safety_critic_epoch}_{trajectory_version}.pkl"
    elif model_type == "TD3_Lightweight":
        model_name = "TD3_lightweight_best"
        model_path = project_root / "models" / "TD3_lightweight" / "Nov24_22-43-08_cheeson"
        trajectory_path = project_root / "assets" / f"trajectories_lightweight_8_polar_{trajectory_version}.pkl"
    elif model_type == "TD3_BFTQ":
        model_name = "checkpoint_epoch_058"
        model_path = project_root / "models" / "TD3_BFTQ_8obs" / "Dec29_18-02-40_cheeson_bftq"
        # 轨迹文件名必须与budget值匹配
        budget_str = f"{budget:.1f}"
        trajectory_path = project_root / "assets" / "Traj_bftq_checkpoint_best" / f"trajectories_lightweight_8_polar_td3_bftq_budget{budget_str}_{trajectory_version}.pkl"
    elif model_type == "TD3_CVaRCPO":
        # model_name = "TD3_cvar_cpo_epoch_059"
        model_name = "TD3_cvar_cpo_best"
        model_path = project_root / "models" / "TD3_cvar_cpo" / "Jan08_21-25-09_cheeson_cvar_cpo_ablation"
        # 对于CVaRCPO模型，轨迹文件名为 trajectories_lightweight_8_polar_td3_cvarcpo_vX.pkl
        trajectory_path = project_root / "assets" / f"trajectories_lightweight_12_polar_td3_cvarcpo_12_{trajectory_version}.pkl"
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    # ==============================

    print("\n[1/3] 加载轨迹...")
    print(f"  轨迹文件: {trajectory_path}")
    trajectories = load_trajectories(pkl_path=trajectory_path)
    n_trajectories = len(trajectories)
    print(f"  ✅ 加载 {n_trajectories} 条轨迹")

    total_states = sum(t['steps'] for t in trajectories)
    print(f"  总状态数: {total_states}")

    print("\n[2/3] 准备并行计算...")

    observation_error = 0.01
    sample_interval = 1

    n_workers = min(n_trajectories, n_cores // 2)
    print(f"  模型类型: {model_type}")
    print(f"  模型名称: {model_name}")
    print(f"  模型路径: {model_path}")
    if model_type == "TD3_BFTQ":
        print(f"  Budget参数: {budget}")
    if model_type == "TD3_CVaRCPO":
        print(f"  e_t参数: {e_t}")
    print(f"  并行进程数: {n_workers}")
    print(f"  观测误差: ±{observation_error}")
    print(f"  采样间隔: 每 {sample_interval} 步")

    args_list = [
        (i, traj, model_path, model_type, model_name, observation_error, sample_interval, budget, e_t)
        for i, traj in enumerate(trajectories)
    ]
    
    print(f"\n[3/3] 启动 {n_workers} 个并行进程...")
    print("="*70)
    
    start_time = time.time()
    
    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(verify_single_trajectory_worker, args_list)
    except Exception as e:
        print(f"\n❌ 并行验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("验证统计:")
    print("="*70)
    
    results = sorted(results, key=lambda x: x[0])
    all_results = [r[1] for r in results]
    
    total_samples = sum(r['n_samples'] for r in all_results)
    total_safe = sum(r['safe_count'] for r in all_results)
    overall_safety_rate = total_safe / total_samples if total_samples > 0 else 0
    
    print(f"\n整体可达集安全性:")
    print(f"  总采样点: {total_samples}")
    print(f"  安全点数: {total_safe}")
    print(f"  安全率: {overall_safety_rate*100:.1f}%")
    
    # 轨迹分类统计
    goal_trajectories = [r for r in all_results if r['goal_reached']]
    collision_trajectories = [r for r in all_results if r['collision']]
    
    print(f"\n按轨迹结果分类:")
    print(f"  到达目标的轨迹: {len(goal_trajectories)}")
    if goal_trajectories:
        goal_safety = np.mean([r['safety_rate'] for r in goal_trajectories])
        print(f"    平均安全率: {goal_safety*100:.1f}%")
    
    print(f"  碰撞的轨迹: {len(collision_trajectories)}")
    if collision_trajectories:
        collision_safety = np.mean([r['safety_rate'] for r in collision_trajectories])
        print(f"    平均安全率: {collision_safety*100:.1f}%")
    
    # ===== ✅ 修正：增强的可达集宽度统计（加上最小值）=====
    all_widths_v = []
    all_widths_omega = []
    
    for result in all_results:
        for r in result['results']:
            all_widths_v.append(r['width_v'])
            all_widths_omega.append(r['width_omega'])
    
    print(f"\n可达集宽度统计:")
    print(f"  线速度:")
    print(f"    最小: {np.min(all_widths_v):.6f}")  
    print(f"    平均: {np.mean(all_widths_v):.6f}")
    print(f"    中位数: {np.median(all_widths_v):.6f}")
    print(f"    标准差: {np.std(all_widths_v):.6f}")
    print(f"    最大: {np.max(all_widths_v):.6f}")
    print(f"    95%分位: {np.percentile(all_widths_v, 95):.6f}")

    print(f"  角速度:")
    print(f"    最小: {np.min(all_widths_omega):.6f}")
    print(f"    平均: {np.mean(all_widths_omega):.6f}")
    print(f"    中位数: {np.median(all_widths_omega):.6f}")
    print(f"    标准差: {np.std(all_widths_omega):.6f}")
    print(f"    最大: {np.max(all_widths_omega):.6f}")
    print(f"    95%分位: {np.percentile(all_widths_omega, 95):.6f}")
    
    print(f"\n性能统计:")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print(f"  平均每轨迹: {total_elapsed/n_trajectories:.1f} 秒")
    print(f"  平均每采样点: {total_elapsed/total_samples:.2f} 秒")
    
    # 计算加速比
    avg_traj_time = np.mean([r['compute_time'] for r in all_results])
    serial_time = avg_traj_time * n_trajectories
    speedup = serial_time / total_elapsed
    
    print(f"\n并行加速:")
    print(f"  串行预计耗时: {serial_time/60:.1f} 分钟")
    print(f"  并行实际耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  并行效率: {speedup/n_workers*100:.1f}%")

    # 生成输出文件名
    if model_type == "TD3_BFTQ":
        budget_str = f"{budget:.1f}"
        output_filename = Path("Traj_bftq_checkpoint_best") / f"reachability_results_pure_polar_td3_bftq_budget{budget_str}_{trajectory_version}.json"
    elif model_type == "TD3_CVaRCPO":
        # 从第一个结果中获取实际使用的e_t值（所有worker使用相同的e_t）
        actual_e_t = all_results[0].get('e_t_used', 0.0)
        # 使用4位小数精度显示e_t值
        e_t_str = f"{actual_e_t:.4f}".replace('.', 'p')
        output_filename = f"reachability_results_pure_polar_td3_cvar_cpo_12_varu{e_t_str}_{trajectory_version}.json"
        print(f"\n📊 实际使用的 var_u 值: {actual_e_t:.4f}")
    elif model_type in ["TD3_SafetyCritic", "TD3_SafetyCritic_Freeze"]:
        output_filename = f"reachability_results_pure_polar_lightweight_8_freeze_{safety_critic_epoch}_{trajectory_version}.json"
    else:
        output_filename = f"reachability_results_pure_polar_{model_type.lower()}_{trajectory_version}.json"

    output_path = Path(__file__).parent.parent / "assets" / output_filename

    output_data = {
        'metadata': {
            'method': 'pure_polar_paper_aligned',
            'model': model_name,
            'model_type': model_type,
            'hidden_dim': 26,
            'n_trajectories': n_trajectories,
            'total_samples': total_samples,
            'observation_error': observation_error,
            'sample_interval': sample_interval,
            'bern_order': 1,
            'error_steps': 4000,
            'n_workers': n_workers,
            'n_cores': n_cores,
            'elapsed_time': total_elapsed,
            'speedup': speedup,
            'trajectory_file': str(trajectory_path.name),
            'budget': budget if model_type == "TD3_BFTQ" else None,
            'e_t': all_results[0].get('e_t_used') if model_type == "TD3_CVaRCPO" else None,
            'safety_thresholds': {
                'collision_delta': 0.4,
                'safety_margin': 0.00,
            },
        },
        'summary': {
            'overall_safety_rate': overall_safety_rate,
            'total_safe': total_safe,
            'total_samples': total_samples,
            'goal_trajectories': len(goal_trajectories),
            'collision_trajectories': len(collision_trajectories),
            'width_statistics': {
                'linear': {
                    'min': float(np.min(all_widths_v)),
                    'mean': float(np.mean(all_widths_v)),
                    'median': float(np.median(all_widths_v)),
                    'std': float(np.std(all_widths_v)),
                    'max': float(np.max(all_widths_v)),
                    'p95': float(np.percentile(all_widths_v, 95)),
                },
                'angular': {
                    'min': float(np.min(all_widths_omega)),
                    'mean': float(np.mean(all_widths_omega)),
                    'median': float(np.median(all_widths_omega)),
                    'std': float(np.std(all_widths_omega)),
                    'max': float(np.max(all_widths_omega)),
                    'p95': float(np.percentile(all_widths_omega, 95)),
                },
            },
        },
        'trajectories': all_results,
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ 结果已保存到: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("="*70)
    print("\n🎉 可达性验证完成！")


if __name__ == "__main__":
    main()