#!/usr/bin/env python3
"""
轨迹收集脚本
在本地 Gazebo 环境中运行，收集所有评估场景的轨迹
输出：保存到 assets/trajectories_lightweight_8_polar_freeze.pkl
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3 import TD3
from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from TD3.TD3_lightweight_safety_critic import TD3_SafetyCritic as TD3_SafetyCritic_Base
from TD3.TD3_lightweight_safety_critic_with_freeze import TD3_SafetyCritic as TD3_SafetyCritic_Freeze
from TD3.TD3_lightweight_BFTQ import TD3_BFTQ
from TD3.TD3_cvar_cpo import TD3_CVaRCPO
from TD3.TD3_wcsac import TD3_WCSAC
from TD3.TD3_rcpo_strict import TD3_RCPO_Strict
from TD3.TD3_lagrangian import TD3Lagrangian
from TD7.TD7_lightweight import TD7
from ros_python import ROS_env
from utils import pos_data


def load_eval_scenarios(json_path=None):
    """加载评估场景"""
    if json_path is None:
        json_path = Path(__file__).parent.parent / "assets" / "eval_scenarios_8_polar.json"
    
    if not json_path.exists():
        print(f"⚠️  场景文件不存在: {json_path}")
        print("   将生成随机场景")
        from utils import record_eval_positions
        scenarios = record_eval_positions(
            n_eval_scenarios=10,
            save_to_file=True,
            enable_random_obstacles=False
        )
        return scenarios
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    scenarios = []
    for scenario_dict in data['scenarios']:
        scenario = []
        for element_dict in scenario_dict['elements']:
            element = pos_data()
            element.name = element_dict['name']
            element.x = element_dict['x']
            element.y = element_dict['y']
            element.angle = element_dict['angle']
            scenario.append(element)
        scenarios.append(scenario)
    
    return scenarios


def collect_single_trajectory(agent, env, scenario, max_steps=300, model_type="TD3_Lightweight",
                             budget=0.5, initial_e_t=0.0, gamma=0.99, danger_threshold=0.5):
    """
    收集单个场景的轨迹（添加位姿追踪）
    修复：使用真实位姿而非估计值

    Args:
        agent: 训练好的智能体
        env: ROS环境
        scenario: 场景配置
        max_steps: 最大步数
        model_type: 模型类型 ("TD3_Lightweight", "TD3_SafetyCritic", "TD3_SafetyCritic_Freeze", "TD3_BFTQ", "TD3_CVaRCPO", "TD3_WCSAC", "TD3_RCPO_Strict", "TD3_Lagrangian", "TD7")
        budget: BFTQ模型的budget参数(仅用于TD3_BFTQ)
        initial_e_t: CVaR-CPO模型的初始e_t值(仅用于TD3_CVaRCPO，应设为model.var_u.item())
        gamma: discount factor (用于e_t更新，仅CVaR-CPO需要)
        danger_threshold: 危险区域阈值 (用于计算cost，仅CVaR-CPO需要)
    """
    from squaternion import Quaternion

    trajectory = []
    actions = []
    rewards = []
    poses = []  # 保存每一步的真实位姿

    # 重置环境
    latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario)

    # 记录初始信息
    robot_pos = (scenario[-2].x, scenario[-2].y, scenario[-2].angle)
    target_pos = (scenario[-1].x, scenario[-1].y)

    # CVaR-CPO需要动态追踪e_t
    e_t = initial_e_t

    step_count = 0
    while step_count < max_steps:
        # ===== 获取真实位姿（从Gazebo传感器） =====
        latest_position = env.sensor_subscriber.latest_position
        latest_orientation = env.sensor_subscriber.latest_heading
        
        if latest_position is not None and latest_orientation is not None:
            # 提取真实的x, y坐标
            odom_x = latest_position.x
            odom_y = latest_position.y
            # 提取真实的theta角度
            quaternion = Quaternion(
                latest_orientation.w,
                latest_orientation.x,
                latest_orientation.y,
                latest_orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            theta = euler[2]
            current_pose = [odom_x, odom_y, theta]
        else:
            # 如果传感器数据不可用，使用初始位姿
            current_pose = [robot_pos[0], robot_pos[1], robot_pos[2]]
        # ===== 真实位姿获取结束 =====
        
        # 准备状态
        state, terminal = agent.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        trajectory.append(state)
        rewards.append(reward)
        poses.append(current_pose.copy())  # 保存真实位姿
        
        if terminal:
            break

        # 获取动作 (根据模型类型决定参数)
        if model_type == "TD3_BFTQ":
            action = agent.get_action(state, budget=budget, add_noise=False)
        elif model_type == "TD3_CVaRCPO":
            # CVaR-CPO使用act方法（与训练时的evaluate一致）
            action = agent.act(state, e_t)
        else:
            action = agent.get_action(state, add_noise=False)
        actions.append(action)
        a_in = [(action[0] + 1) / 2, action[1]]

        # 执行动作
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )

        # CVaR-CPO需要更新e_t（与训练时的evaluate一致）
        if model_type == "TD3_CVaRCPO":
            # 计算cost
            min_distance = min(latest_scan)
            if collision:
                cost = 1.0
            elif min_distance < danger_threshold:
                cost = 1.0
            else:
                cost = 0.0
            # 更新e_t: e_t ← (e_t - cost) / γ
            e_t = (e_t - cost) / gamma

        step_count += 1
    
    # 打包数据
    trajectory_data = {
        'states': np.array(trajectory),  # (T, 25)
        'actions': np.array(actions),     # (T-1, 2)
        'rewards': np.array(rewards),     # (T,)
        'poses': np.array(poses),         # ← 新增：(T, 3) - (x, y, θ)
        'collision': collision,
        'goal_reached': goal,
        'steps': len(trajectory),
        'total_reward': sum(rewards),
        'robot_start': robot_pos,
        'target_pos': target_pos,
    }
    
    return trajectory_data


def main():
    """主函数：收集所有场景的轨迹"""
    print("\n" + "="*70)
    print("轨迹收集工具")
    print("="*70)
    
    # ===== 1. 加载模型 =====
    print("\n[1/4] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 配置要加载的模型 =====
    # 可选模型类型:
    # - "TD3_Lightweight": 基础轻量级TD3模型
    # - "TD3_SafetyCritic": 安全批评家模型（基础版，来自 TD3_lightweight_safety_critic.py）
    # - "TD3_SafetyCritic_Freeze": 安全批评家模型（支持冻结Task Critic，来自 TD3_lightweight_safety_critic_with_freeze.py）
    # - "TD3_BFTQ": BFTQ模型（Budgeted Reinforcement Learning，来自 TD3_lightweight_BFTQ.py）
    # - "TD3_CVaRCPO": CVaR-CPO模型（CVaR-Constrained Policy Optimization，来自 TD3_cvar_cpo.py）
    # - "TD3_WCSAC": WCSAC模型（Worst-Case Soft Actor Critic，来自 TD3_wcsac.py）
    # - "TD3_RCPO_Strict": RCPO模型（Reward Constrained Policy Optimization，来自 TD3_rcpo_strict.py）
    # - "TD3_Lagrangian": Lagrangian模型（SAC-Lagrangian风格，来自 TD3_lagrangian.py）
    # - "TD7": TD7轻量级模型（来自 TD7_lightweight.py）

    # model_type = "TD3_Lightweight"  # ← 修改这里选择模型类型
    # model_type = "TD3_CVaRCPO"
    # model_type = "TD3_WCSAC"
    # model_type = "TD3_RCPO_Strict"
    model_type = "TD3_Lagrangian"
    # model_type = "TD3_BFTQ"
    # model_type = "TD3_SafetyCritic_Freeze"
    # model_type = "TD3_Lightweight"
    # model_type = "TD7"
    # model_name = "TD3_lightweight_best"     # 要加载的权重文件名（不含后缀）
    # model_name = "TD3_cvar_cpo_best"     # 要加载的权重文件名（不含后缀）
    # model_name = "TD3_rcpo_strict_best"
    model_name = "TD3_lagrangian_best"
    # model_name = "checkpoint_epoch_058"
    # model_name = "TD3_BFTQ_best"
    # model_name = "TD3_safety_epoch_010"
    # model_name = "TD7_lightweight_best"
    # model_name = "TD3_wcsac_best"     # WCSAC模型权重

    # 模型目录配置
    # 先resolve获取绝对路径，再计算parent
    # __file__.resolve() → .../scripts/collect_trajectories_lightweight.py
    # → scripts/ → drl_navigation_ros2/
    script_path = Path(__file__).resolve()  # 先获取绝对路径
    drl_nav_root = script_path.parent.parent  # 回到 drl_navigation_ros2 目录
    # model_dir = Path("models/TD3_lightweight/Nov20_16-39-56_cheeson")
    # project_root is src/drl_navigation_ros2/, need to go up 2 levels to reach git repo root
    repo_root = project_root.resolve().parent.parent
    # model_dir = repo_root / "models" / "TD3_cvar_cpo" / "Jan19_19-23-47_cheeson_cvar_cpo_ablation" # wc0.5
    # model_dir = repo_root / "models" / "TD3_cvar_cpo" / "Jan20_23-09-40_cheeson_cvar_cpo_ablation" # wc0.9
    # model_dir = repo_root / "models" / "TD3_cvar_cpo" / "Jan25_14-26-21_cheeson_cvar_cpo_ablation" # wc0.5_v2
    # model_dir = repo_root / "models" / "TD3_wcsac" / "Jan23_17-04-44_cheeson_td3_wcsac_ablation"  # WCSAC
    # model_dir = repo_root / "models" / "TD3_rcpo_strict" / "Jan29_14-23-54_cheeson_rcpo_strict"  # RCPO_Strict
    model_dir = repo_root / "models" / "TD3_lagrangian" / "Jan31_00-33-06_cheeson_td3_lagrangian_ablation"  # TD3_Lagrangian
    # model_dir = drl_nav_root / "models" / "TD7_lightweight" / "Jan14_14-26-38_cheeson"
    # model_dir = project_root / "models" / "TD3_cvar_cpo" / "Jan06_16-43-33_cheeson_cvar_cpo_ablation" # wc0.1
    # model_dir = Path("models/TD3_BFTQ_8obs/Dec29_18-02-40_cheeson_bftq")
    # model_dir = Path("models/TD3_safety/Dec09_20-20-51_cheeson_from_baseline_frozen")
    # model_dir = Path("models/TD3_lightweight/Nov19_01-37-30_cheeson")

    # BFTQ模型专用参数
    bftq_budget = 0.5  # BFTQ模型的budget参数 (0.0=保守, 1.0=激进)

    # CVaR-CPO模型专用参数（会在加载模型后设置）
    cvar_initial_e_t = None  # 将从model.var_u读取
    cvar_gamma = 0.99  # discount factor（与训练保持一致）
    cvar_danger_threshold = 0.5  # 危险区域阈值（与训练保持一致）

    # 根据模型类型加载
    if model_type == "TD3_SafetyCritic_Freeze":
        print(f"  加载 TD3_SafetyCritic_Freeze 模型 (支持冻结): {model_name}")
        print(f"  来源: TD3_lightweight_safety_critic_with_freeze.py")
        agent = TD3_SafetyCritic_Freeze(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            run_id="eval_mode",
            lambda_safe=50.0,  # 这些参数在推理时不重要
            load_baseline=False,
            baseline_path=None,
            freeze_task_critic=False,
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_SafetyCritic_Freeze 模型加载成功")

    elif model_type == "TD3_SafetyCritic":
        print(f"  加载 TD3_SafetyCritic 模型 (基础版): {model_name}")
        print(f"  来源: TD3_lightweight_safety_critic.py")
        agent = TD3_SafetyCritic_Base(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,
            save_directory=model_dir,
            model_name=model_name,
            run_id="eval_mode",
            lambda_safe=50.0,
        )
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_SafetyCritic 模型加载成功")

    elif model_type == "TD3_Lightweight":
        print(f"  加载 TD3_Lightweight 模型: {model_name}")
        agent = TD3_Lightweight(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            load_model=True,
            model_name=model_name,
            load_directory=model_dir,
        )
        print(f"  ✅ TD3_Lightweight 模型加载成功")

    elif model_type == "TD3_BFTQ":
        print(f"  加载 TD3_BFTQ 模型: {model_name}")
        print(f"  来源: TD3_lightweight_BFTQ.py")
        print(f"  Budget参数: {bftq_budget}")
        agent = TD3_BFTQ(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            run_id="eval_mode",
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_BFTQ 模型加载成功")

    elif model_type == "TD3_CVaRCPO":
        print(f"  加载 TD3_CVaRCPO 模型: {model_name}")
        print(f"  来源: TD3_cvar_cpo.py")
        agent = TD3_CVaRCPO(
            state_dim=25,  # 原始状态维度，网络内部会扩展为26维
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            run_id="eval_mode",
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))

        # 从模型中读取训练好的var_u作为初始e_t（与训练时的evaluate一致）
        cvar_initial_e_t = agent.var_u.item()
        print(f"  ✅ TD3_CVaRCPO 模型加载成功")
        print(f"  📊 初始 e_t 值: {cvar_initial_e_t:.4f} (从 model.var_u 读取)")
        print(f"  📊 gamma: {cvar_gamma}")
        print(f"  📊 danger_threshold: {cvar_danger_threshold}")

    elif model_type == "TD3_WCSAC":
        print(f"  加载 TD3_WCSAC 模型: {model_name}")
        print(f"  来源: TD3_wcsac.py")
        agent = TD3_WCSAC(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            run_id="eval_mode",
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_WCSAC 模型加载成功")
        print(f"  📊 当前 β 值: {agent.get_beta():.4f}")

    elif model_type == "TD3_RCPO_Strict":
        print(f"  加载 TD3_RCPO_Strict 模型: {model_name}")
        print(f"  来源: TD3_rcpo_strict.py")
        agent = TD3_RCPO_Strict(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            run_id="eval_mode",
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_RCPO_Strict 模型加载成功")
        print(f"  📊 当前 λ 值: {agent.lambda_penalty.item():.6f}")

    elif model_type == "TD3_Lagrangian":
        print(f"  加载 TD3_Lagrangian 模型: {model_name}")
        print(f"  来源: TD3_lagrangian.py")
        agent = TD3Lagrangian(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            run_id="eval_mode",
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_Lagrangian 模型加载成功")
        print(f"  📊 当前 κ 值: {agent.kappa.item():.6f}")

    elif model_type == "TD7":
        print(f"  加载 TD7 模型: {model_name}")
        print(f"  来源: TD7_lightweight.py")
        agent = TD7(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
            run_id="eval_mode",
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD7 模型加载成功")

    else:
        raise ValueError(f"未知的模型类型: {model_type}。请选择: TD3_Lightweight, TD3_SafetyCritic, TD3_SafetyCritic_Freeze, TD3_BFTQ, TD3_CVaRCPO, TD3_WCSAC, TD3_RCPO_Strict, TD3_Lagrangian, 或 TD7")
    
    # ===== 2. 初始化环境 =====
    print("\n[2/4] 初始化 ROS 环境...")
    env = ROS_env(enable_random_obstacles=False)
    print("  ✅ ROS 环境初始化成功")
    
    # ===== 3. 加载场景 =====
    print("\n[3/4] 加载评估场景...")
    scenarios = load_eval_scenarios()
    n_scenarios = len(scenarios)
    print(f"  ✅ 加载 {n_scenarios} 个场景")
    
    # ===== 4. 收集所有轨迹 =====
    print(f"\n[4/4] 收集 {n_scenarios} 个场景的轨迹...")
    print("  (这可能需要几分钟，请耐心等待...)\n")
    
    all_trajectories = []
    
    for i, scenario in enumerate(tqdm(scenarios, desc="收集轨迹")):
        try:
            # 根据模型类型准备参数
            if model_type == "TD3_BFTQ":
                trajectory_data = collect_single_trajectory(
                    agent, env, scenario, max_steps=300,
                    model_type=model_type,
                    budget=bftq_budget
                )
            elif model_type == "TD3_CVaRCPO":
                trajectory_data = collect_single_trajectory(
                    agent, env, scenario, max_steps=300,
                    model_type=model_type,
                    initial_e_t=cvar_initial_e_t,
                    gamma=cvar_gamma,
                    danger_threshold=cvar_danger_threshold
                )
            else:
                trajectory_data = collect_single_trajectory(
                    agent, env, scenario, max_steps=300,
                    model_type=model_type
                )
            all_trajectories.append(trajectory_data)
            
            # 打印简要信息
            status = "🎯 目标" if trajectory_data['goal_reached'] else "💥 碰撞"
            print(f"  场景 {i+1}/{n_scenarios}: {status} | "
                  f"步数={trajectory_data['steps']} | "
                  f"奖励={trajectory_data['total_reward']:.1f}")
        
        except Exception as e:
            print(f"  ❌ 场景 {i+1} 失败: {e}")
            all_trajectories.append(None)
    

    # ===== 5. 保存轨迹 =====
    if model_type == "TD3_BFTQ":
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_{model_type.lower()}_budget{bftq_budget}_v1.pkl"
    elif model_type == "TD3_CVaRCPO":
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_wc0.5_v2_{model_type.lower()}_v9.pkl"
    elif model_type == "TD3_WCSAC":
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_{model_type.lower()}_v9.pkl"
    elif model_type == "TD3_RCPO_Strict":
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_{model_type.lower()}_v1.pkl"
    elif model_type == "TD3_Lagrangian":
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_{model_type.lower()}_v9.pkl"
    elif model_type == "TD7":
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_{model_type.lower()}_v9.pkl"
    else:
        output_path = Path(__file__).parent.parent / "assets" / f"trajectories_lightweight_8_polar_Nov20_{model_type.lower()}_v9.pkl"
    # output_path = Path(__file__).parent.parent / "assets" / "trajectories_lightweight_8_polar_freeze_010_v9.pkl"
    # output_path = Path(__file__).parent.parent / "assets" / "trajectories_lightweight_8_polar_v9.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"\n✅ 所有轨迹已保存到: {output_path}")
    
    # ===== 6. 统计信息 =====
    print("\n" + "="*70)
    print("收集统计:")
    
    successful = [t for t in all_trajectories if t is not None]
    print(f"  成功收集: {len(successful)}/{n_scenarios}")
    
    goal_count = sum(1 for t in successful if t['goal_reached'])
    collision_count = sum(1 for t in successful if t['collision'])
    
    print(f"  到达目标: {goal_count} ({goal_count/len(successful)*100:.1f}%)")
    print(f"  发生碰撞: {collision_count} ({collision_count/len(successful)*100:.1f}%)")
    
    total_steps = sum(t['steps'] for t in successful)
    avg_steps = total_steps / len(successful)
    print(f"  总步数: {total_steps}")
    print(f"  平均步数: {avg_steps:.1f}")
    
    avg_reward = np.mean([t['total_reward'] for t in successful])
    print(f"  平均奖励: {avg_reward:.2f}")
    
    print("="*70)
    print("\n🎉 轨迹收集完成！")
    print(f"现在可以将 {output_path} 复制到服务器进行批量验证")


if __name__ == "__main__":
    main()