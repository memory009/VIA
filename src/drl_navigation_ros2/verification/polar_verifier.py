#!/usr/bin/env python3
"""
POLAR 可达性验证主函数
适配 TD3 网络结构：25 → 800 → 600 → 2
"""

import numpy as np
import torch
import sympy as sym
from .taylor_model import (
    TaylorModel,
    TaylorArithmetic,
    BernsteinPolynomial,
    compute_tm_bounds,
    apply_activation,
)


def extract_actor_weights(actor):
    """
    提取 Actor 网络权重
    
    Args:
        actor: TD3 的 Actor 网络
    
    Returns:
        weights: list of numpy arrays [(800, 25), (600, 800), (2, 600)]
        biases: list of numpy arrays [(800,), (600,), (2,)]
    """
    weights = []
    biases = []
    
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if 'weight' in name:
                weights.append(param.cpu().numpy())
            elif 'bias' in name:
                biases.append(param.cpu().numpy())
    
    return weights, biases


def compute_reachable_set(
    actor,
    state,
    observation_error=0.01,
    bern_order=1,
    error_steps=4000,
    max_action=1.0,
):
    """
    POLAR 可达性验证 - 完整流程
    
    Args:
        actor: TD3 Actor 网络
        state: 当前观测状态 [25维]
        observation_error: 观测误差范围（默认 0.01）
        bern_order: Bernstein 多项式阶数（默认 1）
        error_steps: 误差估计采样步数（默认 4000）
        max_action: 最大动作值（默认 1.0）
    
    Returns:
        action_ranges: list of [min, max]，每个动作维度的可达集
        例如：[[-0.523, -0.487], [0.123, 0.156]]
    """
    
    # ===== 步骤1：提取网络权重 =====
    weights, biases = extract_actor_weights(actor)
    
    # 验证网络结构
    assert weights[0].shape == (800, 25), f"Layer 1 weight shape mismatch: {weights[0].shape}"
    assert weights[1].shape == (600, 800), f"Layer 2 weight shape mismatch: {weights[1].shape}"
    assert weights[2].shape == (2, 600), f"Layer 3 weight shape mismatch: {weights[2].shape}"
    
    # ===== 步骤2：创建符号变量 =====
    state_dim = len(state)
    z_symbols = [sym.Symbol(f'z{i}') for i in range(state_dim)]
    
    # ===== 步骤3：构造输入 Taylor 模型 =====
    # 论文 Equation (3): xᵢ = state[i] + ε * zᵢ, zᵢ ∈ [-1, 1]
    TM_state = []
    for i in range(state_dim):
        poly = sym.Poly(
            observation_error * z_symbols[i] + state[i], 
            *z_symbols  # 关键：必须指定生成器
        )
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))
    
    # ===== 步骤4：逐层传播 =====
    TM_input = TM_state
    TA = TaylorArithmetic()
    BP = BernsteinPolynomial(error_steps=error_steps)
    
    num_layers = len(biases)  # 应该是 3 层
    
    for layer_idx in range(num_layers):
        TM_temp = []
        W = weights[layer_idx]  # 当前层权重
        b = biases[layer_idx]   # 当前层偏置
        
        num_neurons = len(b)
        
        for neuron_idx in range(num_neurons):
            # ----- 4.1 线性变换（论文 Equation 4） -----
            tm_neuron = TA.weighted_sumforall(
                TM_input,
                W[neuron_idx],  # 第 neuron_idx 个神经元的权重
                b[neuron_idx]   # 第 neuron_idx 个神经元的偏置
            )
            
            # ----- 4.2 激活函数 -----
            is_hidden = (layer_idx < num_layers - 1)
            
            if is_hidden:
                # 隐藏层：ReLU（论文 Equation 8 优化）
                a, b_bound = compute_tm_bounds(tm_neuron)
                
                if a >= 0:
                    # 情况1：完全激活，ReLU(x) = x
                    TM_after = tm_neuron
                
                elif b_bound <= 0:
                    # 情况2：完全不激活，ReLU(x) = 0
                    zero_poly = sym.Poly(0, *z_symbols)
                    TM_after = TaylorModel(zero_poly, [0, 0])
                
                else:
                    # 情况3：跨越零点，使用 Bernstein 多项式
                    bern_poly = BP.approximate(a, b_bound, bern_order, 'relu')
                    bern_error = BP.compute_error(a, b_bound, 'relu')
                    TM_after = apply_activation(
                        tm_neuron, bern_poly, bern_error, bern_order
                    )
            
            else:
                # 输出层：Tanh
                a, b_bound = compute_tm_bounds(tm_neuron)
                bern_poly = BP.approximate(a, b_bound, bern_order, 'tanh')
                bern_error = BP.compute_error(a, b_bound, 'tanh')
                TM_after = apply_activation(
                    tm_neuron, bern_poly, bern_error, bern_order
                )
                
                # 缩放到动作空间 [-max_action, max_action]
                TM_after = TA.constant_product(TM_after, max_action)
            
            TM_temp.append(TM_after)
        
        # 更新输入到下一层
        TM_input = TM_temp
    
    # ===== 步骤5：计算动作可达集 =====
    action_ranges = []
    for tm in TM_input:
        a, b = compute_tm_bounds(tm)
        action_ranges.append([a, b])
    
    return action_ranges


def check_action_safety(action_ranges, state, collision_threshold=0.4):
    """
    检查动作可达集是否安全
    
    安全条件：
    1. 可达集宽度不能太大（不确定性约束）
    2. 没有碰撞风险（根据激光雷达）
    3. 动作在合理范围内
    
    Args:
        action_ranges: [[v_min, v_max], [ω_min, ω_max]]
        state: 当前观测（前20维是激光雷达数据）
        collision_threshold: 碰撞阈值（默认 0.4m）
    
    Returns:
        bool: 是否安全
    """
    # 1. 不确定性检查
    for i, (min_val, max_val) in enumerate(action_ranges):
        range_width = max_val - min_val
        if range_width > 1.5:  # 可达集太宽
            return False
    
    # 2. 碰撞风险检查（假设前20维是激光雷达）
    if len(state) >= 20:
        laser_readings = state[:20]
        min_laser = np.min(laser_readings)
        
        if min_laser < collision_threshold:  # 距离障碍物很近
            linear_vel_range = action_ranges[0]
            if linear_vel_range[1] > 0.3:  # 可能前进
                return False
    
    # 3. 动作范围检查（根据你的训练代码：线速度 [0, 1], 角速度 [-1, 1]）
    if action_ranges[0][0] < -1.1 or action_ranges[0][1] > 1.1:
        return False
    if action_ranges[1][0] < -1.1 or action_ranges[1][1] > 1.1:
        return False
    
    return True


def verify_safety(agent, state, observation_error=0.01, **kwargs):
    """
    便捷接口：直接从 agent 对象验证安全性
    
    Args:
        agent: TD3 对象（包含 actor 和 max_action）
        state: 当前状态
        observation_error: 观测误差
        **kwargs: 传递给 compute_reachable_set 的其他参数
    
    Returns:
        is_safe: bool
        action_ranges: list of [min, max]
    """
    # 从 agent 获取 max_action
    max_action = getattr(agent, 'max_action', 1.0)
    
    # 计算可达集
    action_ranges = compute_reachable_set(
        agent.actor,
        state,
        observation_error=observation_error,
        max_action=max_action,
        **kwargs
    )
    
    # 安全性判断
    is_safe = check_action_safety(action_ranges, state)
    
    return is_safe, action_ranges