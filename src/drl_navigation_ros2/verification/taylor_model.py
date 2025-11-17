#!/usr/bin/env python3
"""
Taylor Model 和 POLAR 可达性验证核心模块
严格遵循论文 "Reachability Verification Based Reliability Assessment for DRL"
"""

import numpy as np
import sympy as sym
from functools import reduce
import operator as op
import math


def ncr(n, r):
    """组合数 C(n, r)"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


class TaylorModel:
    """
    Taylor 模型：TM = p(z) + I
    其中 p(z) 是多项式，I 是误差区间，z ∈ [-1, 1]^n
    """
    def __init__(self, poly, remainder):
        self.poly = poly            # sympy.Poly 对象
        self.remainder = remainder  # [lower, upper]


class TaylorArithmetic:
    """Taylor 算术运算"""
    
    def weighted_sumforall(self, taylor_models, weights, bias):
        """
        论文 Equation (4): 神经网络线性层的 Taylor 模型传播
        计算：Σ(wᵢ * TMᵢ) + b
        
        Args:
            taylor_models: [TM₀, TM₁, ..., TMₙ]
            weights: [w₀, w₁, ..., wₙ]
            bias: b
        
        Returns:
            TaylorModel
        """
        # 多项式求和
        temp_poly = 0
        for i, tm in enumerate(taylor_models):
            temp_poly += weights[i] * tm.poly
        temp_poly += bias
        
        # 论文关键：误差累积必须取绝对值（保守估计）
        temp_remainder = 0
        for i, tm in enumerate(taylor_models):
            temp_remainder += abs(weights[i]) * tm.remainder[1]
        
        remainder = [-temp_remainder, temp_remainder]
        
        # 确保是 Poly 对象
        if not isinstance(temp_poly, sym.Poly):
            if hasattr(taylor_models[0].poly, 'gens') and taylor_models[0].poly.gens:
                temp_poly = sym.Poly(temp_poly, *taylor_models[0].poly.gens)
            else:
                temp_poly = sym.Poly(temp_poly)
        
        return TaylorModel(temp_poly, remainder)
    
    def constant_product(self, taylor_model, constant):
        """
        常数乘法：c * TM
        
        Args:
            taylor_model: TM
            constant: c
        
        Returns:
            TaylorModel
        """
        new_poly = constant * taylor_model.poly
        new_remainder = [
            constant * taylor_model.remainder[0],
            constant * taylor_model.remainder[1]
        ]
        
        # 确保是 Poly 对象
        if not isinstance(new_poly, sym.Poly):
            if hasattr(taylor_model.poly, 'gens') and taylor_model.poly.gens:
                new_poly = sym.Poly(new_poly, *taylor_model.poly.gens)
            else:
                new_poly = sym.Poly(new_poly)
        
        return TaylorModel(new_poly, new_remainder)


class BernsteinPolynomial:
    """
    Bernstein 多项式逼近激活函数
    论文 Equation (1), (2)
    """
    
    def __init__(self, error_steps=4000):
        """
        Args:
            error_steps: 误差估计采样点数（论文中用 4000）
        """
        self.error_steps = error_steps
        self.bern_poly = None
    
    def approximate(self, a, b, order, activation_name):
        """
        论文 Equation (1): Bernstein 多项式逼近
        
        p_σ = Σᵢ₌₀ᵏ σ(a + (b-a)/k * i) * C(k,i) * Bᵢ(y)
        
        Args:
            a, b: 输入区间 [a, b]
            order: Bernstein 多项式阶数
            activation_name: 'relu' 或 'tanh'
        
        Returns:
            sympy.Poly: Bernstein 多项式
        """
        # 自适应阶数选择
        d_max = 8
        d_p = np.floor(d_max / math.log10(1 / (b - a)))
        d_p = np.abs(d_p)
        if d_p < 2:
            d_p = 2
        d = min(order, d_p)
        
        # 映射系数：将 [a, b] 映射到 [0, 1]
        coe1_1 = -a / (b - a)
        coe1_2 = 1 / (b - a)
        coe2_1 = b / (b - a)
        coe2_2 = -1 / (b - a)
        
        x = sym.Symbol('x')
        bern_poly = 0 * x
        
        # Bernstein 基函数展开
        for v in range(int(d) + 1):
            c = ncr(int(d), v)  # 组合数
            point = a + (b - a) / d * v
            
            # 计算激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            else:
                raise ValueError(f"不支持的激活函数: {activation_name}")
            
            # Bernstein 基
            basis = (
                ((coe1_2 * x + coe1_1) ** v) * 
                ((coe2_1 + coe2_2 * x) ** (d - v))
            )
            
            bern_poly += c * f_value * basis
        
        if bern_poly == 0:
            bern_poly = 1e-16 * x
        
        self.bern_poly = bern_poly
        return sym.Poly(bern_poly)
    
    def compute_error(self, a, b, activation_name):
        """
        论文 Equation (2): 计算 Bernstein 近似的误差上界
        
        ε = max_{i=0,...,m} |p_σ(sᵢ) - σ(sᵢ)| + (b-a)/m
        
        Args:
            a, b: 输入区间
            activation_name: 激活函数名称
        
        Returns:
            float: 误差上界
        """
        epsilon = 0
        m = self.error_steps
        
        for v in range(m + 1):
            # 采样点
            point = a + (b - a) / m * (v + 0.5)
            
            # 真实激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            
            # Bernstein 多项式值
            b_value = sym.Poly(self.bern_poly).eval(point)
            
            # 更新最大误差
            temp_diff = abs(f_value - b_value)
            epsilon = max(epsilon, temp_diff)
        
        # 加上离散化误差
        return epsilon + (b - a) / m


def compute_tm_bounds(tm):
    """
    论文 Equation (7): 计算 Taylor 模型的上下界（Minkowski 加法）
    
    对于 TM = p(z) + I, z ∈ [-1, 1]ⁿ，计算:
    - lower = min p(z) + I⁻
    - upper = max p(z) + I⁺
    
    Args:
        tm: TaylorModel
    
    Returns:
        (float, float): (lower_bound, upper_bound)
    """
    poly = tm.poly
    
    temp_upper = 0
    temp_lower = 0
    
    # 遍历多项式的每一项
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        
        if sum(monom) == 0:  # 常数项
            temp_upper += coeff
            temp_lower += coeff
        else:  # 变量项
            # z ∈ [-1, 1]，保守估计：取绝对值
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    # Minkowski 加法：P ⊕ I
    a = temp_lower + tm.remainder[0]
    b = temp_upper + tm.remainder[1]
    
    return float(a), float(b)


def compute_poly_bounds(poly):
    """
    计算多项式的上下界（不考虑误差区间）
    
    Args:
        poly: sympy.Poly
    
    Returns:
        (float, float): (lower_bound, upper_bound)
    """
    temp_upper = 0
    temp_lower = 0
    
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        
        if sum(monom) == 0:
            temp_upper += coeff
            temp_lower += coeff
        else:
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    return float(temp_lower), float(temp_upper)


def apply_activation(tm, bern_poly, bern_error, max_order):
    """
    通过 Bernstein 多项式传播 Taylor 模型过激活函数
    
    步骤：
    1. 多项式合成：p_out = p_bern(p_in)
    2. 截断到指定阶数
    3. 计算截断误差
    4. 累积 Bernstein 近似误差
    
    Args:
        tm: 输入 TaylorModel
        bern_poly: Bernstein 多项式
        bern_error: Bernstein 近似误差
        max_order: 最大保留阶数
    
    Returns:
        TaylorModel: 输出 Taylor 模型
    """
    # 1. 多项式合成
    composed = sym.polys.polytools.compose(bern_poly, tm.poly)
    
    # 2. 截断到 max_order
    poly_truncated = 0
    for i in range(len(composed.monoms())):
        monom = composed.monoms()[i]
        if sum(monom) <= max_order:
            temp = 1
            for j in range(len(monom)):
                temp *= composed.gens[j] ** monom[j]
            poly_truncated += composed.coeffs()[i] * temp
    
    poly_truncated = sym.Poly(poly_truncated)
    
    # 3. 计算截断误差
    poly_remainder = composed - poly_truncated
    _, truncation_error = compute_poly_bounds(poly_remainder)
    
    # 4. 计算总误差
    total_remainder = 0
    
    # Bernstein 多项式对输入误差的传播
    for i in range(len(bern_poly.monoms())):
        monom = bern_poly.monoms()[i]
        degree = sum(monom)
        
        if degree < 1:
            continue
        elif degree == 1:
            total_remainder += abs(bern_poly.coeffs()[i] * tm.remainder[1])
        else:
            # 高阶项：误差会被放大
            total_remainder += abs(bern_poly.coeffs()[i] * (tm.remainder[1] ** degree))
    
    # 总误差 = 传播误差 + 截断误差 + Bernstein误差
    remainder = [
        -total_remainder - truncation_error - bern_error,
        total_remainder + truncation_error + bern_error
    ]
    
    return TaylorModel(poly_truncated, remainder)