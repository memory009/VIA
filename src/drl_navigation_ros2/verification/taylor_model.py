#!/usr/bin/env python3
"""
Taylor Model 和 POLAR 可达性验证核心模块
严格遵循论文 "Reachability Verification Based Reliability Assessment for DRL"

修正版本：
- 修复 weighted_sumforall 的误差累积逻辑
- 封装 ReLU 三段式优化（Equation 8）
- 性能优化：缓存 lambdify 函数
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
    
    论文 Remark 2:
    - p(z): 多项式部分，代表"中心"（类似 norm ball 的中心）
    - I = [I⁻, I⁺]: remainder，代表"半径"
    """
    def __init__(self, poly, remainder):
        self.poly = poly            # sympy.Poly 对象
        self.remainder = remainder  # [lower, upper]
    
    def __repr__(self):
        return f"TaylorModel(poly={self.poly}, remainder={self.remainder})"


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
        # ===== 1. 多项式求和 =====
        temp_poly = 0
        for i, tm in enumerate(taylor_models):
            temp_poly += weights[i] * tm.poly
        temp_poly += bias
        
        # ===== 2. 误差累积（修正版）=====
        # ✅ 修正：考虑区间的最大幅度，而非只用上界
        temp_remainder = 0
        for i, tm in enumerate(taylor_models):
            # 取 remainder 区间的最大绝对值（保守估计）
            remainder_mag = max(abs(tm.remainder[0]), abs(tm.remainder[1]))
            temp_remainder += abs(weights[i]) * remainder_mag
        
        remainder = [-temp_remainder, temp_remainder]
        
        # ===== 3. 确保是 Poly 对象 =====
        if not isinstance(temp_poly, sym.Poly):
            if hasattr(taylor_models[0].poly, 'gens') and taylor_models[0].poly.gens:
                temp_poly = sym.Poly(temp_poly, *taylor_models[0].poly.gens)
            else:
                temp_poly = sym.Poly(temp_poly)
        
        return TaylorModel(temp_poly, remainder)
    
    def add(self, tm1, tm2):
        """
        Taylor Model 加法：TM₁ + TM₂
        
        Args:
            tm1, tm2: TaylorModel
        
        Returns:
            TaylorModel
        """
        new_poly = tm1.poly + tm2.poly
        new_remainder = [
            tm1.remainder[0] + tm2.remainder[0],
            tm1.remainder[1] + tm2.remainder[1]
        ]
        
        if not isinstance(new_poly, sym.Poly):
            if hasattr(tm1.poly, 'gens') and tm1.poly.gens:
                new_poly = sym.Poly(new_poly, *tm1.poly.gens)
            else:
                new_poly = sym.Poly(new_poly)
        
        return TaylorModel(new_poly, new_remainder)
    
    def product(self, tm1, tm2):
        """
        Taylor Model 乘法：TM₁ × TM₂
        
        ⚠️ 这是简化实现，完整版需要考虑高阶项截断
        
        Args:
            tm1, tm2: TaylorModel 或 scalar
        
        Returns:
            TaylorModel
        """
        # 处理标量乘法
        if isinstance(tm2, (int, float)):
            return self.constant_product(tm1, tm2)
        if isinstance(tm1, (int, float)):
            return self.constant_product(tm2, tm1)
        
        # Taylor Model 乘法
        new_poly = tm1.poly * tm2.poly
        
        # 误差传播（简化）
        # 完整公式见论文，这里用保守估计
        poly1_bound = compute_poly_bounds(tm1.poly)
        poly2_bound = compute_poly_bounds(tm2.poly)
        
        remainder_mag1 = max(abs(tm1.remainder[0]), abs(tm1.remainder[1]))
        remainder_mag2 = max(abs(tm2.remainder[0]), abs(tm2.remainder[1]))
        
        # |TM₁ × TM₂| ≤ |poly₁| × |I₂| + |poly₂| × |I₁| + |I₁| × |I₂|
        error = (
            max(abs(poly1_bound[0]), abs(poly1_bound[1])) * remainder_mag2 +
            max(abs(poly2_bound[0]), abs(poly2_bound[1])) * remainder_mag1 +
            remainder_mag1 * remainder_mag2
        )
        
        new_remainder = [-error, error]
        
        if not isinstance(new_poly, sym.Poly):
            if hasattr(tm1.poly, 'gens') and tm1.poly.gens:
                new_poly = sym.Poly(new_poly, *tm1.poly.gens)
            else:
                new_poly = sym.Poly(new_poly)
        
        return TaylorModel(new_poly, new_remainder)
    
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
        self._poly_func = None  # ✅ 新增：缓存编译后的函数
    
    def approximate(self, a, b, order, activation_name):
        """
        论文 Equation (1): Bernstein 多项式逼近
        
        p_σ = Σᵢ₌₀ᵏ σ(a + (b-a)/k * i) * C(k,i) * Bᵢ(y)
        
        Args:
            a, b: 输入区间 [a, b]
            order: Bernstein 多项式阶数
            activation_name: 'relu', 'tanh', 'cos', 'sin'
        
        Returns:
            sympy.Poly: Bernstein 多项式
        """
        # ===== 1. 自适应阶数选择 =====
        d_max = 8
        interval_width = b - a
        
        if interval_width > 1e-8:  # 避免除零
            d_p = np.floor(d_max / math.log10(1 / interval_width))
            d_p = np.abs(d_p)
        else:
            d_p = order
        
        if d_p < 2:
            d_p = 2
        d = min(order, d_p)
        
        # ===== 2. 映射系数：将 [a, b] 映射到 [0, 1] =====
        # y_normalized = (x - a) / (b - a)
        # 1 - y_normalized = (b - x) / (b - a)
        coe1_1 = -a / (b - a)      # (x - a) / (b - a) 的常数项
        coe1_2 = 1 / (b - a)       # (x - a) / (b - a) 的 x 系数
        coe2_1 = b / (b - a)       # (b - x) / (b - a) 的常数项
        coe2_2 = -1 / (b - a)      # (b - x) / (b - a) 的 x 系数
        
        x = sym.Symbol('x')
        bern_poly = 0 * x
        
        # ===== 3. Bernstein 基函数展开 =====
        for v in range(int(d) + 1):
            c = ncr(int(d), v)  # 组合数 C(d, v)
            point = a + (b - a) / d * v  # 控制点
            
            # 计算激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            elif activation_name == 'cos':
                f_value = np.cos(float(point))
            elif activation_name == 'sin':
                f_value = np.sin(float(point))
            else:
                raise ValueError(f"不支持的激活函数: {activation_name}")
            
            # Bernstein 基：B_v^d(y) = C(d,v) * y^v * (1-y)^(d-v)
            # 其中 y = (x - a) / (b - a)
            basis = (
                ((coe1_2 * x + coe1_1) ** v) *      # y^v
                ((coe2_1 + coe2_2 * x) ** (d - v))  # (1-y)^(d-v)
            )
            
            bern_poly += c * f_value * basis
        
        # 避免零多项式
        if bern_poly == 0:
            bern_poly = 1e-16 * x
        
        self.bern_poly = bern_poly
        
        # ✅ 新增：预编译 lambdify 函数（性能优化）
        self._poly_func = sym.lambdify(x, bern_poly, 'numpy')
        
        return sym.Poly(bern_poly, x)
    
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
        
        # ✅ 使用缓存的函数（性能优化）
        if self._poly_func is None:
            x = sym.Symbol('x')
            self._poly_func = sym.lambdify(x, self.bern_poly, 'numpy')
        
        for v in range(m + 1):
            # 采样点（区间中心）
            point = a + (b - a) / m * (v + 0.5)
            
            # 真实激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            elif activation_name == 'cos':
                f_value = np.cos(float(point))
            elif activation_name == 'sin':
                f_value = np.sin(float(point))
            
            # Bernstein 多项式值
            b_value = self._poly_func(point)
            
            # 更新最大误差
            temp_diff = abs(f_value - b_value)
            epsilon = max(epsilon, temp_diff)
        
        # 加上离散化误差
        return epsilon + (b - a) / m


def compute_tm_bounds(tm):
    """
    论文 Equation (7): 计算 Taylor 模型的上下界（Minkowski 加法）
    
    对于 TM = p(z) + I, z ∈ [-1, 1]ⁿ，计算:
    P ⊕ I = [min p(z) + I⁻, max p(z) + I⁺]
    
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
            # z ∈ [-1, 1]，保守估计：
            # max z^monom = 1, min z^monom = -1 (如果指数为奇数)
            # 简化：取绝对值（保守）
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    # Minkowski 加法：P ⊕ I
    a = temp_lower + tm.remainder[0]
    b = temp_upper + tm.remainder[1]
    
    return float(a), float(b)


def compute_poly_bounds(poly):
    """
    计算多项式的上下界（不考虑误差区间）
    用于乘法运算中的误差估计
    
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
    # ===== 1. 多项式合成 =====
    composed = sym.polys.polytools.compose(bern_poly, tm.poly)
    
    # ===== 2. 截断到 max_order =====
    poly_truncated = 0
    for i in range(len(composed.monoms())):
        monom = composed.monoms()[i]
        if sum(monom) <= max_order:
            temp = 1
            for j in range(len(monom)):
                temp *= composed.gens[j] ** monom[j]
            poly_truncated += composed.coeffs()[i] * temp
    
    poly_truncated = sym.Poly(poly_truncated, *composed.gens) if composed.gens else sym.Poly(poly_truncated)
    
    # ===== 3. 计算截断误差 =====
    poly_remainder = composed - poly_truncated
    _, truncation_error = compute_poly_bounds(poly_remainder)
    
    # ===== 4. 计算总误差 =====
    total_remainder = 0
    
    # Bernstein 多项式对输入误差的传播
    for i in range(len(bern_poly.monoms())):
        monom = bern_poly.monoms()[i]
        degree = sum(monom)
        
        if degree < 1:
            continue
        elif degree == 1:
            # 一阶项：线性传播
            total_remainder += abs(bern_poly.coeffs()[i] * tm.remainder[1])
        else:
            # 高阶项：误差会被放大
            remainder_mag = max(abs(tm.remainder[0]), abs(tm.remainder[1]))
            total_remainder += abs(bern_poly.coeffs()[i] * (remainder_mag ** degree))
    
    # 总误差 = 传播误差 + 截断误差 + Bernstein误差
    remainder = [
        -total_remainder - truncation_error - bern_error,
        total_remainder + truncation_error + bern_error
    ]
    
    return TaylorModel(poly_truncated, remainder)


def apply_relu_optimized(tm, bern_order=1, error_steps=4000):
    """
    ✅ 新增：论文 Equation (8) - ReLU 的三段式优化传播
    
    TMᵒ = {
        0,                        if b ≤ 0  (完全不激活)
        p_σ(TMⁱ) + Int(rₖ) + Iσ,  if a ≤ 0 & b ≥ 0  (跨越零点)
        TMⁱ,                      if a ≥ 0  (完全激活)
    }
    
    这个优化可以：
    - 减少计算时间（避免不必要的 Bernstein 采样）
    - 减少误差累积（完全激活/不激活时无额外误差）
    
    Args:
        tm: 输入 TaylorModel
        bern_order: Bernstein 多项式阶数（仅情况3使用）
        error_steps: 误差采样点数（仅情况3使用）
    
    Returns:
        TaylorModel: 输出 Taylor 模型
    """
    # 计算输入 TM 的界
    a, b = compute_tm_bounds(tm)
    
    # ===== 情况1: 完全激活 (a ≥ 0) =====
    if a >= 0:
        # ReLU(x) = x，无需近似，无额外误差
        return tm
    
    # ===== 情况2: 完全不激活 (b ≤ 0) =====
    if b <= 0:
        # ReLU(x) = 0
        zero_poly = sym.Poly(0, *tm.poly.gens) if tm.poly.gens else sym.Poly(0)
        return TaylorModel(zero_poly, [0.0, 0.0])
    
    # ===== 情况3: 跨越零点 (a < 0 < b) =====
    # 使用 Bernstein 多项式近似
    BP = BernsteinPolynomial(error_steps=error_steps)
    bern_poly = BP.approximate(a, b, bern_order, 'relu')
    bern_error = BP.compute_error(a, b, 'relu')
    return apply_activation(tm, bern_poly, bern_error, bern_order)


# ===== 辅助函数：用于动力学传播（未来扩展） =====

def create_initial_tm(state, observation_error):
    """
    从状态向量创建初始 Taylor 模型
    
    Args:
        state: numpy array, 状态向量
        observation_error: float, 观测误差半径
    
    Returns:
        list of TaylorModel
    """
    import sympy as sym
    
    state_dim = len(state)
    z_symbols = [sym.Symbol(f'z{i}') for i in range(state_dim)]
    
    TM_state = []
    for i in range(state_dim):
        # TM = (observation_error * z_i + state[i]) + [0, 0]
        # 其中 z_i ∈ [-1, 1]
        poly = sym.Poly(
            observation_error * z_symbols[i] + state[i],
            *z_symbols
        )
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))
    
    return TM_state


def tm_to_numpy(tm_list):
    """
    从 Taylor Model 列表提取中心点（用于神经网络输入）
    
    Args:
        tm_list: list of TaylorModel
    
    Returns:
        numpy array: 中心点
    """
    centers = []
    for tm in tm_list:
        # 提取常数项（即中心点）
        if len(tm.poly.monoms()) > 0:
            # 找到零次项
            for i, monom in enumerate(tm.poly.monoms()):
                if sum(monom) == 0:
                    centers.append(float(tm.poly.coeffs()[i]))
                    break
            else:
                # 没有常数项，说明中心在原点
                centers.append(0.0)
        else:
            centers.append(0.0)
    
    return np.array(centers)


def tm_to_interval(tm):
    """
    从 Taylor Model 提取区间
    
    Args:
        tm: TaylorModel
    
    Returns:
        [lower, upper]
    """
    a, b = compute_tm_bounds(tm)
    return [a, b]