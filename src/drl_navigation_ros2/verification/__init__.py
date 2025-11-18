#!/usr/bin/env python3
"""
POLAR 可达性验证模块
"""

from .taylor_model import (
    TaylorModel,
    TaylorArithmetic,
    BernsteinPolynomial,
    compute_tm_bounds,
    apply_activation,
)

from .polar_verifier import (
    extract_actor_weights,
    compute_reachable_set,
    check_action_safety,
    verify_safety,
)

from .ray_casting import (  # 新增
    ObstacleMap,
    get_obstacle_map,
)

__all__ = [
    'TaylorModel',
    'TaylorArithmetic',
    'BernsteinPolynomial',
    'compute_tm_bounds',
    'apply_activation',
    'extract_actor_weights',
    'compute_reachable_set',
    'check_action_safety',
    'verify_safety',
    'ObstacleMap',  # 新增
    'get_obstacle_map',  # 新增
]