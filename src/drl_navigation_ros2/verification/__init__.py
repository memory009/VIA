#!/usr/bin/env python3
"""
POLAR reachability verification module.
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

from .ray_casting import (
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
    'ObstacleMap',
    'get_obstacle_map',
]