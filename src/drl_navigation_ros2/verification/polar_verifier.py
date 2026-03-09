#!/usr/bin/env python3
"""
POLAR reachability verifier for the TD3_lightweight actor network (25 -> 26 -> 26 -> 2).
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
from .ray_casting import get_obstacle_map


def extract_actor_weights(actor):
    """
    Extract weights and biases from an Actor network.

    Returns:
        weights : list of numpy arrays
        biases  : list of numpy arrays
    """
    weights, biases = [], []
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
    Full POLAR reachability computation for the actor network.

    Each input dimension is bounded by state[i] +/- observation_error,
    encoded as TM_i = observation_error * z_i + state[i], z_i in [-1, 1].
    Taylor models are propagated layer by layer; ReLU uses the three-case
    optimisation (eq. 8) and the output Tanh uses Bernstein approximation.

    Args:
        actor            : Actor network instance
        state            : current state vector (length state_dim)
        observation_error: half-width of the input uncertainty set
        bern_order       : Bernstein polynomial order
        error_steps      : sampling points for Bernstein error estimation
        max_action       : action scaling factor (applied after Tanh)

    Returns:
        action_ranges : list of [min, max] for each action dimension
    """
    weights, biases = extract_actor_weights(actor)

    state_dim = len(state)
    z_symbols = [sym.Symbol(f'z{i}') for i in range(state_dim)]

    # Build input Taylor models: x_i = state[i] + eps * z_i, z_i in [-1, 1]
    TM_state = [
        TaylorModel(
            sym.Poly(observation_error * z_symbols[i] + state[i], *z_symbols),
            [0.0, 0.0]
        )
        for i in range(state_dim)
    ]

    TM_input = TM_state
    TA = TaylorArithmetic()
    BP = BernsteinPolynomial(error_steps=error_steps)
    num_layers = len(biases)

    for layer_idx in range(num_layers):
        W = weights[layer_idx]
        b = biases[layer_idx]
        TM_temp = []

        for neuron_idx in range(len(b)):
            tm_neuron = TA.weighted_sumforall(TM_input, W[neuron_idx], b[neuron_idx])

            is_hidden = (layer_idx < num_layers - 1)

            if is_hidden:
                # ReLU with three-case optimisation
                a, b_bound = compute_tm_bounds(tm_neuron)
                if a >= 0:
                    TM_after = tm_neuron
                elif b_bound <= 0:
                    TM_after = TaylorModel(sym.Poly(0, *z_symbols), [0, 0])
                else:
                    bern_poly = BP.approximate(a, b_bound, bern_order, 'relu')
                    bern_error = BP.compute_error(a, b_bound, 'relu')
                    TM_after = apply_activation(tm_neuron, bern_poly, bern_error, bern_order)
            else:
                # Output layer: Tanh + action scaling
                a, b_bound = compute_tm_bounds(tm_neuron)
                bern_poly = BP.approximate(a, b_bound, bern_order, 'tanh')
                bern_error = BP.compute_error(a, b_bound, 'tanh')
                TM_after = apply_activation(tm_neuron, bern_poly, bern_error, bern_order)
                TM_after = TA.constant_product(TM_after, max_action)

            TM_temp.append(TM_after)

        TM_input = TM_temp

    action_ranges = [list(compute_tm_bounds(tm)) for tm in TM_input]
    return action_ranges


def check_action_safety(action_ranges, current_pose, current_laser, dt=0.1, collision_threshold=0.4):
    """
    Check whether all reachable next states are collision-free.

    Tests the four corner cases of the action range box using unicycle kinematics
    and ray-casting to predict the next laser scan.

    Args:
        action_ranges      : [[v_min, v_max], [omega_min, omega_max]]
        current_pose       : (x, y, theta)
        current_laser      : current laser readings (used for context)
        dt                 : time step
        collision_threshold: minimum safe distance (m)

    Returns:
        bool: True if all corners are collision-free
    """
    x, y, theta = current_pose
    v_min, v_max = action_ranges[0]
    omega_min, omega_max = action_ranges[1]

    obstacle_map = get_obstacle_map()

    for v in [v_min, v_max]:
        for omega in [omega_min, omega_max]:
            v_real = v * 0.5  # TurtleBot3 velocity scaling
            x_next = x + dt * v_real * np.cos(theta)
            y_next = y + dt * v_real * np.sin(theta)
            theta_next = theta + omega * dt

            laser_next = obstacle_map.predict_laser_scan(x_next, y_next, theta_next)
            if np.min(laser_next) < collision_threshold:
                return False

            safe_zone = obstacle_map.boundary['robot_safe_zone']
            if (x_next < safe_zone['x_min'] or x_next > safe_zone['x_max'] or
                    y_next < safe_zone['y_min'] or y_next > safe_zone['y_max']):
                return False

    return True


def verify_safety(agent, state, current_pose, observation_error=0.01, **kwargs):
    """Full pipeline: compute reachable set then check collision safety."""
    max_action = getattr(agent, 'max_action', 1.0)

    action_ranges = compute_reachable_set(
        agent.actor, state,
        observation_error=observation_error,
        max_action=max_action,
        **kwargs
    )

    is_safe = check_action_safety(
        action_ranges,
        current_pose,
        state[:20],
    )
    return is_safe, action_ranges
