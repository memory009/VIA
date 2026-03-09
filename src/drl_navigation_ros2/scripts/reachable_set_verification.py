#!/usr/bin/env python3
"""
Parallel Reachable Set Verification using POLAR

Supported model types:
  - TD3_Lightweight : baseline model
  - TD3_VIA         : VIA safety model (e_t is read automatically from the checkpoint)

Network reachable sets are computed via Taylor Model arithmetic with Bernstein
polynomial approximation (POLAR), then checked against obstacle geometry.
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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from TD3.TD3_VIA import TD3_VIA


def point_to_box_distance(point, box_center, box_size, box_yaw=0.0):
    """Shortest signed distance from a point to an axis-aligned (or rotated) box.

    Args:
        point      : (x, y)
        box_center : [x, y] center of the box
        box_size   : [width, height, _]
        box_yaw    : rotation angle in radians

    Returns:
        float: distance in metres; negative means the point is inside the box.
    """
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
    """Shortest signed distance from a point to a circle.

    Args:
        point         : (x, y)
        circle_center : [x, y] center of the circle
        radius        : float

    Returns:
        float: distance in metres; negative means the point is inside the circle.
    """
    px, py = point
    if isinstance(circle_center, (list, tuple)):
        cx, cy = circle_center[0], circle_center[1]
    else:
        cx, cy = circle_center, circle_center

    return np.sqrt((px - cx)**2 + (py - cy)**2) - radius


def compute_robot_swept_area(pose, v_max, omega_max, dt, robot_radius=0.17, n_samples=10):
    """Sample boundary points of the area swept by the robot over one time step."""
    x0, y0, theta0 = pose

    critical_points = []

    angles_to_check = [0, np.pi/2, np.pi, -np.pi/2]
    for angle_offset in angles_to_check:
        edge_angle = theta0 + angle_offset
        critical_points.append((
            x0 + robot_radius * np.cos(edge_angle),
            y0 + robot_radius * np.sin(edge_angle),
        ))

    for t in np.linspace(0, dt, n_samples):
        theta_t = theta0 + omega_max * t
        x_t = x0 + v_max * np.cos(theta0) * t
        y_t = y0 + v_max * np.sin(theta0) * t

        for angle_offset in angles_to_check:
            edge_angle = theta_t + angle_offset
            critical_points.append((
                x_t + robot_radius * np.cos(edge_angle),
                y_t + robot_radius * np.sin(edge_angle),
            ))

    theta_final = theta0 + omega_max * dt
    x_final = x0 + v_max * np.cos(theta0) * dt
    y_final = y0 + v_max * np.sin(theta0) * dt

    for angle_offset in angles_to_check:
        edge_angle = theta_final + angle_offset
        critical_points.append((
            x_final + robot_radius * np.cos(edge_angle),
            y_final + robot_radius * np.sin(edge_angle),
        ))

    return critical_points


def compute_reachable_set_pure_polar(
    actor,
    state,
    observation_error=0.01,
    bern_order=1,
    error_steps=4000,
    max_action=1.0,
    e_t=None,  # required for TD3_VIA; pass None for TD3_Lightweight
):
    """Compute the action reachable set using POLAR (Taylor Model + Bernstein approximation).

    Automatically adapts to the network's hidden dimension.
    For TD3_VIA, e_t is appended to the state as a deterministic input.

    Returns:
        list of [lo, hi] intervals for each action dimension.
    """
    import sympy as sym
    from verification.taylor_model import (
        TaylorModel,
        TaylorArithmetic,
        BernsteinPolynomial,
        compute_tm_bounds,
        apply_activation,
    )

    # Extract actor weights
    weights = []
    biases = []
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if 'weight' in name:
                weights.append(param.cpu().numpy())
            elif 'bias' in name:
                biases.append(param.cpu().numpy())

    state_dim = len(state)
    input_dim = state_dim + 1 if e_t is not None else state_dim

    assert weights[0].shape[1] == input_dim, \
        f"Input dimension mismatch: expected {input_dim}, got {weights[0].shape[1]}"
    assert weights[-1].shape[0] == 2, \
        f"Output dimension mismatch: expected 2, got {weights[-1].shape[0]}"

    hidden_dim = weights[0].shape[0]  # inferred from network

    # Symbolic variables
    z_symbols = [sym.Symbol(f'z{i}') for i in range(input_dim)]

    # Build input Taylor models (state components with observation error)
    TM_state = []
    for i in range(state_dim):
        poly = sym.Poly(observation_error * z_symbols[i] + state[i], *z_symbols)
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))

    # e_t is treated as a deterministic constant (no observation error)
    if e_t is not None:
        poly = sym.Poly(e_t, *z_symbols)
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))

    # Forward propagation through layers
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
            tm_neuron = TA.weighted_sumforall(TM_input, W[neuron_idx], b[neuron_idx])

            is_hidden = (layer_idx < num_layers - 1)

            if is_hidden:
                # ReLU activation
                a, b_bound = compute_tm_bounds(tm_neuron)
                if a >= 0:
                    TM_after = tm_neuron
                elif b_bound <= 0:
                    zero_poly = sym.Poly(0, *z_symbols)
                    TM_after = TaylorModel(zero_poly, [0, 0])
                else:
                    bern_poly = BP.approximate(a, b_bound, bern_order, 'relu')
                    bern_error = BP.compute_error(a, b_bound, 'relu')
                    TM_after = apply_activation(tm_neuron, bern_poly, bern_error, bern_order)
            else:
                # Output layer: Tanh
                a, b_bound = compute_tm_bounds(tm_neuron)
                bern_poly = BP.approximate(a, b_bound, bern_order, 'tanh')
                bern_error = BP.compute_error(a, b_bound, 'tanh')
                TM_after = apply_activation(tm_neuron, bern_poly, bern_error, bern_order)
                TM_after = TA.constant_product(TM_after, max_action)

            TM_temp.append(TM_after)

        TM_input = TM_temp

    # Compute action bounds
    action_ranges = []
    for tm in TM_input:
        a, b = compute_tm_bounds(tm)
        action_ranges.append([a, b])

    return action_ranges


def check_action_safety_geometric_complete(action_ranges, state, pose, obstacle_map):
    """Check whether all actions in the reachable set are collision-free.

    Samples multiple (v, omega) pairs from the reachable intervals and checks
    the robot's swept area against every obstacle in the map.

    Returns:
        bool: True if all sampled actions are safe, False otherwise.
    """
    COLLISION_DELTA = 0.4
    SAFETY_MARGIN = 0.0
    DT = 0.1
    ROBOT_RADIUS = 0.17
    N_ACTION_SAMPLES = 3
    N_TRAJECTORY_SAMPLES = 8

    safe_threshold = COLLISION_DELTA + SAFETY_MARGIN

    v_interval = action_ranges[0]
    omega_interval = action_ranges[1]

    v_min = (v_interval[0] + 1) / 2
    v_max = (v_interval[1] + 1) / 2
    omega_min = omega_interval[0]
    omega_max = omega_interval[1]

    v_samples = np.linspace(v_min, v_max, N_ACTION_SAMPLES)
    omega_samples = np.linspace(omega_min, omega_max, N_ACTION_SAMPLES)

    for v_test in v_samples:
        for omega_test in omega_samples:
            swept_area_points = compute_robot_swept_area(
                pose, v_test, abs(omega_test), DT, ROBOT_RADIUS, N_TRAJECTORY_SAMPLES
            )

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
                        dist = point_to_circle_distance(point, obs['position'], radius)
                    else:
                        continue

                    if dist < safe_threshold:
                        return False

    return True


def verify_single_trajectory_worker(args):
    """Worker function for parallel reachable set verification of a single trajectory."""
    trajectory_idx, trajectory_data, model_path, model_type, model_name, \
        observation_error, sample_interval, e_t = args

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "TD3_Lightweight":
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

    elif model_type == "TD3_VIA":
        agent = TD3_VIA(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            hidden_dim=26,
            load_model=False,
            save_directory=model_path,
            model_name=model_name,
            run_id="polar_verification_via",
        )
        agent.load(filename=model_name, directory=str(model_path))

        # Use var_u from the checkpoint as e_t if not specified on the command line
        if e_t is None or e_t == 0.0:
            e_t = agent.var_u.item()
            print(f"[Worker {trajectory_idx+1}] var_u read from checkpoint: {e_t:.4f}")

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose TD3_Lightweight or TD3_VIA.")

    # Load obstacle map
    obstacle_map_path = (
        project_root / "assets" / "eval_scenarios_8_polar" /
        f"obstacle_map_scenario_{trajectory_idx:02d}.json"
    )

    if obstacle_map_path.exists():
        with open(obstacle_map_path, 'r') as f:
            obstacle_map = json.load(f)
        print(f"[Worker {trajectory_idx+1}] Loaded obstacle map: {obstacle_map_path.name} "
              f"({obstacle_map['total_obstacles']} obstacles)")
    else:
        raise FileNotFoundError(
            f"Obstacle map not found: {obstacle_map_path}\n"
            f"Run scripts/export_gazebo_map.py first."
        )

    # Sample trajectory states
    states = trajectory_data['states']
    poses = trajectory_data['poses']

    sampled_states = states[::sample_interval]
    sampled_poses = poses[::sample_interval]
    n_samples = len(sampled_states)

    print(f"[Worker {trajectory_idx+1}] Verifying {n_samples} sampled states...")

    # Point-wise verification
    results = []
    safe_count = 0
    start_time = time.time()

    for i, (state, pose) in enumerate(zip(sampled_states, sampled_poses)):
        step_idx = i * sample_interval

        if i % max(1, n_samples // 4) == 0:
            elapsed = time.time() - start_time
            print(f"[Worker {trajectory_idx+1}] {i+1}/{n_samples} "
                  f"({i/n_samples*100:.0f}%) | elapsed: {elapsed/60:.1f} min")

        action_ranges = compute_reachable_set_pure_polar(
            agent.actor,
            state,
            observation_error=observation_error,
            bern_order=1,
            error_steps=4000,
            max_action=1.0,
            e_t=e_t if model_type == "TD3_VIA" else None,
        )

        is_safe = check_action_safety_geometric_complete(
            action_ranges, state, pose, obstacle_map
        )

        if model_type == "TD3_VIA":
            det_action = agent.get_action(state, e_t, add_noise=False)
        else:
            det_action = agent.get_action(state, add_noise=False)

        width_v = action_ranges[0][1] - action_ranges[0][0]
        width_omega = action_ranges[1][1] - action_ranges[1][0]

        if is_safe:
            safe_count += 1

        results.append({
            'step': step_idx,
            'pose': pose.tolist(),
            'det_action': det_action.tolist(),
            'action_ranges': action_ranges,
            'is_safe': is_safe,
            'width_v': float(width_v),
            'width_omega': float(width_omega),
            'min_laser': float(np.min(state[0:20])),
            'distance': float(state[20]),
        })

    elapsed_time = time.time() - start_time
    safety_rate = safe_count / n_samples if n_samples > 0 else 0

    print(f"[Worker {trajectory_idx+1}] Done. Safety rate: {safety_rate*100:.1f}% | "
          f"time: {elapsed_time/60:.1f} min")

    return (trajectory_idx, {
        'trajectory_idx': trajectory_idx,
        'n_samples': n_samples,
        'safe_count': safe_count,
        'safety_rate': safety_rate,
        'collision': trajectory_data['collision'],
        'goal_reached': trajectory_data['goal_reached'],
        'steps': trajectory_data['steps'],
        'total_reward': float(trajectory_data['total_reward']),
        'compute_time': elapsed_time,
        'results': results,
        'obstacle_map_file': obstacle_map_path.name,
        'obstacle_count': obstacle_map['total_obstacles'],
        'e_t_used': e_t if model_type == "TD3_VIA" else None,
    })


def load_trajectories(pkl_path):
    """Load trajectories from a .pkl file produced by collect_trajectories.py."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        trajectories = pickle.load(f)

    return [t for t in trajectories if t is not None]


def main():
    """Entry point for parallel POLAR reachable set verification."""
    parser = argparse.ArgumentParser(
        description='Parallel POLAR reachable set verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify TD3 baseline
  python reachable_set_verification.py --model-type TD3_Lightweight

  # Verify VIA (e_t read automatically from checkpoint)
  python reachable_set_verification.py --model-type TD3_VIA

  # Verify VIA with a manually specified e_t
  python reachable_set_verification.py --model-type TD3_VIA --e-t 5.0
        """
    )
    parser.add_argument('--version', type=str, default='v1',
                        help='Trajectory version tag (default: v1)')
    parser.add_argument('--model-type', type=str, default='TD3_Lightweight',
                        choices=['TD3_Lightweight', 'TD3_VIA'],
                        help='Model type (default: TD3_Lightweight)')
    parser.add_argument('--e-t', type=float, default=0.0,
                        help='e_t value for TD3_VIA (default: 0.0 = read var_u from checkpoint)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("POLAR Parallel Reachable Set Verification")
    print("="*70)

    n_cores = cpu_count()
    print(f"\nDetected CPU cores: {n_cores}")

    model_type = args.model_type
    trajectory_version = args.version
    e_t = args.e_t

    print(f"\nConfiguration:")
    print(f"  Model type : {model_type}")
    print(f"  Trajectory : version {trajectory_version}")
    if model_type == "TD3_VIA":
        if e_t == 0.0:
            print(f"  e_t        : read from checkpoint (var_u)")
        else:
            print(f"  e_t        : {e_t} (manually specified)")

    # =========================================================================
    # ⚠️  User Configuration
    #
    # Replace the paths below with your own trained model and trajectory paths:
    #   - model_name      : filename prefix used when saving the model (without extension)
    #   - model_path      : directory containing the model weights
    #                       (the run directory output by train.py / train_VIA.py)
    #   - trajectory_path : path to the trajectory .pkl file produced by
    #                       collect_trajectories.py
    # =========================================================================
    if model_type == "TD3_Lightweight":
        model_name = "TD3_lightweight_best"  # e.g. "TD3_lightweight_best"
        model_path = project_root / "models" / "TD3_lightweight" / "<your_run_id>"
        trajectory_path = project_root / "assets" / f"<your_trajectory_file>_{trajectory_version}.pkl"
    elif model_type == "TD3_VIA":
        model_name = "TD3_VIA_best"  # e.g. "TD3_VIA_best"
        model_path = project_root / "models" / "TD3_VIA" / "<your_run_id>"
        trajectory_path = project_root / "assets" / f"<your_trajectory_file>_{trajectory_version}.pkl"

    print("\n[1/3] Loading trajectories...")
    print(f"  File: {trajectory_path}")
    trajectories = load_trajectories(pkl_path=trajectory_path)
    n_trajectories = len(trajectories)
    total_states = sum(t['steps'] for t in trajectories)
    print(f"  Loaded {n_trajectories} trajectories ({total_states} total states)")

    print("\n[2/3] Preparing parallel workers...")

    observation_error = 0.01
    sample_interval = 1

    n_workers = min(n_trajectories, n_cores // 2)
    print(f"  Model      : {model_type} / {model_name}")
    print(f"  Model path : {model_path}")
    if model_type == "TD3_VIA":
        print(f"  e_t        : {e_t}")
    print(f"  Workers    : {n_workers}")
    print(f"  Obs. error : ±{observation_error}")
    print(f"  Interval   : every {sample_interval} step(s)")

    args_list = [
        (i, traj, model_path, model_type, model_name, observation_error, sample_interval, e_t)
        for i, traj in enumerate(trajectories)
    ]

    print(f"\n[3/3] Starting {n_workers} parallel workers...")
    print("="*70)

    start_time = time.time()

    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(verify_single_trajectory_worker, args_list)
    except Exception as e:
        print(f"\nError during parallel verification: {e}")
        import traceback
        traceback.print_exc()
        raise

    total_elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)

    results = sorted(results, key=lambda x: x[0])
    all_results = [r[1] for r in results]

    total_samples = sum(r['n_samples'] for r in all_results)
    total_safe = sum(r['safe_count'] for r in all_results)
    overall_safety_rate = total_safe / total_samples if total_samples > 0 else 0

    print(f"\nReachable set safety:")
    print(f"  Total states : {total_samples}")
    print(f"  Safe states  : {total_safe}")
    print(f"  Safety rate  : {overall_safety_rate*100:.1f}%")

    goal_trajectories = [r for r in all_results if r['goal_reached']]
    collision_trajectories = [r for r in all_results if r['collision']]

    print(f"\nBy trajectory outcome:")
    print(f"  Goal reached : {len(goal_trajectories)}")
    if goal_trajectories:
        print(f"    Avg safety rate: {np.mean([r['safety_rate'] for r in goal_trajectories])*100:.1f}%")
    print(f"  Collision    : {len(collision_trajectories)}")
    if collision_trajectories:
        print(f"    Avg safety rate: {np.mean([r['safety_rate'] for r in collision_trajectories])*100:.1f}%")

    all_widths_v = [r['width_v'] for res in all_results for r in res['results']]
    all_widths_omega = [r['width_omega'] for res in all_results for r in res['results']]

    print(f"\nReachable set width — linear velocity:")
    print(f"  min={np.min(all_widths_v):.6f}  mean={np.mean(all_widths_v):.6f}  "
          f"median={np.median(all_widths_v):.6f}  std={np.std(all_widths_v):.6f}  "
          f"max={np.max(all_widths_v):.6f}  p95={np.percentile(all_widths_v, 95):.6f}")

    print(f"Reachable set width — angular velocity:")
    print(f"  min={np.min(all_widths_omega):.6f}  mean={np.mean(all_widths_omega):.6f}  "
          f"median={np.median(all_widths_omega):.6f}  std={np.std(all_widths_omega):.6f}  "
          f"max={np.max(all_widths_omega):.6f}  p95={np.percentile(all_widths_omega, 95):.6f}")

    avg_traj_time = np.mean([r['compute_time'] for r in all_results])
    serial_time = avg_traj_time * n_trajectories
    speedup = serial_time / total_elapsed

    print(f"\nPerformance:")
    print(f"  Total time     : {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    print(f"  Per trajectory : {total_elapsed/n_trajectories:.1f} s")
    print(f"  Per state      : {total_elapsed/total_samples:.2f} s")
    print(f"  Speedup        : {speedup:.1f}x  (efficiency: {speedup/n_workers*100:.1f}%)")

    # Build output filename
    if model_type == "TD3_VIA":
        actual_e_t = all_results[0].get('e_t_used', 0.0)
        e_t_str = f"{actual_e_t:.4f}".replace('.', 'p')
        output_filename = f"reachability_results_pure_polar_td3_via_varu{e_t_str}_{trajectory_version}.json"
        print(f"\n  var_u used: {actual_e_t:.4f}")
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
            'e_t': all_results[0].get('e_t_used') if model_type == "TD3_VIA" else None,
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
        print(f"\nResults saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"\nFailed to save results: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("="*70)
    print("Reachable set verification complete.")


if __name__ == "__main__":
    main()
