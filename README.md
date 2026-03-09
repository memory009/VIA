# DRL Robot Navigation with Reachability-Based Safety Verification

Deep Reinforcement Learning for mobile robot navigation in ROS2/Gazebo, extended with VIA (CVaR-Constrained Policy Optimization) for safe navigation and POLAR-based reachable set verification.

<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2/blob/main/gif.gif">
</p>

## Attribution

This repository extends the following open-source work:

- **Base codebase**: [reiniscimurs/DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) — ROS2 DRL navigation framework (TD3/SAC, Gazebo)
- **ROS2 adaptation**: [tomasvr/turtlebot3_drlnav](https://github.com/tomasvr/turtlebot3_drlnav)
- **TD3 implementation**: [reiniscimurs/DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation)

**Extensions in this repository:**

- `TD3_lightweight`: compact TD3 (hidden_dim=26) compatible with POLAR reachability verification
- `TD3_VIA`: TD3 with VIA (CVaR-Constrained Policy Optimization) for safe DRL navigation
- POLAR reachable set computation via Taylor Model arithmetic and Bernstein polynomial approximation
- Trajectory collection and parallel verification pipeline

## Requirements

- ROS2 Foxy on Ubuntu 20.04
- Python 3.8.10
- PyTorch ≥ 1.10
- TensorBoard
- sympy, numpy, tqdm

## Installation

```bash
git clone <this-repo>
cd DRL-Robot-Navigation-ROS2

sudo apt install python3-rosdep2
rosdep update
rosdep install -i --from-path src --rosdistro foxy -y
sudo apt install python3-colcon-common-extensions
colcon build
```

Set up environment variables (add to `~/.bashrc` or run each session):

```bash
export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH=~/DRL-Robot-Navigation-ROS2
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
export TURTLEBOT3_MODEL=waffle
source /opt/ros/foxy/setup.bash
source install/setup.bash
```

## Training

All training scripts are run from the repository root with the Gazebo simulator running in a separate terminal.

**Terminal 1 — launch Gazebo:**

```bash
ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```

### Baseline: TD3_lightweight

Trains the compact TD3 model (hidden_dim=26) used as the baseline:

```bash
# Terminal 2
python3 src/drl_navigation_ros2/train.py

# Optional arguments
python3 src/drl_navigation_ros2/train.py --max-epochs 100 --episodes-per-epoch 70
```

Model weights are saved to:
```
src/drl_navigation_ros2/models/TD3/<run_id>/TD3_best_*.pth
```

### VIA: Safe Navigation with CVaR Constraint

Trains the VIA model (augmented state s̄ = (s, e_t), noncrossing quantile cost critic):

```bash
# Terminal 2
python3 src/drl_navigation_ros2/train_VIA.py

# Optional arguments
python3 src/drl_navigation_ros2/train_VIA.py --max-epochs 100 --episodes-per-epoch 70
```

Model weights are saved to:
```
src/drl_navigation_ros2/models/TD3_VIA/<run_id>/TD3_VIA_best_*.pth
```

**Monitor training** (either model):
```bash
tensorboard --logdir runs
```

## Post-Training: Trajectory Collection → Reachable Set Verification

Once you have a trained model, the verification workflow is two steps:

### Step 1 — Collect Trajectories

Edit the **User Configuration** block at the top of the script:

```python
# src/drl_navigation_ros2/scripts/collect_trajectories.py

model_type  = "TD3_Lightweight"           # "TD3_Lightweight" or "TD3_VIA"
model_name  = "TD3_lightweight_best"      # filename prefix of the saved weights
model_dir   = project_root / "models" / "TD3_lightweight" / "<your_run_id>"
output_name = "trajectories_td3_v1"       # output filename (saved as assets/<output_name>.pkl)
```

Then run with Gazebo active:

```bash
# Terminal 1 (if not already running)
ros2 launch turtlebot3_gazebo ros2_drl.launch.py

# Terminal 2
python3 src/drl_navigation_ros2/scripts/collect_trajectories.py
```

Output: `src/drl_navigation_ros2/assets/<output_name>.pkl`

### Step 2 — Reachable Set Verification

Edit the **User Configuration** block in the verification script:

```python
# src/drl_navigation_ros2/scripts/reachable_set_verification.py  (lines ~510-525)

# For TD3_Lightweight:
model_name      = "TD3_lightweight_best"
model_path      = project_root / "models" / "TD3_lightweight" / "<your_run_id>"
trajectory_path = project_root / "assets" / "<your_trajectory_file>_v1.pkl"

# For TD3_VIA:
model_name      = "TD3_VIA_best"
model_path      = project_root / "models" / "TD3_VIA" / "<your_run_id>"
trajectory_path = project_root / "assets" / "<your_trajectory_file>_v1.pkl"
```

Then run (no Gazebo required):

```bash
# TD3_Lightweight
python3 src/drl_navigation_ros2/scripts/reachable_set_verification.py \
    --model-type TD3_Lightweight --version v1

# TD3_VIA
python3 src/drl_navigation_ros2/scripts/reachable_set_verification.py \
    --model-type TD3_VIA --version v1

# Optionally override e_t for TD3_VIA (defaults to the value stored in the checkpoint)
python3 src/drl_navigation_ros2/scripts/reachable_set_verification.py \
    --model-type TD3_VIA --version v1 --e-t 5.0
```

The script runs verification in parallel across CPU cores. Output includes per-trajectory safety rates and an aggregate **Action Safety Rate (ASR)**.

## Project Structure

```
src/drl_navigation_ros2/
├── train.py                        # Baseline (TD3_lightweight) training entry point
├── train_VIA.py                    # VIA training entry point
├── TD3/
│   ├── TD3_lightweight.py          # Compact TD3 (hidden_dim=26, POLAR-compatible)
│   ├── TD3_VIA.py                  # TD3 + VIA safety constraint
│   └── TD3.py                      # Original full-size TD3 (reference)
├── replay_buffer.py                # Standard replay buffer
├── via_replay_buffer.py            # VIA replay buffer (8-tuple with e_t tracking)
├── ros_python.py                   # ROS2/Gazebo environment wrapper
├── pretrain_utils.py               # Pre-training from offline data
├── utils.py                        # Evaluation scenario utilities
├── scripts/
│   ├── collect_trajectories.py     # Collect rollout trajectories from a trained model
│   └── reachable_set_verification.py  # Parallel POLAR reachable set verification
├── verification/
│   ├── taylor_model.py             # Taylor Model arithmetic core
│   ├── polar_verifier.py           # POLAR layer-by-layer propagation
│   └── ray_casting.py              # Laser scan prediction via ray-box intersection
└── assets/
    ├── data.yml                    # Pre-training offline data
    ├── eval_scenarios_*.json       # Fixed evaluation scenario configurations
    └── obstacle_map.json           # Environment obstacle geometry for ray casting
```
