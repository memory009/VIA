#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path


@dataclass
class pos_data:
    name = None
    x = None
    y = None
    angle = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'x': float(self.x) if self.x is not None else None,
            'y': float(self.y) if self.y is not None else None,
            'angle': float(self.angle) if self.angle is not None else None
        }


def check_position(x, y, element_positions, min_dist):
    pos = True
    for element in element_positions:
        distance_vector = [element[0] - x, element[1] - y]
        distance = np.linalg.norm(distance_vector)
        if distance < min_dist:
            pos = False
    return pos


def set_random_position(name, element_positions):
    angle = np.random.uniform(-np.pi, np.pi)
    pos = False
    while not pos:
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        pos = check_position(x, y, element_positions, 1.8)
    element_positions.append([x, y])
    eval_element = pos_data()
    eval_element.name = name
    eval_element.x = x
    eval_element.y = y
    eval_element.angle = angle
    return eval_element


def record_eval_positions(
    n_eval_scenarios=10,
    save_to_file=True,
    random_seed=None,
    enable_random_obstacles=True,
    n_random_obstacles=16,
    save_filename="eval_scenarios.json",
):
    """
    Generate evaluation scenarios with random positions for obstacles, robot, and target.
    
    Args:
        n_eval_scenarios: Number of scenarios to generate
        save_to_file: Whether to save scenarios to a JSON file
        random_seed: Random seed for reproducibility (None for random)
        enable_random_obstacles: Whether to include additional random obstacles (starting from obstacle5)
        n_random_obstacles: How many random obstacles to create (ignored if enable_random_obstacles=False)
        save_filename: Name of the JSON file to write (stored under assets/)
    
    Returns:
        List of scenarios, each containing position data for all elements
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    scenarios = []
    for scenario_idx in range(n_eval_scenarios):
        eval_scenario = []
        element_positions = [[-2.93, 3.17], [2.86, -3.0], [-2.77, -0.96], [2.83, 2.93]]
        
        # 可选：添加指定数量的随机障碍物
        if enable_random_obstacles:
            total_random = max(0, n_random_obstacles)
            for i in range(4, 4 + total_random):
                name = "obstacle" + str(i + 1)
                eval_element = set_random_position(name, element_positions)
                eval_scenario.append(eval_element)

        eval_element = set_random_position("turtlebot3_waffle", element_positions)
        eval_scenario.append(eval_element)

        eval_element = set_random_position("target", element_positions)
        eval_scenario.append(eval_element)

        scenarios.append(eval_scenario)

    # Save scenarios to file
    if save_to_file:
        save_path = Path(__file__).parent / "assets" / save_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        scenarios_dict = {
            'n_scenarios': n_eval_scenarios,
            'random_seed': random_seed,
            'min_distance': 1.8,
            'enable_random_obstacles': enable_random_obstacles,
            'n_obstacles': 4 + n_random_obstacles if enable_random_obstacles else 4,
            'scenarios': []
        }
        
        for idx, scenario in enumerate(scenarios):
            # 机器人和目标的索引取决于是否启用随机障碍物
            robot_idx = n_random_obstacles if enable_random_obstacles else 0
            target_idx = robot_idx + 1
            
            scenario_dict = {
                'scenario_id': idx,
                'elements': [element.to_dict() for element in scenario],
                'robot_start': scenario[robot_idx].to_dict(),  # turtlebot3_waffle
                'target': scenario[target_idx].to_dict()  # target
            }
            scenarios_dict['scenarios'].append(scenario_dict)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(scenarios_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation scenarios saved to: {save_path}")
        print(f"   - Number of scenarios: {n_eval_scenarios}")
        print(f"   - Random seed: {random_seed if random_seed else 'None (random)'}")
        if enable_random_obstacles:
            print(f"   - Random obstacles: Enabled (4 fixed + {n_random_obstacles} random)")
            print(f"   - Total obstacles per scenario: {4 + n_random_obstacles}")
        else:
            print("   - Random obstacles: Disabled (4 fixed only)")
            print("   - Total obstacles per scenario: 4")
        print(f"   - Min distance between elements: 1.8m")

    return scenarios
