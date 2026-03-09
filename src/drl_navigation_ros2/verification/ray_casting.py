#!/usr/bin/env python3
"""
Ray casting: predict laser scan readings at a given robot pose.
Used by the POLAR safety checker to assess reachable next states.
"""

import numpy as np
import json
from pathlib import Path


class ObstacleMap:
    """Obstacle map with ray-box intersection for laser scan prediction."""

    def __init__(self, map_json_path=None):
        if map_json_path is None:
            map_json_path = Path(__file__).parent.parent / "assets" / "obstacle_map.json"

        with open(map_json_path, 'r') as f:
            self.map_data = json.load(f)

        self.obstacles = self._parse_obstacles()
        self.boundary = self.map_data['boundary']
        self.laser_range = self.map_data['metadata']['laser_range']
        self.n_beams = self.map_data['metadata']['laser_beams']
        self.laser_fov = np.deg2rad(self.map_data['metadata']['laser_fov'])

    def _parse_obstacles(self):
        """Convert box obstacles to axis-aligned bounding boxes (AABB)."""
        parsed = []
        for obs in self.map_data['obstacles']:
            if obs['shape'] == 'box':
                center = np.array(obs['position'][:2])
                size = np.array(obs['size'][:2])
                parsed.append({
                    'name':  obs['name'],
                    'type':  obs['type'],
                    'x_min': center[0] - size[0] / 2,
                    'x_max': center[0] + size[0] / 2,
                    'y_min': center[1] - size[1] / 2,
                    'y_max': center[1] + size[1] / 2,
                })
        return parsed

    def ray_box_intersection(self, ray_origin, ray_direction, box):
        """
        Slab method ray-AABB intersection test.

        Returns the intersection distance, or None if no intersection.
        """
        epsilon = 1e-8
        dir_x = ray_direction[0] if abs(ray_direction[0]) > epsilon else epsilon
        dir_y = ray_direction[1] if abs(ray_direction[1]) > epsilon else epsilon

        t_x_min = (box['x_min'] - ray_origin[0]) / dir_x
        t_x_max = (box['x_max'] - ray_origin[0]) / dir_x
        if t_x_min > t_x_max:
            t_x_min, t_x_max = t_x_max, t_x_min

        t_y_min = (box['y_min'] - ray_origin[1]) / dir_y
        t_y_max = (box['y_max'] - ray_origin[1]) / dir_y
        if t_y_min > t_y_max:
            t_y_min, t_y_max = t_y_max, t_y_min

        if t_x_min > t_y_max or t_y_min > t_x_max:
            return None

        t_near = max(t_x_min, t_y_min)
        t_far  = min(t_x_max, t_y_max)

        if t_far < 0:
            return None

        return max(0, t_near)

    def cast_ray(self, origin, angle):
        """Cast a single ray and return the distance to the nearest obstacle."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        min_distance = self.laser_range

        for obs in self.obstacles:
            dist = self.ray_box_intersection(origin, direction, obs)
            if dist is not None and dist < min_distance:
                min_distance = dist

        return min_distance

    def predict_laser_scan(self, x, y, yaw):
        """
        Predict a full laser scan at pose (x, y, yaw).

        Returns:
            laser_readings : numpy array of shape (n_beams,)
        """
        origin = np.array([x, y])
        angle_min = yaw - self.laser_fov / 2
        angle_max = yaw + self.laser_fov / 2
        angles = np.linspace(angle_min, angle_max, self.n_beams)
        return np.array([self.cast_ray(origin, angle) for angle in angles])


# Module-level singleton — avoids repeated JSON loading
_obstacle_map_singleton = None


def get_obstacle_map():
    global _obstacle_map_singleton
    if _obstacle_map_singleton is None:
        _obstacle_map_singleton = ObstacleMap()
    return _obstacle_map_singleton
