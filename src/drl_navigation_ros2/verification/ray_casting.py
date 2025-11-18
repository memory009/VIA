#!/usr/bin/env python3
"""
光线投射：预测给定位姿下的激光雷达读数
"""

import numpy as np
import json
from pathlib import Path


class ObstacleMap:
    """环境障碍物地图"""
    
    def __init__(self, map_json_path=None):
        """
        Args:
            map_json_path: 地图JSON文件路径
        """
        if map_json_path is None:
            map_json_path = Path(__file__).parent.parent / "assets" / "obstacle_map.json"
        
        with open(map_json_path, 'r') as f:
            self.map_data = json.load(f)
        
        self.obstacles = self._parse_obstacles()
        self.boundary = self.map_data['boundary']
        
        # 激光雷达参数
        self.laser_range = self.map_data['metadata']['laser_range']
        self.n_beams = self.map_data['metadata']['laser_beams']
        self.laser_fov = np.deg2rad(self.map_data['metadata']['laser_fov'])
    
    def _parse_obstacles(self):
        """解析障碍物为便于碰撞检测的格式"""
        parsed = []
        
        for obs in self.map_data['obstacles']:
            if obs['shape'] == 'box':
                # 将 box 转换为 AABB (Axis-Aligned Bounding Box)
                center = np.array(obs['position'][:2])  # 只取 x, y
                size = np.array(obs['size'][:2])
                
                aabb = {
                    'name': obs['name'],
                    'type': obs['type'],
                    'x_min': center[0] - size[0]/2,
                    'x_max': center[0] + size[0]/2,
                    'y_min': center[1] - size[1]/2,
                    'y_max': center[1] + size[1]/2,
                }
                parsed.append(aabb)
        
        return parsed
    
    def ray_box_intersection(self, ray_origin, ray_direction, box):
        """
        光线与 AABB 相交测试
        
        Args:
            ray_origin: [x, y]
            ray_direction: [dx, dy] (单位向量)
            box: {'x_min', 'x_max', 'y_min', 'y_max'}
        
        Returns:
            distance: 相交距离，None 表示不相交
        """
        # 避免除零
        epsilon = 1e-8
        dir_x = ray_direction[0] if abs(ray_direction[0]) > epsilon else epsilon
        dir_y = ray_direction[1] if abs(ray_direction[1]) > epsilon else epsilon
        
        # 计算与 x 轴平行面的交点
        t_x_min = (box['x_min'] - ray_origin[0]) / dir_x
        t_x_max = (box['x_max'] - ray_origin[0]) / dir_x
        
        if t_x_min > t_x_max:
            t_x_min, t_x_max = t_x_max, t_x_min
        
        # 计算与 y 轴平行面的交点
        t_y_min = (box['y_min'] - ray_origin[1]) / dir_y
        t_y_max = (box['y_max'] - ray_origin[1]) / dir_y
        
        if t_y_min > t_y_max:
            t_y_min, t_y_max = t_y_max, t_y_min
        
        # 检查是否相交
        if t_x_min > t_y_max or t_y_min > t_x_max:
            return None
        
        t_near = max(t_x_min, t_y_min)
        t_far = min(t_x_max, t_y_max)
        
        # 检查是否在射线方向上
        if t_far < 0:
            return None
        
        # 返回第一个交点距离
        return max(0, t_near)
    
    def cast_ray(self, origin, angle):
        """
        投射单条光线
        
        Args:
            origin: [x, y] 光线起点
            angle: 光线角度（弧度）
        
        Returns:
            distance: 到最近障碍物的距离
        """
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        min_distance = self.laser_range
        
        # 检查所有障碍物
        for obs in self.obstacles:
            dist = self.ray_box_intersection(origin, direction, obs)
            if dist is not None and dist < min_distance:
                min_distance = dist
        
        return min_distance
    
    def predict_laser_scan(self, x, y, yaw):
        """
        预测给定位姿下的激光雷达扫描
        
        Args:
            x, y: 机器人位置
            yaw: 机器人朝向（弧度）
        
        Returns:
            laser_readings: array of shape (n_beams,)
        """
        origin = np.array([x, y])
        
        # 激光雷达角度范围：相对于机器人朝向的 [-FOV/2, FOV/2]
        angle_min = yaw - self.laser_fov / 2
        angle_max = yaw + self.laser_fov / 2
        angles = np.linspace(angle_min, angle_max, self.n_beams)
        
        laser_readings = np.array([self.cast_ray(origin, angle) for angle in angles])
        
        return laser_readings


# 全局单例（避免重复加载）
_obstacle_map_singleton = None

def get_obstacle_map():
    """获取障碍物地图单例"""
    global _obstacle_map_singleton
    if _obstacle_map_singleton is None:
        _obstacle_map_singleton = ObstacleMap()
    return _obstacle_map_singleton