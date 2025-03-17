#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/07 14:27
@Author  : weiyutao
@File    : coordinate_transform.py
"""

import numpy as np
import cv2


class CoordinateTramsform:
    def __init__(self, original_image=None, input_size=(640, 640), polygon_points: list = None):
        self.original_image = original_image
        self.input_size = input_size
        self.polygon_points = polygon_points

    def calculate_overlap_ratio(self, point1, point2, polygon_points: list = None, overlap_ratio_threshold: float = 0.1):
        """
        计算矩形框与多边形的重叠比例

        Args:
        - point1: (x1, y1) 矩形框左上角坐标
        - point2: (x2, y2) 矩形框右下角坐标
        - polygon_points: [(x1,y1), (x2,y2), ...] 多边形顶点列表
        
        Returns:
        - bool: 是否重叠面积占比超过20%
        """
        polygon_points = self.polygon_points if polygon_points is None else polygon_points
        # 如果polygon_points为None，直接返回False
        if polygon_points is None:
            return False

        #  如果任一点为None，直接返回False
        if point1 is None or point2 is None:
            return False

        # 确保polygon_points非空
        if len(polygon_points) < 3:
            return False
        
        # 计算矩形框的四个顶点
        x1, y1 = point1
        x2, y2 = point2

        # 注意顺时针，要和polygon匹配，因为它是顺时针
        rect_polygon = [
            (x1, y1),  # 左上角
            (x2, y1),  # 右上角
            (x2, y2),  # 右下角
            (x1, y2)   # 左下角
        ]

        # 计算矩形框面积
        rect_area = abs((x2 - x1) * (y2 - y1))

        # 计算矩形框与多边形的交集面积
        intersection_polygon = self.calculate_intersection_polygon(rect_polygon, polygon_points)

        # 计算交集面积
        if intersection_polygon:
            intersection_area = self.calculate_polygon_area(intersection_polygon)
        else:
            intersection_area = 0
        
        # 计算重叠比例
        overlap_ratio = intersection_area / rect_area
        # 返回是否超过阈值
        return overlap_ratio >= overlap_ratio_threshold


    def calculate_polygon_area(self, polygon_points):
        """
        计算多边形面积（使用shoelace公式）
        """
        n = len(polygon_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon_points[i][0] * polygon_points[j][1]
            area -= polygon_points[j][0] * polygon_points[i][1]
        area = abs(area) / 2.0
        return area
    
    def calculate_intersection_polygon(self, polygon1, polygon2):
        """
        计算两个多边形的交集
        使用Sutherland-Hodgman裁剪算法
        """
        def inside(point, cp1, cp2):
            return (cp2[0] - cp1[0]) * (point[1] - cp1[1]) > (cp2[1] - cp1[1]) * (point[0] - cp1[0])

        def compute_intersection(cp1, cp2, s, e):
            dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
            dp = [s[0] - e[0], s[1] - e[1]]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        outputList = polygon1
        cp1 = polygon2[-1]

        for clipVertex in polygon2:
            cp2 = clipVertex
            inputList = outputList
            outputList = []
            s = inputList[-1]

            for subjectVertex in inputList:
                e = subjectVertex
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        outputList.append(compute_intersection(cp1, cp2, s, e))
                    outputList.append(e)
                elif inside(s, cp1, cp2):
                    outputList.append(compute_intersection(cp1, cp2, s, e))
                s = e
            cp1 = cp2
            if len(outputList) == 0:
                return None

        return outputList


    def is_point_in_polygon(self, point, polygon_points: list = None):
        """
        判断点是否在多边形内部
        
        Args:
        - point: (x, y) 目标框中心点坐标
        - polygon: [(x1,y1), (x2,y2), ...] 多边形顶点列表
        
        Returns:
        - bool: 是否在多边形内
        """
        polygon_points = self.polygon_points if polygon_points is None else polygon_points
        if point is None:
            return False

        x, y = point
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def _calculate_transform_params(
            self, 
            original_image: str = None, 
            input_size: tuple = None,
            polygon_points: list = None
        ):
        original_image = self.original_image if original_image is None else original_image
        input_size = self.input_size if input_size is None else self.input_size
        polygon_points = self.polygon_points if polygon_points is None else self.polygon_points
        original_image_size = ()
        if isinstance(original_image, str):
            image = cv2.imread(original_image)
            original_image_size = (image.shape[1], image.shape[0])
        
        if isinstance(original_image, np.ndarray):
            original_image_size = (original_image.shape[1], original_image.shape[0])


        orig_h, orig_w = original_image_size
        input_w, input_h = self.input_size

        # 计算缩放比例
        # 计算缩放比例（保持宽高比）
        scale = min(input_w / orig_w, input_h / orig_h)

        # 计算缩放后尺寸
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 计算居中填充
        dw = (input_w - new_w) / 2
        dh = (input_h - new_h) / 2

        # 转换坐标点并严格限制在[0,1]范围
        normalized_points = []
        for x, y in polygon_points:
            # 1. 缩放
            rx = x * scale
            ry = y * scale

            # 2. 添加填充
            rx += dw
            ry += dh

            # 3. 严格归一化，使用max和min确保在[0,1]范围
            nx = max(0, min(rx / input_w, 1))
            ny = max(0, min(ry / input_h, 1))

            normalized_points.append((nx, ny))

        return normalized_points

