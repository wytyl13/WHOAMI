import matplotlib.pyplot as plt

# 定义多边形
# 逆时针旋转，注意顶点顺序
rect_polygon = [
    (149.39955139160156, 464.7524719238281),
    (149.39955139160156, 909.7803344726562)
    (742.2192993164062, 909.7803344726562),
    (742.2192993164062, 464.7524719238281),
]

polygon_points = [
    (195.47826086956536, 338.5797101449276),
    (188.23188405797117, 789.304347826087),
    (254.89855072463783, 1222.6376811594203),
    (304.55896561523167, 1295.0),
    (846.2028985507247, 1290.7536231884058),
    (483.88405797101456, 225.53623188405803)
]

# 绘制多边形
plt.figure(figsize=(10, 10))
plt.plot([p[0] for p in rect_polygon] + [rect_polygon[0][0]], [p[1] for p in rect_polygon] + [rect_polygon[0][1]], 'b-')
plt.plot([p[0] for p in polygon_points] + [polygon_points[0][0]], [p[1] for p in polygon_points] + [polygon_points[0][1]], 'r-')
plt.axis('equal')  # 保持坐标轴比例
plt.savefig('tets.png', dpi=300)