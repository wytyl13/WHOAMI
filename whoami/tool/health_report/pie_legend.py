
import matplotlib
import matplotlib.pyplot as plt

# 创建自定义的图例艺术家类
class PieLegendHandler(matplotlib.legend_handler.HandlerBase):
    def __init__(self, percentage):
        self.percentage = percentage
        super().__init__()
        
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # 计算圆的中心和半径
        radius = min(width, height) / 2
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2
        
        # 创建一个圆形作为背景
        circle = plt.Circle((center_x, center_y), radius, 
                          fill=True, 
                          facecolor='white', 
                          edgecolor='black', 
                          linewidth=0.5, 
                          transform=trans)
        
        # 创建扇形
        wedge = matplotlib.patches.Wedge(
            (center_x, center_y), 
            radius, 
            0,  # 起始角度
            360 * self.percentage / 100,  # 结束角度
            facecolor=orig_handle.get_facecolor(),
            edgecolor='black',
            linewidth=0.5,
            transform=trans
        )
        
        return [circle, wedge]