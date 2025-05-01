import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_standard_diamond():
    """加载标准圆形钻石数据"""
    # 读取文件内容
    with open('data/attachment_2_standarded_round_diamond_geometry_data_file.csv', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 存储不同部分的点
    sections = {
        'tabletop': [],
        'star_facets': [],
        'crown_main_facets': [],
        'upper_girdle_facets': [],
        'lower_side_facets': [],
        'pavilion_main_facets': []
    }
    
    current_section = None
    current_points = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是新的部分
        if line.startswith('1 tabletop') or line.startswith('8 star facets') or \
           line.startswith('8 crown main facets') or line.startswith('16 upper girdle facets') or \
           line.startswith('16 lower side facets') or line.startswith('8 pavilion main facets'):
            if current_section and current_points:
                sections[current_section] = current_points
            current_section = line.split('(')[0].strip().replace(' ', '_').lower()
            current_points = []
            continue
            
        # 处理数据行
        parts = line.split(',')
        if len(parts) >= 4 and parts[1] and parts[2] and parts[3]:
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                current_points.append([x, y, z])
            except ValueError:
                continue
    
    # 添加最后一个部分
    if current_section and current_points:
        sections[current_section] = current_points
    
    return sections

def create_faces_for_section(points):
    """为每个部分创建面片"""
    faces = []
    for i in range(0, len(points), 3):
        if i + 2 < len(points):
            faces.append([i+1, i+2, i+3])  # 1-based索引
    return faces

def visualize_diamond(sections):
    """可视化钻石模型"""
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义每个部分的颜色
    colors = {
        'tabletop': 'lightblue',
        'star_facets': 'lightgreen',
        'crown_main_facets': 'lightyellow',
        'upper_girdle_facets': 'lightpink',
        'lower_side_facets': 'lightcoral',
        'pavilion_main_facets': 'lightskyblue'
    }
    
    # 绘制每个部分
    for section_name, points in sections.items():
        if not points:
            continue
            
        points = np.array(points)
        faces = create_faces_for_section(points)
        
        # 准备三角形面片
        triangles = []
        for face in faces:
            triangle = [points[face[0]-1], points[face[1]-1], points[face[2]-1]]
            triangles.append(triangle)
        
        # 创建3D多边形集合
        poly3d = Poly3DCollection(triangles, alpha=0.7)
        poly3d.set_facecolor(colors.get(section_name, 'gray'))
        
        # 添加到图形中
        ax.add_collection3d(poly3d)
    
    # 设置坐标轴范围
    all_points = np.vstack([np.array(points) for points in sections.values() if points])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2.0
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('standared round diamond model')
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    # 显示图形
    plt.show()

def main():
    # 加载标准钻石数据
    sections = load_standard_diamond()
    
    # 打印每个部分的点数
    for section_name, points in sections.items():
        print(f"{section_name}: {len(points)} 个点")
    
    # 可视化钻石
    visualize_diamond(sections)

if __name__ == "__main__":
    main() 