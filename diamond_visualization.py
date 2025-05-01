import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_diamond_data(file_path):
    # 读取CSV文件，指定编码为utf-8
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 初始化数据结构
    sections = {}
    current_section = None
    current_points = []
    current_faces = []
    
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        parts = line.split(',')
        
        if len(parts) >= 4 and parts[1] == '' and parts[2] == '' and parts[3] == '':
            # 这是一个新的部分标题
            if current_section is not None:
                sections[current_section] = {
                    'points': np.array(current_points),
                    'faces': current_faces
                }
            
            current_section = parts[0]
            current_points = []
            current_faces = []
        elif len(parts) >= 4 and parts[0] and parts[1] and parts[2] and parts[3]:
            # 这是坐标数据
            face_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            
            # 添加点
            current_points.append([x, y, z])
            
            # 如果是新的面，添加面索引
            if not current_faces or current_faces[-1][0] != face_id:
                current_faces.append([face_id, len(current_points) - 1])
            else:
                current_faces[-1].append(len(current_points) - 1)
        
        line_idx += 1
    
    # 添加最后一部分
    if current_section is not None:
        sections[current_section] = {
            'points': np.array(current_points),
            'faces': current_faces
        }
    
    return sections

def plot_diamond(sections):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每个部分设置不同的颜色
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    i = 0
    for section_name, section_data in sections.items():
        color = colors[i % len(colors)]
        i += 1
        
        points = section_data['points']
        faces = section_data['faces']
        
        # 绘制点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                 c=color, s=5)
        
        # 绘制多边形面
        for face in faces:
            face_id = face[0]
            vertices_idx = face[1:]
            
            if len(vertices_idx) >= 3:  # 确保至少有3个点形成一个面
                vertices = points[vertices_idx]
                
                # 创建一个3D多边形
                poly = Poly3DCollection([vertices], alpha=0.3)
                poly.set_facecolor(color)
                ax.add_collection3d(poly)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置坐标轴范围
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.1])
    
    # 添加图例
    legend_elements = []
    i = 0
    for section_name in sections.keys():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        label=section_name, markerfacecolor=colors[i % len(colors)], markersize=10))
        i += 1
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 设置标题
    plt.title('standard round diamond geometry')
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    # 读取数据
    sections = read_diamond_data('data/attachment_2_standarded_round_diamond_geometry_data_file.csv')
    
    # 绘制钻石
    plot_diamond(sections) 