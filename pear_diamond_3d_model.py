import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import matplotlib as mpl

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：无法设置中文字体，图表标题可能无法正确显示")

# 读取CSV文件
def read_diamond_data(file_path):
    # 读取节点数据（前52行）
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 解析节点数据
    nodes_data = []
    for i in range(1, 53):  # 跳过标题行，读取52个节点
        parts = lines[i].strip().split(',')
        node_num = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        nodes_data.append([x, y, z])
    nodes = np.array(nodes_data)
    
    # 解析三角形数据
    triangles_data = []
    for i in range(54, 154):  # 跳过标题行，读取100个三角形
        parts = lines[i].strip().split(',')
        v1 = int(parts[1]) - 1  # 减1是因为Python索引从0开始，而文件中从1开始
        v2 = int(parts[2]) - 1
        v3 = int(parts[3]) - 1
        triangles_data.append([v1, v2, v3])
    triangles = np.array(triangles_data)
    
    return nodes, triangles

# 创建3D模型
def create_3d_model(nodes, triangles):
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三角形面
    mesh = []
    for triangle in triangles:
        vertices = nodes[triangle]
        mesh.append(vertices)
    
    # 创建3D多边形集合，增加光泽感
    poly3d = Poly3DCollection(mesh, alpha=0.9, linewidths=0.1, edgecolors='lightgray')
    
    # 设置面颜色为透明的浅蓝色，模拟钻石效果
    face_color = (0.7, 0.9, 1, 0.7)  # 浅蓝色，半透明
    poly3d.set_facecolor(face_color)
    
    # 添加到图形中
    ax.add_collection3d(poly3d)
    
    # 设置坐标轴范围
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    z_min, z_max = nodes[:, 2].min(), nodes[:, 2].max()
    
    # 略微扩大范围以便更好地显示
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # 设置坐标轴范围
    ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    ax.set_zlim(z_min - 0.1*z_range, z_max + 0.1*z_range)
    
    # 设置标题和标签
    ax.set_title('Pear Diamond 3D Model (梨形钻石3D模型)', fontsize=16)
    ax.set_xlabel('X (cm)', fontsize=12)
    ax.set_ylabel('Y (cm)', fontsize=12)
    ax.set_zlabel('Z (cm)', fontsize=12)
    
    # 设置更好的视角
    ax.view_init(elev=25, azim=60)
    
    # 等比例显示，确保不变形
    ax.set_box_aspect([1, 1, 1])
    
    # 设置白色背景
    ax.set_facecolor('white')
    # 去掉网格线使图像更清晰
    ax.grid(False)
    
    return fig, ax

# 主函数
def main():
    # CSV文件路径
    file_path = 'data/attachment_3_standarded_pear_diamond_geometry_data_file.csv'
    print("正在读取数据文件...")
    
    # 读取数据
    nodes, triangles = read_diamond_data(file_path)
    print(f"已读取 {len(nodes)} 个节点和 {len(triangles)} 个三角面")
    
    # 创建3D模型
    fig, ax = create_3d_model(nodes, triangles)
    
    # 显示模型
    print("正在生成梨形钻石3D模型...")
    plt.tight_layout()
    plt.savefig('figure/pear_diamond_3d_model.png', dpi=300, bbox_inches='tight')
    print("已保存梨形钻石3D模型图到 pear_diamond_3d_model.png")
    
    # 添加交互功能
    ax.mouse_init()
    print("3D模型已生成完毕，您可以使用鼠标旋转查看不同角度")
    plt.show()

if __name__ == "__main__":
    main() 