import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 读取节点数据（前59行）
nodes_data = pd.read_csv('data/attachment_1_diamond_original_geometry_data_file.csv', nrows=59)

# 跳过前60行（包括两个标题行），读取面片数据
faces_data = pd.read_csv('data/attachment_1_diamond_original_geometry_data_file.csv', skiprows=60)

# 创建3D图形
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制每个三角形面片
for _, face in faces_data.iterrows():
    # 获取顶点索引（减去1因为Python是0-based索引）
    v1 = int(face['Vertex 1']) - 1
    v2 = int(face['Vertex 2']) - 1
    v3 = int(face['Vertex 3']) - 1
    
    # 获取顶点坐标
    vertices = [nodes_data.iloc[v1], nodes_data.iloc[v2], nodes_data.iloc[v3]]
    
    x = [float(v['x (cm)']) for v in vertices]
    y = [float(v['y (cm)']) for v in vertices]
    z = [float(v['z (cm)']) for v in vertices]
    
    # 绘制三角形面片
    tri = ax.plot_trisurf(x, y, z, color='lightblue', alpha=0.6)

# 设置坐标轴标签
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')

# 设置标题
plt.title('original diamond stone 3D model')

# 调整视角
ax.view_init(elev=20, azim=45)

# 自动调整坐标轴比例
ax.set_box_aspect([1,1,1])

# 显示图形
plt.show() 