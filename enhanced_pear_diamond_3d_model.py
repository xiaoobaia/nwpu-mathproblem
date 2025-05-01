import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from matplotlib.colors import LightSource

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

# 计算面法线
def compute_normals(vertices, triangles):
    normals = []
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        # 计算两条边向量
        edge1 = v1 - v0
        edge2 = v2 - v0
        # 叉乘计算法线
        normal = np.cross(edge1, edge2)
        # 归一化
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        normals.append(normal)
    return np.array(normals)

# 创建增强版3D模型
def create_enhanced_3d_model(nodes, triangles):
    # 创建图形
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算法线用于光照效果
    normals = compute_normals(nodes, triangles)
    
    # 创建光源
    ls = LightSource(azdeg=315, altdeg=45)
    
    # 绘制三角形面
    mesh = []
    for triangle in triangles:
        vertices = nodes[triangle]
        mesh.append(vertices)
    
    # 创建多个多边形集合，使用不同的透明度和颜色，模拟钻石不同切面的效果
    # 主体部分 - 淡蓝色
    poly3d_1 = Poly3DCollection(mesh, alpha=0.8, linewidths=0.1, edgecolor='white')
    main_color = np.array([0.7, 0.9, 1.0, 0.8])  # 淡蓝色
    poly3d_1.set_facecolor(main_color)
    
    # 应用光照效果
    illuminated_surface = ls.shade_normals(normals, fraction=0.5)
    poly3d_1.set_array(illuminated_surface)
    
    # 添加到图形中
    ax.add_collection3d(poly3d_1)
    
    # 设置坐标轴范围
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    z_min, z_max = nodes[:, 2].min(), nodes[:, 2].max()
    
    # 计算中心点，用于旋转
    center = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])
    
    # 略微扩大范围以便更好地显示
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    
    # 设置坐标轴范围 - 确保各方向比例相同
    ax.set_xlim(center[0] - max_range/2 * 1.1, center[0] + max_range/2 * 1.1)
    ax.set_ylim(center[1] - max_range/2 * 1.1, center[1] + max_range/2 * 1.1)
    ax.set_zlim(center[2] - max_range/2 * 1.1, center[2] + max_range/2 * 1.1)
    
    # 设置标题和标签
    ax.set_title('高级梨形钻石3D模型渲染', fontsize=18, pad=20)
    ax.set_xlabel('X (cm)', fontsize=14, labelpad=10)
    ax.set_ylabel('Y (cm)', fontsize=14, labelpad=10)
    ax.set_zlabel('Z (cm)', fontsize=14, labelpad=10)
    
    # 设置更好的视角
    ax.view_init(elev=30, azim=45)
    
    # 等比例显示，确保不变形
    ax.set_box_aspect([1, 1, 1])
    
    # 设置背景为渐变色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('aliceblue')
    
    # 去掉网格线和坐标轴
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 只显示主要刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    return fig, ax

# 创建动态旋转视图的函数
def create_rotating_view(nodes, triangles, n_frames=36, save_gif=True):
    from matplotlib.animation import FuncAnimation
    import io
    from PIL import Image
    
    # 创建初始图形
    fig, ax = create_enhanced_3d_model(nodes, triangles)
    
    # 动画函数
    def update(frame):
        ax.view_init(elev=20, azim=frame * 10)
        return [ax]
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=n_frames, blit=True)
    if save_gif:
        print("正在保存GIF动画，这可能需要一点时间...")
        # 保存为GIF
        ani.save('figure/pear_diamond_rotating.gif', writer='pillow', fps=10, dpi=100)
        print("已保存旋转动画到 pear_diamond_rotating.gif")
    
    return ani

# 主函数
def main():
    # CSV文件路径
    file_path = 'data/attachment_3_standarded_pear_diamond_geometry_data_file.csv'
    print("正在读取数据文件...")
    
    # 读取数据
    nodes, triangles = read_diamond_data(file_path)
    print(f"已读取 {len(nodes)} 个节点和 {len(triangles)} 个三角面")
    
    # 创建增强版3D模型
    print("正在生成高级梨形钻石3D模型...")
    fig, ax = create_enhanced_3d_model(nodes, triangles)
    
    # 保存静态图像
    plt.tight_layout()
    plt.savefig('figure/enhanced_pear_diamond_3d_model.png', dpi=300, bbox_inches='tight')
    print("已保存增强版梨形钻石3D模型图到 enhanced_pear_diamond_3d_model.png")
    
    # 创建旋转视图
    print("是否创建旋转动画? (y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        ani = create_rotating_view(nodes, triangles)
        plt.show()
    else:
        # 添加交互功能
        ax.mouse_init()
        print("3D模型已生成完毕，您可以使用鼠标旋转查看不同角度")
        plt.show()

if __name__ == "__main__":
    main() 