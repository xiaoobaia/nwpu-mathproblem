import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
import copy

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：无法设置中文字体，图表标题可能无法正确显示")

# 读取原始钻石数据
def read_rough_stone(file_path):
    # 读取节点数据（前59行）
    nodes_data = pd.read_csv(file_path, nrows=59, encoding='utf-8')
    
    # 跳过前60行（包括两个标题行），读取面片数据
    faces_data = pd.read_csv(file_path, skiprows=60, encoding='utf-8')
    
    # 提取节点坐标
    vertices = []
    for _, row in nodes_data.iterrows():
        x = float(row['x (cm)']) if 'x (cm)' in row else float(row['x(cm)'])
        y = float(row['y (cm)']) if 'y (cm)' in row else float(row['y(cm)'])
        z = float(row['z (cm)']) if 'z (cm)' in row else float(row['z(cm)'])
        vertices.append([x, y, z])
    
    # 提取三角形面
    triangles = []
    for _, face in faces_data.iterrows():
        v1 = int(face['Vertex 1']) - 1
        v2 = int(face['Vertex 2']) - 1
        v3 = int(face['Vertex 3']) - 1
        triangles.append([v1, v2, v3])
    
    return np.array(vertices), np.array(triangles)

# 读取标准圆形钻石数据
def read_standard_diamond(file_path):
    # 创建字典存储不同部分的点和面
    sections = {}
    current_section = None
    points = []
    faces = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        parts = line.split(',')
        
        # 如果是新的部分标题
        if len(parts) >= 4 and parts[0] and parts[1] == '' and parts[2] == '' and parts[3] == '':
            if current_section is not None:
                sections[current_section] = {
                    'points': np.array(points),
                    'faces': faces
                }
            
            current_section = parts[0]
            points = []
            faces = []
            i += 1
            continue
        
        # 如果是点数据
        if len(parts) >= 4 and parts[0].strip() and parts[1].strip() and parts[2].strip() and parts[3].strip():
            try:
                face_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                
                points.append([x, y, z])
                
                # 如果是新的面，添加到faces
                if not faces or faces[-1][0] != face_id:
                    faces.append([face_id, len(points) - 1])
                else:
                    faces[-1].append(len(points) - 1)
            except ValueError:
                pass
        
        i += 1
    
    # 添加最后一部分
    if current_section is not None:
        sections[current_section] = {
            'points': np.array(points),
            'faces': faces
        }
    
    # 合并所有点和建立面索引
    all_points = []
    all_faces = []
    point_offset = 0
    
    for section_name, section_data in sections.items():
        section_points = section_data['points']
        section_faces = section_data['faces']
        
        # 添加点
        all_points.extend(section_points)
        
        # 添加调整后的面索引
        for face in section_faces:
            face_id = face[0]
            vertices = face[1:]
            # 调整索引值，考虑偏移
            adjusted_vertices = [v + point_offset for v in vertices]
            if len(adjusted_vertices) >= 3:  # 确保至少有3个点
                all_faces.append(adjusted_vertices[:3])  # 只取前3个点组成三角形
        
        point_offset += len(section_points)
    
    return np.array(all_points), np.array(all_faces)

# 计算模型质心
def calculate_centroid(vertices):
    return np.mean(vertices, axis=0)

# 计算模型包围盒
def calculate_bounding_box(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    return min_coords, max_coords

# 计算模型体积
def calculate_volume(vertices, triangles):
    volume = 0
    for triangle in triangles:
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        # 计算三角形的体积贡献
        volume += np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
    return volume

# 旋转点云
def rotate_points(points, theta, phi, sigma):
    # 创建旋转矩阵
    rotation = R.from_euler('xyz', [theta, phi, sigma], degrees=True)
    # 应用旋转
    return rotation.apply(points)

# 平移点云
def translate_points(points, translation):
    return points + translation

# 缩放点云
def scale_points(points, scale_factor):
    return points * scale_factor

# 检查一个点是否在三角形内
def point_in_triangle(p, v1, v2, v3):
    def same_side(p1, p2, a, b):
        cp1 = np.cross(b - a, p1 - a)
        cp2 = np.cross(b - a, p2 - a)
        return np.dot(cp1, cp2) >= 0
    
    return (same_side(p, v1, v2, v3) and 
            same_side(p, v2, v1, v3) and 
            same_side(p, v3, v1, v2))

# 射线与三角形相交检测
def ray_triangle_intersect(origin, direction, v0, v1, v2, epsilon=1e-6):
    # 计算两条边向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 计算determinant
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)
    
    # 如果determinant接近0，射线与三角形平行
    if abs(det) < epsilon:
        return False, float('inf')
    
    inv_det = 1.0 / det
    
    # 计算第一个重心坐标
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det
    
    # 检查边界
    if u < 0 or u > 1:
        return False, float('inf')
    
    # 计算第二个重心坐标
    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det
    
    # 检查边界和u+v<=1
    if v < 0 or u + v > 1:
        return False, float('inf')
    
    # 计算t，射线与三角形的交点距离
    t = np.dot(edge2, qvec) * inv_det
    
    return True, t

# 从diamond12.csv读取钻石参数
def read_diamond_parameters(file_path):
    params = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                # 去除可能的度数符号
                value = parts[1].replace('°', '')
                try:
                    # 尝试转换为浮点数
                    params[parts[0]] = float(value)
                except ValueError:
                    # 如果不能转换，保留原始字符串
                    params[parts[0]] = parts[1]
    return params

# 定义钻石的几何参数
def set_diamond_parameters():
    # 从diamond12.csv中获取的参数
    params = {
        'Table_size': 0.53,  # 台面尺寸
        'Crown_height': 0.43,  # 冠部高度
        'Pavilion_depth': 0.16,  # 亭部深度
        'Crown_Angle': 35,    # 冠部角度（度）
        'Pavilion_Angle': 41,  # 亭部角度（度）
        'a': 0.5,            # 椭圆主半轴长度（标准圆形钻石为0.5）
        'b': 0.5,            # 椭圆次半轴长度（标准圆形钻石为0.5）
        'e': 0.0,            # 椭圆偏心率（标准圆形钻石为0）
        'D': 0.03,           # 腰部高度
        'Lp': 0.43,          # 下锥体高度
        'mp': 0.8,           # 下锥体高度比例
        'Lc': 0.16,          # 上锥体高度
        'mc': 0.8,           # 上锥体高度比例
        'd': 0.015,          # 半腰部高度
    }
    return params

# 检查点是否在模型内部（简化版本）
def is_point_inside(point, vertices, triangles):
    # 使用射线法检查点是否在模型内部
    directions = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]
    
    inside_count = 0
    for direction in directions:
        intersection_count = 0
        # 只检查一部分三角形，提高速度
        sample_triangles = np.random.choice(len(triangles), min(30, len(triangles)), replace=False)
        
        for idx in sample_triangles:
            triangle = triangles[idx]
            v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
            hit, _ = ray_triangle_intersect(point, direction, v0, v1, v2)
            if hit:
                intersection_count += 1
        
        # 如果射线与表面相交的次数为奇数，则点在内部
        if intersection_count % 2 == 1:
            inside_count += 1
    
    # 如果大多数方向射线判断为内部，则点在内部
    return inside_count > len(directions) / 2

# 优化目标函数
def objective_function(params, G_vertices, G_triangles, C_vertices, C_triangles):
    # 从参数中提取位置、旋转和缩放值
    x, y, z, theta, phi, sigma, scale = params
    
    translation = np.array([x, y, z])
    
    # 复制一份钻石模型点云进行变换
    transformed_vertices = np.copy(C_vertices)
    
    # 旋转
    transformed_vertices = rotate_points(transformed_vertices, theta, phi, sigma)
    # 缩放
    transformed_vertices = scale_points(transformed_vertices, scale)
    # 平移
    transformed_vertices = translate_points(transformed_vertices, translation)
    
    # 使用采样点检查钻石是否在原石内部
    sample_size = min(50, len(transformed_vertices))
    sample_indices = np.random.choice(len(transformed_vertices), sample_size, replace=False)
    
    inside_count = 0
    for idx in sample_indices:
        vertex = transformed_vertices[idx]
        if is_point_inside(vertex, G_vertices, G_triangles):
            inside_count += 1
    
    # 如果大部分采样点都在内部，则计算适应度，否则返回0
    if inside_count / sample_size < 0.9:
        return 0.0
    
    # 计算变换后的钻石体积
    diamond_volume = calculate_volume(transformed_vertices, C_triangles)
    # 计算原石体积
    rough_volume = calculate_volume(G_vertices, G_triangles)
    
    # 返回体积比（越大越好）
    return diamond_volume / rough_volume

# 运行优化算法
def optimize_diamond_position(G_vertices, G_triangles, C_vertices, C_triangles):
    # 计算原石质心作为初始猜测位置
    G_centroid = calculate_centroid(G_vertices)
    
    # 定义参数范围：[x, y, z, theta, phi, sigma, scale]
    bounds = [
        (G_centroid[0] - 0.3, G_centroid[0] + 0.3),  # x
        (G_centroid[1] - 0.3, G_centroid[1] + 0.3),  # y
        (G_centroid[2] - 0.3, G_centroid[2] + 0.3),  # z
        (0, 360),  # theta (degrees)
        (0, 360),  # phi (degrees)
        (0, 360),  # sigma (degrees)
        (0.3, 0.8)  # scale (initial guess)
    ]
    
    # 运行差分进化算法
    result = differential_evolution(
        lambda p: -objective_function(p, G_vertices, G_triangles, C_vertices, C_triangles),
        bounds,
        popsize=10,
        maxiter=20,
        tol=0.02,
        disp=True,
        workers=1  # 使用单线程以减少内存使用
    )
    
    return result

# 可视化结果
def visualize_result(G_vertices, G_triangles, C_vertices, C_triangles, params):
    x, y, z, theta, phi, sigma, scale = params
    translation = np.array([x, y, z])
    
    # 变换钻石模型
    transformed_vertices = np.copy(C_vertices)
    transformed_vertices = rotate_points(transformed_vertices, theta, phi, sigma)
    transformed_vertices = scale_points(transformed_vertices, scale)
    transformed_vertices = translate_points(transformed_vertices, translation)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原石（半透明）
    rough_mesh = []
    for triangle in G_triangles:
        vertices = G_vertices[triangle]
        rough_mesh.append(vertices)
    
    rough_poly = Poly3DCollection(rough_mesh, alpha=0.2, linewidths=0.2, edgecolors='k')
    rough_poly.set_facecolor('gray')
    ax.add_collection3d(rough_poly)
    
    # 绘制优化后的钻石
    diamond_mesh = []
    for triangle in C_triangles:
        vertices = transformed_vertices[triangle]
        diamond_mesh.append(vertices)
    
    diamond_poly = Poly3DCollection(diamond_mesh, alpha=0.8, linewidths=0.2, edgecolors='k')
    diamond_poly.set_facecolor('lightblue')
    ax.add_collection3d(diamond_poly)
    
    # 计算包围盒确定坐标范围
    min_coords_G, max_coords_G = calculate_bounding_box(G_vertices)
    min_coords_C, max_coords_C = calculate_bounding_box(transformed_vertices)
    
    min_coords = np.minimum(min_coords_G, min_coords_C)
    max_coords = np.maximum(max_coords_G, max_coords_C)
    
    # 设置坐标轴范围
    ax.set_xlim(min_coords[0], max_coords[0])
    ax.set_ylim(min_coords[1], max_coords[1])
    ax.set_zlim(min_coords[2], max_coords[2])
    
    # 设置标题和标签
    ax.set_title('优化后的钻石在原石中的位置')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    
    # 等比例显示
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig('optimized_diamond_position.png', dpi=300)
    plt.show()
    
    return fig, ax

# 主函数
def main():
    # 读取原石数据
    rough_vertices, rough_triangles = read_rough_stone('data/attachment_1_diamond_original_geometry_data_file.csv')
    
    # 读取标准圆形钻石数据
    diamond_vertices, diamond_triangles = read_standard_diamond('data/attachment_2_standarded_round_diamond_geometry_data_file.csv')
    
    # 读取钻石参数
    diamond_params = read_diamond_parameters('data/diamond12.csv')
    
    print("原石信息:")
    print(f"顶点数: {len(rough_vertices)}")
    print(f"三角面数: {len(rough_triangles)}")
    
    print("\n钻石信息:")
    print(f"顶点数: {len(diamond_vertices)}")
    print(f"三角面数: {len(diamond_triangles)}")
    
    # 计算原石体积
    rough_volume = calculate_volume(rough_vertices, rough_triangles)
    print(f"\n原石体积: {rough_volume:.6f} 立方厘米")
    
    # 计算钻石模型的初始体积
    diamond_volume = calculate_volume(diamond_vertices, diamond_triangles)
    print(f"钻石模型初始体积: {diamond_volume:.6f} 立方厘米")
    
    # 运行优化算法
    print("\n开始优化钻石位置...")
    result = optimize_diamond_position(rough_vertices, rough_triangles, diamond_vertices, diamond_triangles)
    
    print("\n优化结果:")
    print(f"成功: {result.success}")
    print(f"状态: {result.message}")
    
    # 提取最优参数
    best_params = result.x
    x, y, z, theta, phi, sigma, scale = best_params
    
    print(f"\n最优位置: ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"最优旋转角度: ({theta:.2f}°, {phi:.2f}°, {sigma:.2f}°)")
    print(f"最优缩放比例: {scale:.4f}")
    
    # 计算变换后的钻石体积和体积比
    transformed_vertices = np.copy(diamond_vertices)
    transformed_vertices = rotate_points(transformed_vertices, theta, phi, sigma)
    transformed_vertices = scale_points(transformed_vertices, scale)
    transformed_vertices = translate_points(transformed_vertices, np.array([x, y, z]))
    
    optimized_diamond_volume = calculate_volume(transformed_vertices, diamond_triangles)
    volume_ratio = optimized_diamond_volume / rough_volume
    
    print(f"\n优化后的钻石体积: {optimized_diamond_volume:.6f} 立方厘米")
    print(f"优化后的体积比(钻石/原石): {volume_ratio:.4f} ({volume_ratio*100:.2f}%)")
    
    # 计算钻石克拉数 (1克拉 = 0.2克)
    density = 3.52  # 克/立方厘米
    carats = optimized_diamond_volume * density / 5
    print(f"优化后的钻石重量: {carats:.2f} 克拉")
    
    # 可视化结果
    print("\n正在生成可视化结果...")
    visualize_result(rough_vertices, rough_triangles, diamond_vertices, diamond_triangles, best_params)
    
    print("\n计算完成！结果已保存为'optimized_diamond_position.png'")

if __name__ == "__main__":
    main() 