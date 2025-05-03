import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import math
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from scipy.spatial import ConvexHull
import copy
import os
import matplotlib.font_manager as fm
from tqdm import tqdm
import argparse # 添加导入

# 确保输出目录存在
if not os.path.exists('multiple'):
    os.makedirs('multiple')
if not os.path.exists('multiple/results'):
    os.makedirs('multiple/results')

# 设置随机种子以保证结果可重复性
np.random.seed(43)
random.seed(42)

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义严格的检查容差
STRICT_TOLERANCE = 1e-9

def read_original_diamond():
    """读取原钻石原石数据"""
    # 读取节点数据（前59行）
    nodes_data = pd.read_csv('data/attachment_1_diamond_original_geometry_data_file.csv', nrows=59)
    
    # 跳过前60行（包括两个标题行），读取面片数据
    faces_data = pd.read_csv('data/attachment_1_diamond_original_geometry_data_file.csv', skiprows=60)
    
    vertices = []
    for i in range(len(nodes_data)):
        vertices.append([
            float(nodes_data.iloc[i]['x (cm)']),
            float(nodes_data.iloc[i]['y (cm)']),
            float(nodes_data.iloc[i]['z (cm)'])
        ])
    
    faces = []
    for i in range(len(faces_data)):
        # Python索引从0开始，所以减1
        faces.append([
            int(faces_data.iloc[i]['Vertex 1']) - 1,
            int(faces_data.iloc[i]['Vertex 2']) - 1,
            int(faces_data.iloc[i]['Vertex 3']) - 1
        ])
    
    return np.array(vertices), np.array(faces)

def read_round_diamond():
    """读取标准圆形钻石数据"""
    file_path = 'data/attachment_2_standarded_round_diamond_geometry_data_file.csv'
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
    
    # 将所有点合并为一个数组
    all_points = []
    all_faces = []
    point_offset = 0
    
    for section_name, section_data in sections.items():
        points = section_data['points']
        faces = section_data['faces']
        
        all_points.extend(points)
        
        # 调整面的索引偏移
        for face in faces:
            face_id = face[0]
            vertices_idx = [idx + point_offset for idx in face[1:]]
            all_faces.append(vertices_idx)
        
        point_offset += len(points)
    
    return np.array(all_points), all_faces

def read_pear_diamond():
    """读取梨形钻石数据"""
    # 读取节点数据
    nodes_data = pd.read_csv('data/attachment_3_standarded_pear_diamond_geometry_data_file.csv', nrows=52)
    
    # 跳过前53行（包括标题行），读取面片数据
    faces_data = pd.read_csv('data/attachment_3_standarded_pear_diamond_geometry_data_file.csv', skiprows=53)
    
    vertices = []
    for i in range(len(nodes_data)):
        vertices.append([
            float(nodes_data.iloc[i]['x(cm)']),
            float(nodes_data.iloc[i]['y(cm)']),
            float(nodes_data.iloc[i]['z(cm)'])
        ])
    
    faces = []
    for i in range(len(faces_data)):
        # Python索引从0开始，所以减1
        faces.append([
            int(faces_data.iloc[i]['Vertex 1']) - 1,
            int(faces_data.iloc[i]['Vertex 2']) - 1,
            int(faces_data.iloc[i]['Vertex 3']) - 1
        ])
    
    return np.array(vertices), np.array(faces)

def transform_points(points, translation, rotation, scale):
    """
    变换点云：先旋转，再缩放，最后平移
    
    参数:
    points: 原始点云数组，形状为(n, 3)
    translation: 平移向量，形状为(3,)
    rotation: 欧拉角（x, y, z）
    scale: 缩放因子
    
    返回:
    变换后的点云，形状为(n, 3)
    """
    # 转换为张量
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # 旋转
    rx, ry, rz = rotation
    # 确保旋转角度是张量
    rx = torch.tensor(rx, dtype=torch.float32, device=device)
    ry = torch.tensor(ry, dtype=torch.float32, device=device)
    rz = torch.tensor(rz, dtype=torch.float32, device=device)
    
    # 绕X轴旋转
    rot_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rx), -torch.sin(rx)],
        [0, torch.sin(rx), torch.cos(rx)]
    ], dtype=torch.float32, device=device)
    
    # 绕Y轴旋转
    rot_y = torch.tensor([
        [torch.cos(ry), 0, torch.sin(ry)],
        [0, 1, 0],
        [-torch.sin(ry), 0, torch.cos(ry)]
    ], dtype=torch.float32, device=device)
    
    # 绕Z轴旋转
    rot_z = torch.tensor([
        [torch.cos(rz), -torch.sin(rz), 0],
        [torch.sin(rz), torch.cos(rz), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # 综合旋转矩阵
    rot_matrix = torch.matmul(torch.matmul(rot_z, rot_y), rot_x)
    
    # 应用旋转
    rotated_points = torch.matmul(points_tensor, rot_matrix.T)
    
    # 应用缩放
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
    scaled_points = rotated_points * scale_tensor
    
    # 应用平移
    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=device)
    transformed_points = scaled_points + translation_tensor
    
    return transformed_points.cpu().numpy()

def is_inside_convex_hull(point, hull, tolerance=1e-6):
    """
    判断点是否在凸包内部
    
    参数:
    point: 待检查的点，形状为(3,)
    hull: 凸包对象
    tolerance: 容许的误差
    
    返回:
    布尔值，表示点是否在凸包内部
    """
    equations = hull.equations
    for eq in equations:
        # 法向量
        normal = eq[:-1]
        # 偏移
        offset = eq[-1]
        # 计算点到平面的距离
        distance = np.dot(normal, point) + offset
        # 使用更严格的检查，确保点在内部
        if distance > tolerance:
            return False
    return True

def is_overlapping(points1, points2, hull1=None, hull2=None):
    """
    判断两个钻石是否重叠 (优化版本)
    
    参数:
    points1: 第一个钻石的点云
    points2: 第二个钻石的点云
    hull1: 第一个钻石的凸包对象（可选）
    hull2: 第二个钻石的凸包对象（可选）
    
    返回:
    布尔值，表示是否有重叠
    """
    # 如果没有提供凸包，则计算
    if hull1 is None:
        try:
            hull1 = ConvexHull(points1)
        except Exception as e:
            print(f"警告: 计算凸包1失败，无法检查重叠: {e}")
            return False # 保守处理，假设不重叠
    if hull2 is None:
        try:
            hull2 = ConvexHull(points2)
        except Exception as e:
            print(f"警告: 计算凸包2失败，无法检查重叠: {e}")
            return False # 保守处理

    # 检查第一个钻石的凸包顶点是否在第二个钻石内部
    hull1_vertices = points1[hull1.vertices]
    for point in hull1_vertices:
        # 使用严格容差检查
        if is_inside_convex_hull(point, hull2, tolerance=STRICT_TOLERANCE):
            return True
    
    # 检查第二个钻石的凸包顶点是否在第一个钻石内部
    hull2_vertices = points2[hull2.vertices]
    for point in hull2_vertices:
        # 使用严格容差检查
        if is_inside_convex_hull(point, hull1, tolerance=STRICT_TOLERANCE):
            return True
    
    # # 移除边的相交检查，以提高速度，顶点检查通常足够
    # # Check if edges of the hulls intersect - simplified check
    # # This can be computationally expensive and vertex checks are often sufficient
    
    # 没有重叠
    return False

def line_intersects_triangle(line_start, line_end, triangle):
    """
    检查一条线段是否与三角形相交
    
    参数:
    line_start: 线段起点
    line_end: 线段终点
    triangle: 三角形的三个顶点列表
    
    返回:
    布尔值，表示是否相交
    """
    # 计算三角形的法向量
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    # 计算线段与三角形平面的交点
    line_dir = line_end - line_start
    d = np.dot(v0 - line_start, normal) / np.dot(line_dir, normal)
    
    # 如果d不在[0,1]范围内，则线段不与平面相交
    if d < 0 or d > 1:
        return False
    
    # 计算交点
    intersection = line_start + d * line_dir
    
    # 检查交点是否在三角形内部
    # 使用重心坐标方法
    # 解决方程 intersection = alpha*v0 + beta*v1 + gamma*v2, 其中 alpha + beta + gamma = 1
    
    # 建立并求解线性方程组
    A = np.column_stack((v0, v1, v2, np.ones(3)))
    b = np.append(intersection, 1)
    
    try:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        # 检查所有系数是否在[0,1]范围内
        return all(c >= -1e-10 and c <= 1 + 1e-10 for c in x[:3]) and abs(sum(x[:3]) - 1) < 1e-10
    except:
        # 如果无法求解，保守地返回False
        return False

def calculate_volume(points, faces=None):
    """
    计算多面体的体积
    
    参数:
    points: 点云，形状为(n, 3)
    faces: 面索引（如果提供）
    
    返回:
    体积
    """
    # 如果没有提供面信息，使用凸包
    if faces is None:
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
        except Exception as e:
            print(f"计算凸包时出错: {e}")
            # 如果无法计算凸包，使用更简单的方法估计体积
            # 计算点云的边界框体积并乘以一个系数
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            bbox_volume = np.prod(max_coords - min_coords)
            return bbox_volume * 0.5  # 假设实际体积约为边界框的一半
    else:
        simplices = faces
    
    # 为确保准确性，我们使用四面体体积公式计算
    # 首先找到点云中心点
    centroid = np.mean(points, axis=0)
    
    volume = 0
    for simplex in simplices:
        # 确保simplex有3个顶点
        if len(simplex) >= 3:
            # 取前三个顶点以防面不是三角形
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # 创建三个向量：从中心点到三角形的三个顶点
            # 使用这三个向量构成四面体
            vec1 = v1 - centroid
            vec2 = v2 - centroid
            vec3 = v3 - centroid
            
            # 计算四面体体积: V = (1/6) * |vec1·(vec2×vec3)|
            tetra_volume = abs(np.dot(vec1, np.cross(vec2, vec3))) / 6.0
            volume += tetra_volume
    
    # 移除校正系数，体积比是相对的
    # correction_factor = 0.85
    return volume #* correction_factor

def calculate_volume_ratio(diamond_points1, diamond_points2, original_vertices, original_faces, 
                           individual, use_second_diamond=True):
    """
    计算体积比
    
    参数:
    diamond_points1: 第一个钻石的点云
    diamond_points2: 第二个钻石的点云（可以为None）
    original_vertices: 原石的顶点
    original_faces: 原石的面
    individual: 个体参数 [tx1,ty1,tz1, rx1,ry1,rz1, s1, tx2,ty2,tz2, rx2,ry2,rz2, s2, type1, type2]
    use_second_diamond: 是否使用第二个钻石
    
    返回:
    体积比
    """
    # 提取第一个钻石的参数
    translation1 = individual[:3]
    rotation1 = individual[3:6]
    scale1 = individual[6]
    type1 = individual[14]  # 0表示圆形，1表示梨形
    
    # 变换第一个钻石
    transformed_points1 = transform_points(diamond_points1, translation1, rotation1, scale1)
    
    # 计算原石体积
    original_volume = calculate_volume(original_vertices, original_faces)
    
    # 创建原石凸包
    orig_hull = ConvexHull(original_vertices)
    
    # 严格检查第一个钻石是否在原石内部
    all_inside1 = True
    # 使用更严格的容差，确保所有点都在原石内部
    tolerance = 1e-8
    for p in transformed_points1:
        if not is_inside_convex_hull(p, orig_hull, tolerance):
            all_inside1 = False
            break
    
    if not all_inside1:
        return 0  # 第一个钻石不在原石内部，直接返回0作为惩罚
    
    # 计算第一个钻石的体积
    diamond_volume1 = calculate_volume(transformed_points1)
    
    # 如果不使用第二个钻石，直接返回第一个钻石的体积比
    if not use_second_diamond:
        # 不再使用校正系数
        return diamond_volume1 / original_volume
    
    # 提取第二个钻石的参数
    translation2 = individual[7:10]
    rotation2 = individual[10:13]
    scale2 = individual[13]
    type2 = individual[15]
    
    # 变换第二个钻石
    transformed_points2 = transform_points(diamond_points2, translation2, rotation2, scale2)
    
    # 严格检查第二个钻石是否在原石内部
    all_inside2 = True
    for p in transformed_points2:
        if not is_inside_convex_hull(p, orig_hull, tolerance):
            all_inside2 = False
            break
    
    if not all_inside2:
        return diamond_volume1 / original_volume  # 只计算第一个钻石的体积比，不考虑第二个
    
    # 检查两个钻石是否重叠
    # 计算两个钻石的凸包用于重叠检测
    hull1 = ConvexHull(transformed_points1)
    hull2 = ConvexHull(transformed_points2)
    
    # 严格检查重叠
    if is_overlapping(transformed_points1, transformed_points2, hull1, hull2):
        # 发现重叠，返回惩罚后的适应度，只考虑第一个钻石体积的80%
        return 0.8 * diamond_volume1 / original_volume
    
    # 计算第二个钻石的体积
    diamond_volume2 = calculate_volume(transformed_points2)
    
    # 计算总体积比
    total_diamond_volume = diamond_volume1 + diamond_volume2
    volume_ratio = total_diamond_volume / original_volume
    
    # 不再使用校正系数
    return volume_ratio

def initialize_population(pop_size, original_vertices, diamond1_points, diamond2_points, scale_estimate=None):
    """
    初始化种群
    
    参数:
    pop_size: 种群大小
    original_vertices: 原石的顶点
    diamond1_points: 第一个钻石的点云
    diamond2_points: 第二个钻石的点云
    scale_estimate: 缩放因子的估计上限
    
    返回:
    初始化的种群
    """
    # 计算原石的中心和边界框
    orig_center = np.mean(original_vertices, axis=0)
    orig_min = np.min(original_vertices, axis=0)
    orig_max = np.max(original_vertices, axis=0)
    orig_size = orig_max - orig_min
    orig_radius = np.linalg.norm(orig_size) / 2
    
    # 计算钻石的中心和边界框
    d1_center = np.mean(diamond1_points, axis=0)
    d1_min = np.min(diamond1_points, axis=0)
    d1_max = np.max(diamond1_points, axis=0)
    d1_size = d1_max - d1_min
    
    d2_center = np.mean(diamond2_points, axis=0)
    d2_min = np.min(diamond2_points, axis=0)
    d2_max = np.max(diamond2_points, axis=0)
    d2_size = d2_max - d2_min
    
    # 如果没有提供缩放估计，计算一个默认值
    if scale_estimate is None:
        # 更保守的缩放估计，确保钻石在初始状态下能够适应原石
        scale_estimate1 = min(orig_size / d1_size) * 0.5  # 更保守的估计
        scale_estimate2 = min(orig_size / d2_size) * 0.5
    else:
        scale_estimate1 = scale_estimate * 0.6  # 更保守的初始估计
        scale_estimate2 = scale_estimate * 0.6
    
    # 为了确保初始种群的多样性，我们使用不同的策略初始化个体
    population = []
    
    # 创建原石凸包，用于边界检查
    orig_hull = ConvexHull(original_vertices)
    
    # 生成初始个体方法：合理分布在原石内部
    def generate_valid_individual():
        # 尝试有限次数生成有效个体
        max_attempts = 40  # 增加尝试次数
        
        for _ in range(max_attempts):
            # 为第一个钻石选择位置 - 在原石全范围内更均匀分布
            # 使用均匀分布而不是高斯分布，确保钻石分布更加均匀
            # 使用更保守的范围，确保完全在原石内部
            tx1 = np.random.uniform(orig_min[0] + orig_size[0]*0.25, orig_max[0] - orig_size[0]*0.25)
            ty1 = np.random.uniform(orig_min[1] + orig_size[1]*0.25, orig_max[1] - orig_size[1]*0.25)
            tz1 = np.random.uniform(orig_min[2] + orig_size[2]*0.25, orig_max[2] - orig_size[2]*0.25)
            
            # 随机旋转
            rx1 = np.random.uniform(0, 2 * np.pi)
            ry1 = np.random.uniform(0, 2 * np.pi)
            rz1 = np.random.uniform(0, 2 * np.pi)
            
            # 随机缩放，使用更保守的范围
            # 对于圆形钻石，使用特别保守的初始缩放
            s1 = np.random.uniform(0.4, min(1.3, scale_estimate1))
            
            # 为第二个钻石分配不同区域，避免初始就靠得太近
            # 把原石分成8个象限，确保两个钻石在不同象限
            # 找到与第一个钻石不同的象限
            first_octant = [
                int(tx1 > orig_center[0]),
                int(ty1 > orig_center[1]),
                int(tz1 > orig_center[2])
            ]
            
            # 随机选择不同的象限
            second_octant = first_octant.copy()
            change_idx = np.random.randint(0, 3)  # 随机选择一个维度改变
            second_octant[change_idx] = 1 - second_octant[change_idx]  # 翻转该维度
            
            # 在选定的象限内随机生成坐标，并使用更保守的边界
            tx2_min = orig_center[0] if second_octant[0] == 1 else orig_min[0] + orig_size[0]*0.25
            tx2_max = orig_max[0] - orig_size[0]*0.25 if second_octant[0] == 1 else orig_center[0]
            ty2_min = orig_center[1] if second_octant[1] == 1 else orig_min[1] + orig_size[1]*0.25
            ty2_max = orig_max[1] - orig_size[1]*0.25 if second_octant[1] == 1 else orig_center[1]
            tz2_min = orig_center[2] if second_octant[2] == 1 else orig_min[2] + orig_size[2]*0.25
            tz2_max = orig_max[2] - orig_size[2]*0.25 if second_octant[2] == 1 else orig_center[2]
            
            tx2 = np.random.uniform(tx2_min, tx2_max)
            ty2 = np.random.uniform(ty2_min, ty2_max)
            tz2 = np.random.uniform(tz2_min, tz2_max)
            
            # 随机旋转
            rx2 = np.random.uniform(0, 2 * np.pi)
            ry2 = np.random.uniform(0, 2 * np.pi)
            rz2 = np.random.uniform(0, 2 * np.pi)
            
            # 随机缩放，同样使用保守的范围
            s2 = np.random.uniform(0.4, min(1.3, scale_estimate2))
            
            # 设置钻石类型（在遗传算法函数中会被覆盖）
            type1 = 0  # 0表示圆形，1表示梨形
            type2 = 0  # 同上
            
            # 快速检查第一个钻石是否可能在原石内
            transformed_points1 = transform_points(diamond1_points, [tx1, ty1, tz1], [rx1, ry1, rz1], s1)
            
            # 更严格的检查：检查所有点而不仅是边界点
            all_inside1 = True
            
            # 首先构建凸包
            hull1 = ConvexHull(transformed_points1)
            
            # 检查凸包顶点
            for idx in hull1.vertices:
                vertex = transformed_points1[idx]
                if not is_inside_convex_hull(vertex, orig_hull, tolerance=1e-9):
                    all_inside1 = False
                    break
                    
            # 如果凸包顶点都在内部，再检查一些内部点
            if all_inside1:
                # 选择一些随机点进行检查
                num_samples = min(30, len(transformed_points1))
                sample_indices = np.random.choice(len(transformed_points1), num_samples, replace=False)
                
                for idx in sample_indices:
                    if not is_inside_convex_hull(transformed_points1[idx], orig_hull, tolerance=1e-9):
                        all_inside1 = False
                        break
            
            if not all_inside1:
                continue  # 第一个钻石超出原石，重新生成
            
            # 快速检查第二个钻石是否可能在原石内
            transformed_points2 = transform_points(diamond2_points, [tx2, ty2, tz2], [rx2, ry2, rz2], s2)
            
            # 同样对第二个钻石进行严格检查
            all_inside2 = True
            
            # 构建凸包
            hull2 = ConvexHull(transformed_points2)
            
            # 检查凸包顶点
            for idx in hull2.vertices:
                vertex = transformed_points2[idx]
                if not is_inside_convex_hull(vertex, orig_hull, tolerance=1e-9):
                    all_inside2 = False
                    break
                    
            # 如果凸包顶点都在内部，再检查一些内部点
            if all_inside2:
                # 选择一些随机点进行检查
                num_samples = min(30, len(transformed_points2))
                sample_indices = np.random.choice(len(transformed_points2), num_samples, replace=False)
                
                for idx in sample_indices:
                    if not is_inside_convex_hull(transformed_points2[idx], orig_hull, tolerance=1e-9):
                        all_inside2 = False
                        break
            
            if not all_inside2:
                continue  # 第二个钻石超出原石，重新生成
            
            # 检查两个钻石是否重叠
            if is_overlapping(transformed_points1, transformed_points2, hull1, hull2):
                continue  # 钻石重叠，重新生成
            
            # 通过所有检查，返回有效的个体
            return [tx1, ty1, tz1, rx1, ry1, rz1, s1, 
                    tx2, ty2, tz2, rx2, ry2, rz2, s2, 
                    type1, type2]
        
        # 如果尝试多次都失败，返回一个非常保守的解
        # 这个保守解也需要确保在界内且不重叠
        conservative_attempts = 0
        while conservative_attempts < 10: # 尝试生成一个有效的保守解
            conservative_attempts += 1
            t1_x = orig_center[0] - orig_size[0]*0.15
            t2_x = orig_center[0] + orig_size[0]*0.15
            t_y = orig_center[1]
            t_z = orig_center[2]
            s = 0.4 # 非常小的初始缩放

            ind = [
                t1_x, t_y, t_z, np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), s,
                t2_x, t_y, t_z, np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), s,
                0, 0 # 默认为 round_round
            ]
            
            # 验证保守解
            tp1 = transform_points(diamond1_points, ind[0:3], ind[3:6], ind[6])
            tp2 = transform_points(diamond2_points, ind[7:10], ind[10:13], ind[13])

            try:
                h1 = ConvexHull(tp1)
                h2 = ConvexHull(tp2)
            except:
                continue # 无法计算凸包

            valid = True
            for idx in h1.vertices:
                if not is_inside_convex_hull(tp1[idx], orig_hull, tolerance=STRICT_TOLERANCE):
                    valid = False; break
            if not valid: continue
            for idx in h2.vertices:
                if not is_inside_convex_hull(tp2[idx], orig_hull, tolerance=STRICT_TOLERANCE):
                    valid = False; break
            if not valid: continue

            if is_overlapping(tp1, tp2, h1, h2):
                continue

            print("警告: 无法生成多样化的初始种群，返回保守解")
            return ind # 返回验证通过的保守解

        # 如果连保守解都生成失败
        print("错误: 无法生成任何有效的初始个体！请检查钻石和原石数据以及参数。")
        # 返回一个可能无效但符合格式的个体，让后续流程处理
        return [
            orig_center[0], orig_center[1], orig_center[2], 0,0,0, 0.1,
            orig_center[0], orig_center[1], orig_center[2], 0,0,0, 0.1,
            0, 0
        ]
    
    # 初始化种群
    print("正在生成初始种群，确保钻石在原石内部...")
    for i in range(pop_size):
        if i % 20 == 0:
            print(f"已生成 {i} / {pop_size} 个个体")
        # 生成有效个体
        individual = generate_valid_individual()
        population.append(individual)
    
    print("初始种群生成完成")
    return np.array(population)

def selection(population, fitness_values, selection_size):
    """
    选择操作 - 混合多种选择策略
    
    参数:
    population: 当前种群
    fitness_values: 适应度值
    selection_size: 选择的个体数
    
    返回:
    选择的父代
    """
    selected = []
    pop_size = len(population)
    
    # 保留部分精英个体 - 适应度最高的前10%个体
    elite_count = int(selection_size * 0.1)
    elite_indices = np.argsort(fitness_values)[-elite_count:]
    for idx in elite_indices:
        selected.append(population[idx])
    
    # 剩余部分使用多种选择策略
    remaining = selection_size - elite_count
    
    # 锦标赛选择 - 占60%
    tournament_count = int(remaining * 0.6)
    for _ in range(tournament_count):
        # 随机选择几个个体进行锦标赛
        tournament_size = 5
        tournament_idx = np.random.choice(pop_size, tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament_idx]
        
        # 选择适应度最高的个体
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx])
    
    # 轮盘赌选择 - 占30%
    roulette_count = int(remaining * 0.3)
    
    # 计算选择概率
    # 避免负适应度值
    min_fitness = min(0, np.min(fitness_values))
    adjusted_fitness = fitness_values - min_fitness + 1e-10
    selection_probs = adjusted_fitness / np.sum(adjusted_fitness)
    
    # 进行轮盘赌选择
    roulette_indices = np.random.choice(
        pop_size, 
        size=roulette_count, 
        replace=True, 
        p=selection_probs
    )
    
    for idx in roulette_indices:
        selected.append(population[idx])
    
    # 随机选择 - 占剩余部分
    random_count = remaining - tournament_count - roulette_count
    random_indices = np.random.choice(pop_size, random_count, replace=False)
    for idx in random_indices:
        selected.append(population[idx])
    
    return np.array(selected)

def crossover(parents, crossover_rate):
    """
    交叉操作
    
    参数:
    parents: 父代个体
    crossover_rate: 交叉概率
    
    返回:
    后代个体
    """
    offspring = np.copy(parents)
    num_parents = len(parents)
    
    for i in range(0, num_parents, 2):
        if i + 1 < num_parents and np.random.random() < crossover_rate:
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # 随机选择交叉点 - 分别对两个钻石的参数进行交叉
            # 第一个钻石的参数
            crossover_point1 = np.random.randint(0, 7)
            # 第二个钻石的参数
            crossover_point2 = np.random.randint(7, 14) 
            
            # 交叉第一个钻石的参数
            offspring[i, :crossover_point1] = parent1[:crossover_point1]
            offspring[i, crossover_point1:7] = parent2[crossover_point1:7]
            offspring[i + 1, :crossover_point1] = parent2[:crossover_point1]
            offspring[i + 1, crossover_point1:7] = parent1[crossover_point1:7]
            
            # 交叉第二个钻石的参数
            offspring[i, 7:crossover_point2] = parent1[7:crossover_point2]
            offspring[i, crossover_point2:14] = parent2[crossover_point2:14]
            offspring[i + 1, 7:crossover_point2] = parent2[7:crossover_point2]
            offspring[i + 1, crossover_point2:14] = parent1[crossover_point2:14]
            
            # 钻石类型也可能互换
            if np.random.random() < 0.5:
                offspring[i, 14] = parent2[14]
                offspring[i + 1, 14] = parent1[14]
            
            if np.random.random() < 0.5:
                offspring[i, 15] = parent2[15]
                offspring[i + 1, 15] = parent1[15]
    
    return offspring

# 添加一个辅助函数来检查个体有效性
def is_individual_valid(individual, original_hull, round_points, pear_points):
    """检查个体是否有效（在原石内且不重叠）"""
    try:
        # 选取相应类型的钻石点云
        diamond1_points = round_points if individual[14] == 0 else pear_points
        diamond2_points = round_points if individual[15] == 0 else pear_points

        # 变换第一个钻石
        transformed_points1 = transform_points(diamond1_points, individual[0:3], individual[3:6], individual[6])
        hull1 = ConvexHull(transformed_points1)
        for vertex_idx in hull1.vertices:
            if not is_inside_convex_hull(transformed_points1[vertex_idx], original_hull, STRICT_TOLERANCE):
                return False # 第一个钻石出界

        # 变换第二个钻石
        transformed_points2 = transform_points(diamond2_points, individual[7:10], individual[10:13], individual[13])
        hull2 = ConvexHull(transformed_points2)
        for vertex_idx in hull2.vertices:
             if not is_inside_convex_hull(transformed_points2[vertex_idx], original_hull, STRICT_TOLERANCE):
                return False # 第二个钻石出界

        # 检查重叠
        if is_overlapping(transformed_points1, transformed_points2, hull1, hull2):
            return False # 钻石重叠

        return True # 所有检查通过
    except Exception as e:
        # print(f"警告: 验证个体时出错: {e}") # 可能由于凸包计算失败等
        return False # 保守处理，认为无效

def mutation(offspring, mutation_rate, original_vertices, generation=0, max_generations=100, round_points=None, pear_points=None):
    """
    变异操作 - 每次变异后强制检查有效性
    
    参数:
    offspring: 后代个体
    mutation_rate: 变异概率
    original_vertices: 原石的顶点
    generation: 当前代数
    max_generations: 最大代数
    round_points: 圆形钻石点云
    pear_points: 梨形钻石点云
    
    返回:
    变异后的后代个体
    """
    num_offspring = len(offspring)
    
    # 计算原石的中心和边界框
    orig_center = np.mean(original_vertices, axis=0)
    orig_min = np.min(original_vertices, axis=0)
    orig_max = np.max(original_vertices, axis=0)
    orig_size = orig_max - orig_min
    
    # 创建原石凸包用于边界检查
    orig_hull = ConvexHull(original_vertices)
    
    # 随着代数增加，逐渐减小变异步长
    mutation_scale = max(0.1, 1.0 - 0.9 * (generation / max_generations)) # 更平滑地减小步长
    
    # 设置平移变异的合理范围
    max_translation_mutation = orig_size * 0.05 * mutation_scale # 减小基础步长
    
    # 旋转变异步长
    rotation_mutation_std = np.pi / 8 * mutation_scale # 减小旋转步长
    
    # 缩放变异步长
    scale_mutation_std = 0.1 * mutation_scale # 减小缩放步长

    for i in range(num_offspring):
        # 记录原始个体，用于验证变异后是否超出边界或重叠
        original_individual = offspring[i].copy()
        mutated = False # 标记是否发生了实际的变异

        # 对钻石1和钻石2分别进行变异
        for diamond_idx in range(2):  # 0表示第一个钻石，1表示第二个钻石
            base_idx = diamond_idx * 7  # 参数基础索引
            
            # 平移变异
            for j in range(3):
                if np.random.random() < mutation_rate:
                    mutation_value = np.random.normal(0, max_translation_mutation[j])
                    offspring[i, base_idx + j] += mutation_value
                    mutated = True
                    # 移除这里的边界钳制，由后续的整体检查处理
            
            # 旋转变异
            for j in range(3, 6):
                if np.random.random() < mutation_rate:
                    offspring[i, base_idx + j] += np.random.normal(0, rotation_mutation_std)
                    # 确保角度在0到2π之间
                    offspring[i, base_idx + j] = offspring[i, base_idx + j] % (2 * np.pi)
                    mutated = True
            
            # 缩放变异
            scale_idx = base_idx + 6
            if np.random.random() < mutation_rate:
                scale_change = np.random.normal(0, scale_mutation_std)
                offspring[i, scale_idx] *= (1 + scale_change)
                mutated = True

                # 移除这里的边界钳制，由后续整体检查处理
                # 设置合理的缩放范围，以防无效缩放
                min_scale = 0.1 # 允许更小的缩放
                max_scale = 3.0 # 允许更大的缩放，由边界检查控制
                offspring[i, scale_idx] = np.clip(offspring[i, scale_idx], min_scale, max_scale)

        # 如果发生了变异，则进行严格的有效性检查
        # 不再使用随机概率检查，而是每次变异后都检查
        if mutated and round_points is not None and pear_points is not None:
            if not is_individual_valid(offspring[i], orig_hull, round_points, pear_points):
                # 如果变异后的个体无效（出界或重叠），恢复到原始个体
                offspring[i] = original_individual

    # 移除位置重排逻辑，让标准的变异和选择处理
    # if generation % 5 == 0: ...

    return offspring

def calculate_fitness_batch(population, diamond1_points, diamond2_points, original_vertices, original_faces, batch_size=10):
    """
    批量计算适应度
    
    参数:
    population: 种群
    diamond1_points: 第一个钻石的点云
    diamond2_points: 第二个钻石的点云
    original_vertices: 原石的顶点
    original_faces: 原石的面
    batch_size: 批处理大小
    
    返回:
    适应度值数组
    """
    pop_size = len(population)
    fitness_values = np.zeros(pop_size)
    
    # 预先计算原石凸包，避免重复计算
    orig_hull = ConvexHull(original_vertices)
    
    # 预计算原石体积，避免重复计算
    original_volume = calculate_volume(original_vertices, original_faces)
    
    # 增加线程数以充分利用CPU
    max_workers = min(os.cpu_count() or 4, 12)  # 限制最大线程数，避免过多开销
    
    # 分批计算适应度
    results = {} # 使用字典存储结果以保持顺序
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, individual in enumerate(population):
            # 根据钻石类型选择不同的点云
            # 修正: 使用传入的 diamond1_points (圆形) 和 diamond2_points (梨形)
            d1_points_actual = diamond1_points if individual[14] == 0 else diamond2_points
            d2_points_actual = diamond1_points if individual[15] == 0 else diamond2_points
            
            # 提交任务
            future = executor.submit(
                        calculate_volume_ratio_optimized, 
                        d1_points_actual, d2_points_actual, # 传递修正后的点云
                        original_vertices, original_faces, 
                        individual, orig_hull, original_volume, True
                    )
            futures[future] = i # 记录原始索引

        # 获取结果并按原始顺序排列
        for future in tqdm(futures, desc="计算适应度", total=pop_size, disable=True): # 禁用tqdm输出
             original_index = futures[future]
             try:
                 results[original_index] = future.result()
             except Exception as e:
                 print(f"计算个体 {original_index} 适应度时出错: {e}")
                 results[original_index] = 0 # 出错则适应度为0

    # 按原始顺序填充适应度数组
    for i in range(pop_size):
        fitness_values[i] = results.get(i, 0) # 如果某个计算失败，默认为0

    return fitness_values

def calculate_volume_ratio_optimized(diamond_points1, diamond_points2, original_vertices, original_faces, 
                          individual, orig_hull, original_volume, use_second_diamond=True):
    """
    优化版本的体积比计算函数，接收预计算的凸包和体积
    
    参数:
    diamond_points1: 第一个钻石的点云
    diamond_points2: 第二个钻石的点云
    original_vertices: 原石顶点
    original_faces: 原石面
    individual: 个体参数
    orig_hull: 预计算的原石凸包
    original_volume: 预计算的原石体积
    use_second_diamond: 是否使用第二个钻石
    
    返回:
    体积比
    """
    # 提取第一个钻石的参数
    translation1 = individual[:3]
    rotation1 = individual[3:6]
    scale1 = individual[6]
    type1 = individual[14]  # 钻石类型信息
    
    # 变换第一个钻石
    transformed_points1 = transform_points(diamond_points1, translation1, rotation1, scale1)
    
    # 检查所有点是否在原石内部 - 更严格的检查
    # 首先检查凸包顶点
    hull1 = ConvexHull(transformed_points1)
    hull1_vertices = transformed_points1[hull1.vertices]
    
    # 使用更小的容差进行严格检查
    tolerance = STRICT_TOLERANCE
    
    # 检查所有凸包顶点
    all_inside1 = True
    for p in hull1_vertices:
        if not is_inside_convex_hull(p, orig_hull, tolerance):
            all_inside1 = False
            break
    
    # 如果凸包顶点都在内部，再检查一部分内部点以确保完全在内部
    if all_inside1:
        # 随机选择一些内部点进行额外检查 - 减少检查点数以提速
        num_extra_points = min(10, len(transformed_points1) // 5) # 检查更少的点
        if num_extra_points > 0:
            indices = np.random.choice(len(transformed_points1), num_extra_points, replace=False)
            extra_points = transformed_points1[indices]
            
            for p in extra_points:
                if not is_inside_convex_hull(p, orig_hull, tolerance):
                    all_inside1 = False
                    break
    
    if not all_inside1:
        return 0  # 钻石1 超出原石，惩罚
    
    # 计算第一个钻石的体积
    diamond_volume1 = calculate_volume(transformed_points1)
    
    # 如果不使用第二个钻石，直接返回
    if not use_second_diamond:
        # 不再应用校正系数
        volume_ratio1 = diamond_volume1 / original_volume
        return volume_ratio1
    
    # 提取第二个钻石的参数
    translation2 = individual[7:10]
    rotation2 = individual[10:13]
    scale2 = individual[13]
    type2 = individual[15]  # 钻石类型信息
    
    # 变换第二个钻石
    transformed_points2 = transform_points(diamond_points2, translation2, rotation2, scale2)
    
    # 对第二个钻石执行相同的严格检查
    hull2 = ConvexHull(transformed_points2)
    hull2_vertices = transformed_points2[hull2.vertices]
    
    all_inside2 = True
    for p in hull2_vertices:
        if not is_inside_convex_hull(p, orig_hull, tolerance):
            all_inside2 = False
            break
    
    # 对第二个钻石也进行额外的内部点检查
    if all_inside2:
        # 减少检查点数以提速
        num_extra_points = min(10, len(transformed_points2) // 5)
        if num_extra_points > 0:
            indices = np.random.choice(len(transformed_points2), num_extra_points, replace=False)
            extra_points = transformed_points2[indices]
            
            for p in extra_points:
                if not is_inside_convex_hull(p, orig_hull, tolerance):
                    all_inside2 = False
                    break
    
    if not all_inside2:
        # 只考虑第一个钻石，不再应用校正
        volume_ratio1 = diamond_volume1 / original_volume
        return volume_ratio1
    
    # 检查是否重叠 - 简化版本，只检查关键点
    overlap = False
    
    # 检查第一个钻石的关键点是否在第二个钻石内
    for p in hull1_vertices:
        if is_inside_convex_hull(p, hull2, tolerance):
            overlap = True
            break
    
    # 如果没有发现重叠，再检查第二个钻石的关键点
    if not overlap:
        for p in hull2_vertices:
            if is_inside_convex_hull(p, hull1, tolerance):
                overlap = True
                break
    
    if overlap:
        # 钻石重叠，返回惩罚后的适应度
        # 惩罚：返回两个钻石中较小体积的比例，鼓励至少保留一个
        diamond_volume2 = calculate_volume(transformed_points2)
        volume_ratio1 = diamond_volume1 / original_volume
        volume_ratio2 = diamond_volume2 / original_volume
        # 返回较小体积的60%作为惩罚，鼓励分开
        return 0.6 * min(volume_ratio1, volume_ratio2)
    
    # 计算第二个钻石的体积
    diamond_volume2 = calculate_volume(transformed_points2)
    
    # 计算总体积比，不再应用校正系数
    volume_ratio1 = diamond_volume1 / original_volume
    volume_ratio2 = diamond_volume2 / original_volume
    
    # 返回总体积比
    return volume_ratio1 + volume_ratio2

def genetic_algorithm_multi_diamond(original_vertices, original_faces, 
                                  round_points, pear_points,
                                  pop_size=100, generations=100, 
                                  crossover_rate=0.8, mutation_rate=0.1,
                                  batch_size=10, early_stop_generations=50,
                                  scale_estimate=None,
                                  diamond_type_combination="round_round"):
    """
    使用遗传算法寻找最优解
    
    参数:
    original_vertices: 原石的顶点
    original_faces: 原石的面
    round_points: 圆形钻石点云
    pear_points: 梨形钻石点云
    pop_size: 种群大小
    generations: 迭代代数
    crossover_rate: 交叉概率
    mutation_rate: 变异概率
    batch_size: 批处理大小
    early_stop_generations: 如果适应度在这么多代内没有提升，则提前停止
    scale_estimate: 缩放因子的估计上限
    diamond_type_combination: 钻石类型组合，可选 "round_round", "pear_pear", "round_pear"
    
    返回:
    最佳个体和其适应度值
    """
    # 根据钻石类型组合设置初始种群
    print(f"初始化种群 (大小: {pop_size}) for {diamond_type_combination}...")
    initial_population = initialize_population(pop_size, original_vertices, round_points, pear_points, scale_estimate)
    
    # 设置钻石类型
    if diamond_type_combination == "round_round":
        for i in range(len(initial_population)):
            initial_population[i, 14] = 0  # 第一个钻石为圆形
            initial_population[i, 15] = 0  # 第二个钻石为圆形
    elif diamond_type_combination == "pear_pear":
        for i in range(len(initial_population)):
            initial_population[i, 14] = 1  # 第一个钻石为梨形
            initial_population[i, 15] = 1  # 第二个钻石为梨形
    elif diamond_type_combination == "round_pear":
        for i in range(len(initial_population)):
            initial_population[i, 14] = 0  # 第一个钻石为圆形
            initial_population[i, 15] = 1  # 第二个钻石为梨形
    
    population = initial_population
    
    # 记录每代的最佳适应度
    best_fitness_history = []
    best_individual = None
    best_fitness = -1 # 初始化为负数
    generations_without_improvement = 0
    
    # 保存最近几代的最佳适应度，用于检测是否陷入局部最优
    recent_best_fitnesses = []
    plateau_detection_window = 15  # 增加检测平台期的窗口大小
    dynamic_mutation_rate = mutation_rate # 动态调整的变异率
    
    start_time = time.time()
    
    # 遗传算法主循环
    for generation in range(generations):
        gen_start_time = time.time()
        # 计算每个个体的适应度
        fitness_values = calculate_fitness_batch(
            population, round_points, pear_points, original_vertices, original_faces, batch_size
        )
        
        # 找出当前代的最佳个体
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        current_best_individual = population[current_best_idx]
        
        # 更新全局最佳个体
        improvement = False
        if current_best_fitness > best_fitness:
            # 只有在适应度显著提高时才重置计数器（避免浮点噪音）
            if best_fitness < 0 or (current_best_fitness - best_fitness) > 1e-7:
                 improvement = True
                 best_fitness = current_best_fitness
                 best_individual = copy.deepcopy(current_best_individual)
                 generations_without_improvement = 0
                 
                 # 保存当前最佳个体到文件 (仅在显著改进时保存，减少IO)
                 print(f"找到新最佳解! 类型: {diamond_type_combination}, 代数: {generation + 1}, 体积比: {best_fitness:.6f}")
                 save_best_individual(best_individual, best_fitness, diamond_type_combination)
            else:
                 # 适应度变化微小，可能只是噪声，不重置计数器
                 generations_without_improvement += 1
        else:
            generations_without_improvement += 1
        
        best_fitness_history.append(best_fitness if best_fitness > 0 else 0) # 记录非负适应度
        
        # 更新最近几代的最佳适应度记录
        recent_best_fitnesses.append(current_best_fitness)
        if len(recent_best_fitnesses) > plateau_detection_window:
            recent_best_fitnesses.pop(0)
        
        gen_elapsed_time = time.time() - gen_start_time
        
        # 输出进度
        if (generation + 1) % 5 == 0 or improvement: # 每5代或有改进时输出
            total_elapsed_time = time.time() - start_time
            print(f"类型: {diamond_type_combination}, 代: {generation + 1}/{generations}, "
                  f"最佳体积比: {best_fitness:.6f} (当前: {current_best_fitness:.6f}), "
                  f"无改进: {generations_without_improvement}, "
                  f"代时: {gen_elapsed_time:.2f}s, 总时: {total_elapsed_time:.2f}s, "
                  f"变异率: {dynamic_mutation_rate:.3f}")
        
        # 检测是否陷入平台期 - 如果最近几代的最佳适应度变化小于阈值，增加变异率
        if len(recent_best_fitnesses) == plateau_detection_window:
            fitness_std_dev = np.std(recent_best_fitnesses)
            if fitness_std_dev < 1e-5:  # 标准差很小，可能陷入平台期
                # 暂时增加变异率，促进跳出局部最优
                print(f"检测到平台期 (std dev < 1e-5)，临时增加变异率")
                dynamic_mutation_rate = min(0.5, mutation_rate * 1.5) # 适度增加变异率
            else:
                # 恢复基础变异率
                dynamic_mutation_rate = mutation_rate
        else:
            dynamic_mutation_rate = mutation_rate
        
        # 提前停止条件
        if generations_without_improvement >= early_stop_generations:
            print(f"已连续 {early_stop_generations} 代没有显著改进，提前停止优化 (类型: {diamond_type_combination})")
            break
        
        # 选择
        selection_size = pop_size # 选择与种群数量相同的父代
        selected_parents = selection(population, fitness_values, selection_size)
        
        # 移除随机个体注入逻辑，依赖变异和选择
        
        # 交叉
        offspring = crossover(selected_parents, crossover_rate)
        
        # 变异 - 使用动态调整后的变异率
        offspring = mutation(offspring, dynamic_mutation_rate, original_vertices, generation, generations, round_points, pear_points)
        
        # 精英保留策略：确保最佳个体进入下一代
        # 找到下一代中最差的个体替换为上一代的最佳个体
        if best_individual is not None:
            next_gen_fitness = calculate_fitness_batch(
                 offspring, round_points, pear_points, original_vertices, original_faces, batch_size
             )
            worst_idx = np.argmin(next_gen_fitness)
            offspring[worst_idx] = best_individual # 用最佳个体替换最差个体
        
        # 不再进行洗牌，精英保留已经引入了最佳个体
        # if generation % 15 == 0 and generation > 0: ...
        
        # 更新种群
        population = offspring
    
    total_time = time.time() - start_time
    print(f"优化完成！类型: {diamond_type_combination}, 总用时: {total_time:.2f}秒")
    if best_individual is not None:
        print(f"最终最佳体积比: {best_fitness:.6f}")
        # 确保最终结果被保存
        save_best_individual(best_individual, best_fitness, diamond_type_combination)
         # 计算并保存最终参数
        parameters = calculate_parameters(best_individual, round_points, pear_points, diamond_type_combination)
        original_volume = calculate_volume(original_vertices, original_faces)
        save_parameters(parameters, diamond_type_combination, original_volume)

         # 绘制最终结果图
        plot_result(original_vertices, original_faces, round_points, pear_points,
                      best_individual, diamond_type_combination, parameters)
        try:
            import plotly.graph_objects as go
            plot_result_plotly(original_vertices, original_faces, round_points, pear_points,
                                 best_individual, diamond_type_combination, parameters)
            print(f"已生成交互式HTML文件: multiple/results/multi_diamond_result_{diamond_type_combination}.html")
        except ImportError:
             print("未安装plotly库，跳过生成交互式HTML文件")

    else:
         print("未能找到有效的优化结果。")
    
    # 绘制适应度历史
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.title(f'遗传算法优化过程 - {diamond_type_combination}')
    plt.xlabel('代数')
    plt.ylabel('最佳体积比')
    plt.grid(True)
    plt.savefig(f'multiple/results/optimization_history_{diamond_type_combination}.png', dpi=300)
    
    return best_individual, best_fitness

def save_best_individual(individual, fitness, suffix=""):
    """
    保存最佳个体到文件
    
    参数:
    individual: 个体参数数组
    fitness: 适应度值
    suffix: 文件名后缀，用于保存多个解
    """
    os.makedirs('multiple/results', exist_ok=True)
    
    filename = f"multiple/results/best_individual_{suffix}.txt"
    with open(filename, 'w') as f:
        f.write(f"适应度 (体积比): {fitness}\n")
        f.write(f"第一个钻石:\n")
        f.write(f"  平移向量: {individual[:3]}\n")
        f.write(f"  旋转角度: {individual[3:6]}\n")
        f.write(f"  缩放因子: {individual[6]}\n")
        f.write(f"  类型: {'圆形' if individual[14] == 0 else '梨形'}\n")
        f.write(f"第二个钻石:\n")
        f.write(f"  平移向量: {individual[7:10]}\n")
        f.write(f"  旋转角度: {individual[10:13]}\n")
        f.write(f"  缩放因子: {individual[13]}\n")
        f.write(f"  类型: {'圆形' if individual[15] == 0 else '梨形'}\n")
        f.write(f"原始数组: {individual}\n")

def calculate_parameters(individual, round_points, pear_points, diamond_type_combination):
    """
    计算钻石的参数
    
    参数:
    individual: 最佳个体
    diamond1_points: 第一个钻石的点云
    diamond2_points: 第二个钻石的点云
    diamond_type_combination: 钻石类型组合
    
    返回:
    包含钻石参数的字典列表
    """
    params = []
    
    # 获取第一个钻石的类型
    type1 = 'round' if individual[14] == 0 else 'pear'
    
    # 根据第一个钻石的类型选择点云
    points1 = round_points if type1 == 'round' else pear_points
    
    # 提取第一个钻石的参数
    translation1 = individual[:3]
    rotation1 = individual[3:6]
    scale1 = individual[6]
    
    # 变换第一个钻石的点云
    transformed_points1 = transform_points(points1, translation1, rotation1, scale1)
    
    # 计算第一个钻石的体积
    diamond_volume1 = calculate_volume(transformed_points1)
    # 计算第一个钻石的参数
    # 质心位置
    centroid1 = np.mean(transformed_points1, axis=0)
    
    # 对称轴方向 - 对于圆形钻石，对称轴是z轴旋转后的方向
    # 对于梨形钻石，对称轴是顶部到底部的方向
    if type1 == 'round':
        # Z轴旋转后的方向
        z_axis = np.array([0, 0, 1])
        rx, ry, rz = rotation1
        
        # 创建旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        R_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        rotation_matrix = np.dot(np.dot(R_z, R_y), R_x)
        symmetry_axis1 = np.dot(rotation_matrix, z_axis)
    else:
        # 对于梨形，找到最高点和最低点
        z_values = transformed_points1[:, 2]
        highest_idx = np.argmax(z_values)
        lowest_idx = np.argmin(z_values)
        highest_point = transformed_points1[highest_idx]
        lowest_point = transformed_points1[lowest_idx]
        
        # 对称轴是从最低点指向最高点的向量
        symmetry_axis1 = highest_point - lowest_point
        symmetry_axis1 = symmetry_axis1 / np.linalg.norm(symmetry_axis1)
    
    # 计算其他参数
    # 以下参数需要根据具体钻石类型来算
    if type1 == 'round':
        # 计算半轴长度
        x_coords = transformed_points1[:, 0] - centroid1[0]
        y_coords = transformed_points1[:, 1] - centroid1[1]
        z_coords = transformed_points1[:, 2] - centroid1[2]
        
        # 计算在XY平面上的距离
        xy_distances = np.sqrt(x_coords**2 + y_coords**2)
        
        # 主半轴和次半轴长度（圆形钻石它们相等）
        major_axis1 = np.max(xy_distances)
        minor_axis1 = major_axis1
        
        # 偏心率
        eccentricity1 = 0.0
        
        # 腰围高度
        girdle_points = transformed_points1[np.abs(z_coords) < 0.01]
        if len(girdle_points) > 0:
            girdle_height1 = np.max(girdle_points[:, 2]) - np.min(girdle_points[:, 2])
        else:
            girdle_height1 = 0.1 * scale1
        
        # 上锥高度
        upper_points = transformed_points1[z_coords > 0]
        upper_cone_height1 = np.max(upper_points[:, 2]) - np.min(upper_points[:, 2])
        mc1 = 0.5  # 假设值
        
        # 下锥高度
        lower_points = transformed_points1[z_coords < 0]
        lower_cone_height1 = np.max(lower_points[:, 2]) - np.min(lower_points[:, 2])
        mp1 = 0.5  # 假设值
        
        # 锥角
        bc1 = np.pi/4  # 假设值
        bp1 = np.pi/4  # 假设值
    else:
        # 梨形钻石参数略有不同
        # 这里需要根据梨形钻石的具体几何特征来计算
        # 简化处理，使用一些估计值
        
        # 计算梨形的主轴和次轴
        # 投影到XY平面
        xy_points = transformed_points1[:, :2]
        xy_center = centroid1[:2]
        distances = np.linalg.norm(xy_points - xy_center, axis=1)
        
        major_axis1 = np.max(distances)
        minor_axis1 = major_axis1 * 0.7  # 梨形通常宽度是长度的0.7左右
        
        # 偏心率
        eccentricity1 = np.sqrt(1 - (minor_axis1/major_axis1)**2)
        
        # 其他参数使用估计值
        girdle_height1 = 0.1 * scale1
        upper_cone_height1 = 0.4 * scale1
        lower_cone_height1 = 0.5 * scale1
        mc1 = 0.6
        mp1 = 0.4
        bc1 = np.pi/5
        bp1 = np.pi/3
    
    # 构建第一个钻石的参数字典
    param1 = {
        'type': type1,
        'volume': diamond_volume1,  # 添加体积信息
        'a': major_axis1,
        'b': minor_axis1,
        'e': eccentricity1,
        'D': girdle_height1,
        'Lp': lower_cone_height1,
        'mp': mp1,
        'Lc': upper_cone_height1,
        'mc': mc1,
        'bc': bc1,
        'bp': bp1,
        'centroid': centroid1,
        'symmetry_axis': symmetry_axis1,
        'scale': scale1
    }
    
    params.append(param1)
    
    # 获取第二个钻石的类型
    type2 = 'round' if individual[15] == 0 else 'pear'
    
    # 根据第二个钻石的类型选择点云
    points2 = round_points if type2 == 'round' else pear_points
    
    # 提取第二个钻石的参数
    translation2 = individual[7:10]
    rotation2 = individual[10:13]
    scale2 = individual[13]
    
    # 变换第二个钻石的点云
    transformed_points2 = transform_points(points2, translation2, rotation2, scale2)
    
    # 计算第二个钻石的体积
    diamond_volume2 = calculate_volume(transformed_points2)
    
    # 提取第二个钻石的参数
    translation2 = individual[7:10]
    rotation2 = individual[10:13]
    scale2 = individual[13]
    
    # 变换第二个钻石的点云
    transformed_points2 = transform_points(points2, translation2, rotation2, scale2)
    
    # 计算第二个钻石的参数
    # 质心位置
    centroid2 = np.mean(transformed_points2, axis=0)
    
    # 对称轴方向
    if type2 == 'round':
        z_axis = np.array([0, 0, 1])
        rx, ry, rz = rotation2
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        R_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        rotation_matrix = np.dot(np.dot(R_z, R_y), R_x)
        symmetry_axis2 = np.dot(rotation_matrix, z_axis)
    else:
        z_values = transformed_points2[:, 2]
        highest_idx = np.argmax(z_values)
        lowest_idx = np.argmin(z_values)
        highest_point = transformed_points2[highest_idx]
        lowest_point = transformed_points2[lowest_idx]
        
        symmetry_axis2 = highest_point - lowest_point
        symmetry_axis2 = symmetry_axis2 / np.linalg.norm(symmetry_axis2)
    
    # 计算其他参数
    if type2 == 'round':
        x_coords = transformed_points2[:, 0] - centroid2[0]
        y_coords = transformed_points2[:, 1] - centroid2[1]
        z_coords = transformed_points2[:, 2] - centroid2[2]
        
        xy_distances = np.sqrt(x_coords**2 + y_coords**2)
        
        major_axis2 = np.max(xy_distances)
        minor_axis2 = major_axis2
        
        eccentricity2 = 0.0
        
        girdle_points = transformed_points2[np.abs(z_coords) < 0.01]
        if len(girdle_points) > 0:
            girdle_height2 = np.max(girdle_points[:, 2]) - np.min(girdle_points[:, 2])
        else:
            girdle_height2 = 0.1 * scale2
        
        upper_points = transformed_points2[z_coords > 0]
        upper_cone_height2 = np.max(upper_points[:, 2]) - np.min(upper_points[:, 2])
        mc2 = 0.5
        
        lower_points = transformed_points2[z_coords < 0]
        lower_cone_height2 = np.max(lower_points[:, 2]) - np.min(lower_points[:, 2])
        mp2 = 0.5
        
        bc2 = np.pi/4
        bp2 = np.pi/4
    else:
        xy_points = transformed_points2[:, :2]
        xy_center = centroid2[:2]
        distances = np.linalg.norm(xy_points - xy_center, axis=1)
        
        major_axis2 = np.max(distances)
        minor_axis2 = major_axis2 * 0.7
        
        eccentricity2 = np.sqrt(1 - (minor_axis2/major_axis2)**2)
        
        girdle_height2 = 0.1 * scale2
        upper_cone_height2 = 0.4 * scale2
        lower_cone_height2 = 0.5 * scale2
        mc2 = 0.6
        mp2 = 0.4
        bc2 = np.pi/5
        bp2 = np.pi/3
    
    # 构建第二个钻石的参数字典
    param2 = {
        'type': type2,
        'volume': diamond_volume2,  # 添加体积信息
        'a': major_axis2,
        'b': minor_axis2,
        'e': eccentricity2,
        'D': girdle_height2,
        'Lp': lower_cone_height2,
        'mp': mp2,
        'Lc': upper_cone_height2,
        'mc': mc2,
        'bc': bc2,
        'bp': bp2,
        'centroid': centroid2,
        'symmetry_axis': symmetry_axis2,
        'scale': scale2
    }
    
    params.append(param2)
    
    return params

def save_parameters(parameters, diamond_type_combination, original_volume):
    """保存钻石参数到文件"""
    os.makedirs('multiple/results', exist_ok=True)
    
    filename = f"multiple/results/diamond_parameters_{diamond_type_combination}.txt"
    with open(filename, 'w', encoding='utf-8') as f:  # 添加utf-8编码
        f.write(f"原石体积: {original_volume:.6f} 立方厘米\n\n")
        
        # 计算总体积和体积比
        total_diamond_volume = 0
        for param in parameters:
            if 'volume' in param:
                total_diamond_volume += param['volume']
        
        volume_ratio = total_diamond_volume / original_volume
        f.write(f"两个钻石总体积: {total_diamond_volume:.6f} 立方厘米\n")
        f.write(f"体积比(两个钻石总体积/原石体积): {volume_ratio:.6f}\n\n")
        
        for i, param in enumerate(parameters):
            f.write(f"钻石 {i+1} ({param['type']}):\n")
            f.write(f"  体积: {param['volume']:.6f} 立方厘米\n")
            f.write(f"  a (Major semi-axis length): {param['a']:.6f}\n")
            f.write(f"  b (Minor semi-axis length): {param['b']:.6f}\n")
            f.write(f"  e (Eccentricity): {param['e']:.6f}\n")
            f.write(f"  D (Height of girdle): {param['D']:.6f}\n")
            f.write(f"  Lp (Lower cone height): {param['Lp']:.6f}\n")
            f.write(f"  mp: {param['mp']:.6f}\n")
            f.write(f"  Lc (Upper cone height): {param['Lc']:.6f}\n")
            f.write(f"  mc: {param['mc']:.6f}\n")
            f.write(f"  bc (Angle between basis semi-major axis and upper cone): {param['bc']:.6f}\n")
            f.write(f"  bp (Angle between basis semi-major axis and lower cone): {param['bp']:.6f}\n")
            f.write(f"  质心位置: {param['centroid']}\n")
            f.write(f"  对称轴方向: {param['symmetry_axis']}\n")
            f.write(f"  缩放比例: {param['scale']:.6f}\n\n")

def plot_result_plotly(original_vertices, original_faces, round_points, pear_points, 
                      best_individual, diamond_type_combination, parameters=None):
    """使用Plotly生成交互式3D可视化结果，将散点连接成封闭图形"""
    import plotly.graph_objects as go
    from scipy.spatial import Delaunay
    
    # 获取第一个钻石的类型和参数
    type1 = 'round' if best_individual[14] == 0 else 'pear'
    points1 = round_points if type1 == 'round' else pear_points
    
    # 变换第一个钻石
    translation1 = best_individual[:3]
    rotation1 = best_individual[3:6]
    scale1 = best_individual[6]
    transformed_points1 = transform_points(points1, translation1, rotation1, scale1)
    
    # 获取第二个钻石的类型和参数
    type2 = 'round' if best_individual[15] == 0 else 'pear'
    points2 = round_points if type2 == 'round' else pear_points
    
    # 变换第二个钻石
    translation2 = best_individual[7:10]
    rotation2 = best_individual[10:13]
    scale2 = best_individual[13]
    transformed_points2 = transform_points(points2, translation2, rotation2, scale2)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加原石的三角面
    i, j, k = [], [], []
    for face in original_faces:
        i.append(face[0])
        j.append(face[1])
        k.append(face[2])
    
    x, y, z = original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2]
    
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.3,
        color='lightgray',
        name='原石'
    ))
    
    # 为第一个钻石生成封闭表面
    # 计算凸包
    try:
        hull1 = ConvexHull(transformed_points1)
        # 添加第一个钻石作为封闭的3D网格
        mesh_i1, mesh_j1, mesh_k1 = [], [], []
        for simplex in hull1.simplices:
            mesh_i1.append(simplex[0])
            mesh_j1.append(simplex[1])
            mesh_k1.append(simplex[2])
        
        fig.add_trace(go.Mesh3d(
            x=transformed_points1[:, 0],
            y=transformed_points1[:, 1],
            z=transformed_points1[:, 2],
            i=mesh_i1, j=mesh_j1, k=mesh_k1,
            opacity=0.7,
            color='blue' if type1 == 'round' else 'green',
            name=f"{'圆形' if type1 == 'round' else '梨形'}钻石1"
        ))
    except Exception as e:
        # 如果生成凸包失败，回退到散点图
        print(f"无法为钻石1生成封闭表面，使用散点图替代: {e}")
        fig.add_trace(go.Scatter3d(
            x=transformed_points1[:, 0],
            y=transformed_points1[:, 1],
            z=transformed_points1[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='blue' if type1 == 'round' else 'green',
                opacity=0.7
            ),
            name=f"{'圆形' if type1 == 'round' else '梨形'}钻石1"
        ))
    
    # 为第二个钻石生成封闭表面
    try:
        hull2 = ConvexHull(transformed_points2)
        # 添加第二个钻石作为封闭的3D网格
        mesh_i2, mesh_j2, mesh_k2 = [], [], []
        for simplex in hull2.simplices:
            mesh_i2.append(simplex[0])
            mesh_j2.append(simplex[1])
            mesh_k2.append(simplex[2])
        
        fig.add_trace(go.Mesh3d(
            x=transformed_points2[:, 0],
            y=transformed_points2[:, 1],
            z=transformed_points2[:, 2],
            i=mesh_i2, j=mesh_j2, k=mesh_k2,
            opacity=0.7,
            color='red' if type2 == 'round' else 'purple',
            name=f"{'圆形' if type2 == 'round' else '梨形'}钻石2"
        ))
    except Exception as e:
        # 如果生成凸包失败，回退到散点图
        print(f"无法为钻石2生成封闭表面，使用散点图替代: {e}")
        fig.add_trace(go.Scatter3d(
            x=transformed_points2[:, 0],
            y=transformed_points2[:, 1],
            z=transformed_points2[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='red' if type2 == 'round' else 'purple',
                opacity=0.7
            ),
            name=f"{'圆形' if type2 == 'round' else '梨形'}钻石2"
        ))
    
    # 如果有参数信息，添加质心和对称轴
    if parameters is not None:
        param1 = parameters[0]
        param2 = parameters[1]
        
        # 添加第一个钻石的质心
        centroid1 = param1['centroid']
        fig.add_trace(go.Scatter3d(
            x=[centroid1[0]],
            y=[centroid1[1]],
            z=[centroid1[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='blue' if type1 == 'round' else 'green',
            ),
            name=f"{'圆形' if type1 == 'round' else '梨形'}钻石1质心"
        ))
        
        # 添加第一个钻石的对称轴
        axis1 = param1['symmetry_axis']
        fig.add_trace(go.Scatter3d(
            x=[centroid1[0], centroid1[0] + axis1[0] * 0.4],
            y=[centroid1[1], centroid1[1] + axis1[1] * 0.4],
            z=[centroid1[2], centroid1[2] + axis1[2] * 0.4],
            mode='lines',
            line=dict(
                color='blue' if type1 == 'round' else 'green',
                width=6
            ),
            name=f"{'圆形' if type1 == 'round' else '梨形'}钻石1对称轴"
        ))
        
        # 添加第二个钻石的质心
        centroid2 = param2['centroid']
        fig.add_trace(go.Scatter3d(
            x=[centroid2[0]],
            y=[centroid2[1]],
            z=[centroid2[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='red' if type2 == 'round' else 'purple',
            ),
            name=f"{'圆形' if type2 == 'round' else '梨形'}钻石2质心"
        ))
        
        # 添加第二个钻石的对称轴
        axis2 = param2['symmetry_axis']
        fig.add_trace(go.Scatter3d(
            x=[centroid2[0], centroid2[0] + axis2[0] * 0.4],
            y=[centroid2[1], centroid2[1] + axis2[1] * 0.4],
            z=[centroid2[2], centroid2[2] + axis2[2] * 0.4],
            mode='lines',
            line=dict(
                color='red' if type2 == 'round' else 'purple',
                width=6
            ),
            name=f"{'圆形' if type2 == 'round' else '梨形'}钻石2对称轴"
        ))
    
    # 设置布局
    fig.update_layout(
        title=f'双钻石切割优化结果 - {diamond_type_combination}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
        )
    )
    
    # 保存为HTML文件
    os.makedirs('multiple/results', exist_ok=True)
    fig.write_html(f'multiple/results/multi_diamond_result_{diamond_type_combination}.html')
    
    return fig

def plot_result(original_vertices, original_faces, round_points, pear_points, 
               best_individual, diamond_type_combination, parameters=None):
    """绘制结果-使用封闭表面而非散点"""
    # 创建3D图形
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原石 - 半透明
    orig_mesh = []
    for face in original_faces:
        vertices = [original_vertices[idx] for idx in face]
        orig_mesh.append(vertices)
    
    orig_poly = Poly3DCollection(orig_mesh, alpha=0.3, facecolor='lightgray', edgecolor='gray')
    ax.add_collection3d(orig_poly)
    
    # 获取第一个钻石的类型和参数
    type1 = 'round' if best_individual[14] == 0 else 'pear'
    points1 = round_points if type1 == 'round' else pear_points
    
    # 变换第一个钻石
    translation1 = best_individual[:3]
    rotation1 = best_individual[3:6]
    scale1 = best_individual[6]
    transformed_points1 = transform_points(points1, translation1, rotation1, scale1)
    
    # 绘制第一个钻石为封闭表面
    try:
        hull1 = ConvexHull(transformed_points1)
        diamond1_mesh = []
        for simplex in hull1.simplices:
            vertices = [transformed_points1[idx] for idx in simplex]
            diamond1_mesh.append(vertices)
        
        color1 = 'blue' if type1 == 'round' else 'green'
        diamond1_poly = Poly3DCollection(diamond1_mesh, alpha=0.7, facecolor=color1, edgecolor='white', linewidth=0.5)
        ax.add_collection3d(diamond1_poly)
    except Exception as e:
        # 如果无法创建凸包，回退到散点图
        print(f"无法为钻石1创建封闭表面: {e}")
        ax.scatter(transformed_points1[:, 0], transformed_points1[:, 1], transformed_points1[:, 2], 
                  c='blue' if type1 == 'round' else 'green', s=10, alpha=0.7, label=f"{'圆形' if type1 == 'round' else '梨形'}钻石1")
    
    # 获取第二个钻石的类型和参数
    type2 = 'round' if best_individual[15] == 0 else 'pear'
    points2 = round_points if type2 == 'round' else pear_points
    
    # 变换第二个钻石
    translation2 = best_individual[7:10]
    rotation2 = best_individual[10:13]
    scale2 = best_individual[13]
    transformed_points2 = transform_points(points2, translation2, rotation2, scale2)
    
    # 绘制第二个钻石为封闭表面
    try:
        hull2 = ConvexHull(transformed_points2)
        diamond2_mesh = []
        for simplex in hull2.simplices:
            vertices = [transformed_points2[idx] for idx in simplex]
            diamond2_mesh.append(vertices)
        
        color2 = 'red' if type2 == 'round' else 'purple'
        diamond2_poly = Poly3DCollection(diamond2_mesh, alpha=0.7, facecolor=color2, edgecolor='white', linewidth=0.5)
        ax.add_collection3d(diamond2_poly)
    except Exception as e:
        # 如果无法创建凸包，回退到散点图
        print(f"无法为钻石2创建封闭表面: {e}")
        ax.scatter(transformed_points2[:, 0], transformed_points2[:, 1], transformed_points2[:, 2], 
                  c='red' if type2 == 'round' else 'purple', s=10, alpha=0.7, label=f"{'圆形' if type2 == 'round' else '梨形'}钻石2")
    
    # 如果有参数信息，绘制质心和对称轴
    if parameters is not None:
        param1 = parameters[0]
        param2 = parameters[1]
        
        # 绘制第一个钻石的质心和对称轴
        centroid1 = param1['centroid']
        axis1 = param1['symmetry_axis']
        ax.scatter([centroid1[0]], [centroid1[1]], [centroid1[2]], c='blue' if type1 == 'round' else 'green', marker='o', s=100)
        ax.quiver(centroid1[0], centroid1[1], centroid1[2], 
                 axis1[0], axis1[1], axis1[2], length=0.4, color='blue' if type1 == 'round' else 'green')
        
        # 绘制第二个钻石的质心和对称轴
        centroid2 = param2['centroid']
        axis2 = param2['symmetry_axis']
        ax.scatter([centroid2[0]], [centroid2[1]], [centroid2[2]], c='red' if type2 == 'round' else 'purple', marker='o', s=100)
        ax.quiver(centroid2[0], centroid2[1], centroid2[2], 
                 axis2[0], axis2[1], axis2[2], length=0.4, color='red' if type2 == 'round' else 'purple')
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置标题
    ax.set_title(f'双钻石切割优化结果 - {diamond_type_combination}')
    
    # 添加图例
    color1 = 'blue' if type1 == 'round' else 'green'
    color2 = 'red' if type2 == 'round' else 'purple'
    ax.scatter([], [], [], c=color1, marker='o', label=f"{'圆形' if type1 == 'round' else '梨形'}钻石1")
    ax.scatter([], [], [], c=color2, marker='o', label=f"{'圆形' if type2 == 'round' else '梨形'}钻石2")
    ax.legend()
    
    # 保存图像
    os.makedirs('multiple/results', exist_ok=True)
    plt.savefig(f'multiple/results/multi_diamond_result_{diamond_type_combination}.png', dpi=300, bbox_inches='tight')
    
    # 禁用自动旋转视图，允许交互旋转
    plt.tight_layout()
    ax.view_init(elev=30, azim=45)
    plt.show()

def main():
    # --- 添加命令行参数解析 ---
    parser = argparse.ArgumentParser(description='双钻石切割优化')
    parser.add_argument('--combination', type=str, choices=['round_round', 'pear_pear', 'round_pear', 'all'],
                        help='选择要优化的钻石组合类型 (round_round, pear_pear, round_pear, all)')
    args = parser.parse_args()

    # 如果没有提供参数，提示用户选择
    selected_combination = args.combination
    available_combinations = ["round_round", "pear_pear", "round_pear"]

    if selected_combination is None:
        print("请选择要运行的钻石组合:")
        for i, combo in enumerate(available_combinations):
            print(f"{i+1}. {combo}")
        print(f"{len(available_combinations)+1}. all (运行所有组合)")
        
        while True:
            try:
                choice = int(input("请输入选项编号 (1-4): "))
                if 1 <= choice <= len(available_combinations):
                    selected_combination = available_combinations[choice-1]
                    break
                elif choice == len(available_combinations) + 1:
                    selected_combination = 'all'
                    break
                else:
                    print("无效选项，请输入有效编号。")
            except ValueError:
                print("无效输入，请输入数字。")

    # --- 结束参数解析 ---

    # 创建结果目录
    os.makedirs('multiple/results', exist_ok=True)
    
    # 读取原石和钻石数据
    print("读取原石数据...")
    original_vertices, original_faces = read_original_diamond()
    
    print("读取圆形钻石数据...")
    round_points, _ = read_round_diamond()
    
    print("读取梨形钻石数据...")
    pear_points, _ = read_pear_diamond()
    
    print(f"原石顶点数: {len(original_vertices)}")
    print(f"原石面数: {len(original_faces)}")
    print(f"圆形钻石点数: {len(round_points)}")
    print(f"梨形钻石点数: {len(pear_points)}")
    
    # 计算合理的缩放范围
    # 计算钻石的边界框
    round_min = np.min(round_points, axis=0)
    round_max = np.max(round_points, axis=0)
    round_size = round_max - round_min
    
    pear_min = np.min(pear_points, axis=0)
    pear_max = np.max(pear_points, axis=0)
    pear_size = pear_max - pear_min
    
    # 计算原石的边界框
    orig_min = np.min(original_vertices, axis=0)
    orig_max = np.max(original_vertices, axis=0)
    orig_size = orig_max - orig_min
    
    # 估算合理的缩放上限 - 更大的缩放范围
    round_scale_estimate = min(orig_size / round_size) * 1.2  # 增大上限
    pear_scale_estimate = min(orig_size / pear_size) * 1.1
    
    print(f"圆形钻石建议的缩放上限: {round_scale_estimate}")
    print(f"梨形钻石建议的缩放上限: {pear_scale_estimate}")
    
    # 设置优化参数 - 调整以提高性能和收敛性
    pop_size = 150       # 适中的种群大小
    generations = 1000    # 增加代数以获得更好的收敛
    crossover_rate = 0.8 # 保持标准的交叉率
    mutation_rate = 0.25 # 适度的基础变异率
    batch_size = 50      # 调整批处理大小
    early_stop_generations = 30 # 保持提前停止代数
    
    # 计算原石体积
    original_volume = calculate_volume(original_vertices, original_faces)
    print(f"原石体积: {original_volume:.6f} 立方厘米")
    
    # 运行不同钻石组合的优化
    combinations = ["round_round", "pear_pear", "round_pear"]
    best_results = {}
    
    # 根据用户选择确定要运行的组合
    if selected_combination == 'all':
        combinations_to_run = available_combinations
    else:
        combinations_to_run = [selected_combination]

    for diamond_type_combination in combinations_to_run:
        print(f"--- 开始优化 {diamond_type_combination} 组合 ---")
        
        # 使用适当的缩放估计
        if diamond_type_combination == "round_round":
            scale_estimate = round_scale_estimate
        elif diamond_type_combination == "pear_pear":
            scale_estimate = pear_scale_estimate
        else:
            scale_estimate = min(round_scale_estimate, pear_scale_estimate)
        
        # 移除运行多次的循环，只运行一次
        # best_run_fitness = 0
        # best_run_individual = None
        # num_runs = 1 # 只运行一次

        print(f"运行遗传算法 (Pop: {pop_size}, Gen: {generations}, Mut: {mutation_rate})...")
        
        # 运行遗传算法 - 直接获取结果
        individual, fitness = genetic_algorithm_multi_diamond(
            original_vertices, original_faces,
            round_points, pear_points,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            batch_size=batch_size,
            early_stop_generations=early_stop_generations,
            scale_estimate=scale_estimate,
            diamond_type_combination=diamond_type_combination
        )
        
        # 保存结果
        # 检查返回的 individual 和 fitness 是否有效
        if individual is not None and fitness >= 0: # 确保适应度有效
            best_results[diamond_type_combination] = (individual, fitness)
            print(f"完成 {diamond_type_combination} 优化，最佳体积比: {fitness:.6f}")
            # 参数计算和绘图已移至 genetic_algorithm_multi_diamond 内部的成功结束部分
        else:
             best_results[diamond_type_combination] = (None, -1) # 标记失败
             print(f"完成 {diamond_type_combination} 优化，但未能找到有效结果。")

    # 比较不同组合的结果 (仅当运行了多个组合时)
    if len(combinations_to_run) > 1 and best_results:
        print("--- 不同钻石组合的最佳体积比: ---")
        valid_results = {k: v for k, v in best_results.items() if v[1] >= 0} # 过滤无效结果
        
        if valid_results:
            for combo, (individual, fitness) in sorted(valid_results.items(), key=lambda item: item[1][1], reverse=True):
                 print(f"{combo}: {fitness:.6f}")
            
            # 找出最佳组合
            best_combo_item = max(valid_results.items(), key=lambda x: x[1][1])
            best_combo_name = best_combo_item[0]
            best_fitness = best_combo_item[1][1]
            print(f"最佳钻石组合: {best_combo_name}, 体积比: {best_fitness:.6f}")

            # 保存总结信息
            with open('multiple/results/optimization_summary.txt', 'w', encoding='utf-8') as f:
                f.write("双钻石切割优化结果汇总\n")
                f.write("========================\n\n")
                
                for combo, (individual, fitness) in sorted(best_results.items(), key=lambda item: item[1][1] if item[1][1] >= 0 else -1, reverse=True):
                    f.write(f"{combo} 组合:\n")
                    if fitness >= 0:
                        f.write(f"  体积比: {fitness:.6f}\n")
                        f.write(f"  详细参数: 见 diamond_parameters_{combo}.txt 和 best_individual_{combo}.txt\n\n")
                    else:
                        f.write(f"  优化失败或未找到有效解。\n\n")
                
                f.write(f"最佳组合: {best_combo_name}\n")
                f.write(f"最佳体积比: {best_fitness:.6f}\n")
        else:
            print("所有组合均未能找到有效优化结果。")
            # 保存空总结
            with open('multiple/results/optimization_summary.txt', 'w', encoding='utf-8') as f:
                f.write("双钻石切割优化结果汇总\n")
                f.write("========================\n\n")
                f.write("所有组合均未能找到有效优化结果。\n\n")

    elif best_results:
         # 只运行了一个组合
         combo_name = list(best_results.keys())[0]
         fitness = best_results[combo_name][1]
         if fitness >= 0:
             print(f"完成 {combo_name} 优化，最佳体积比: {fitness:.6f}")
         else:
             print(f"完成 {combo_name} 优化，但未找到有效结果。")
         # 单个组合的总结信息可以在其各自的文件中找到

    print("优化流程结束。结果保存在 'multiple/results' 目录下。")

if __name__ == "__main__":
    main()