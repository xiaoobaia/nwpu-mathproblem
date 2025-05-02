import numpy as np
import pandas as pd
import matplotlib
# 设置matplotlib为非交互式后端，避免多线程问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import math
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from scipy.spatial import ConvexHull, distance
import copy
import matplotlib.font_manager as fm
import os

# 创建图像保存目录
os.makedirs('figure/multi_diamond', exist_ok=True)

# 设置随机种子以保证结果可重复性
np.random.seed(42)
random.seed(42)

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义钻石类型枚举
ROUND_DIAMOND = 0
PEAR_DIAMOND = 1

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

def read_diamond_data(file_path):
    """
    通用函数，读取钻石数据文件（适用于圆形和梨形钻石）
    
    参数:
    file_path: 数据文件路径
    
    返回:
    points: 钻石点云
    faces: 面数据（如果有）
    """
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

def read_standard_round_diamond():
    """读取标准圆形钻石数据"""
    return read_diamond_data('data/attachment_2_standarded_round_diamond_geometry_data_file.csv')

def read_standard_pear_diamond():
    """读取标准梨形钻石数据"""
    file_path = 'data/attachment_3_standarded_pear_diamond_geometry_data_file.csv'
    
    # 读取节点数据
    try:
        nodes_data = pd.read_csv(file_path, encoding='utf-8-sig', nrows=52)
        
        # 读取面片数据
        faces_data = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=52)
        
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
    except Exception as e:
        print(f"梨形钻石数据读取错误: {e}")
        print("尝试替代读取方法...")
        
        # 替代读取方法
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        # 找到节点和面的分界线
        split_idx = 0
        for i, line in enumerate(lines):
            if 'Triangle element number' in line:
                split_idx = i
                break
        
        # 读取节点
        vertex_lines = lines[1:split_idx]
        vertices = []
        for line in vertex_lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    vertices.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ])
                except:
                    pass
        
        # 读取面
        face_lines = lines[split_idx+1:]
        faces = []
        for line in face_lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    faces.append([
                        int(parts[1]) - 1,
                        int(parts[2]) - 1,
                        int(parts[3]) - 1
                    ])
                except:
                    pass
        
        return np.array(vertices), np.array(faces)

def transform_points(points, translation, rotation, scale):
    """
    变换点云：先旋转，再缩放，最后平移
    
    参数:
    points: 原始点云数组，形状为(n, 3)
    translation: 平移向量，形状为(3,)
    rotation: 欧拉角（x, y, z）或四元数，用于旋转
    scale: 缩放因子 - 这是一个标量，确保等比缩放
    
    返回:
    变换后的点云，形状为(n, 3)
    """
    # 转换为张量
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # 旋转
    # 这里使用欧拉角，按照x, y, z顺序旋转
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
    
    # 应用等比缩放
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
    # 确保使用相同的缩放因子应用于所有维度
    scaled_points = rotated_points * scale_tensor
    
    # 应用平移
    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=device)
    transformed_points = scaled_points + translation_tensor
    
    return transformed_points.cpu().numpy()

def is_inside_convex_hull(point, hull, tolerance=1e-6):
    """
    使用凸包检查点是否在多面体内部
    
    参数:
    point: 需要检查的点
    hull: scipy.spatial.ConvexHull对象
    tolerance: 容差值，允许点稍微在面的外部
    
    返回:
    如果点在凸包内部或边界上，返回True，否则返回False
    """
    equations = hull.equations
    return np.all(np.dot(equations[:, :-1], point) + equations[:, -1] <= tolerance)

def check_diamond_overlap(transformed_points1, transformed_points2, min_distance=0.05):
    """
    检查两个钻石是否重叠
    
    参数:
    transformed_points1: 第一个钻石的变换后点云
    transformed_points2: 第二个钻石的变换后点云
    min_distance: 两个钻石点云之间的最小允许距离
    
    返回:
    如果钻石重叠，返回True，否则返回False
    """
    # 使用凸包快速检测重叠
    try:
        hull1 = ConvexHull(transformed_points1)
        hull2 = ConvexHull(transformed_points2)
        
        # 计算最小距离
        min_dist = float('inf')
        
        # 使用采样点进行距离计算，减少计算量
        sample_size1 = min(100, len(transformed_points1))
        sample_size2 = min(100, len(transformed_points2))
        
        sample_points1 = transformed_points1[np.random.choice(len(transformed_points1), sample_size1, replace=False)]
        sample_points2 = transformed_points2[np.random.choice(len(transformed_points2), sample_size2, replace=False)]
        
        for p1 in sample_points1:
            for p2 in sample_points2:
                dist = np.linalg.norm(p1 - p2)
                if dist < min_dist:
                    min_dist = dist
                    # 如果已经找到小于阈值的距离，可以提前返回
                    if min_dist < min_distance:
                        return True
        
        return min_dist < min_distance
    except:
        # 如果凸包创建失败，采用保守方法认为重叠
        return True

def calculate_volume_ratio_single(diamond_points, original_vertices, original_faces, params):
    """
    计算单个钻石的体积比
    
    参数:
    diamond_points: 钻石点云
    original_vertices: 原石顶点
    original_faces: 原石面
    params: 包含平移、旋转和缩放的参数数组，形状为(7,)
    
    返回:
    体积比，以及变换后的点云
    """
    # 解析参数
    translation = params[:3]
    rotation = params[3:6]
    scale = params[6]
    
    # 变换钻石点云
    transformed_points = transform_points(diamond_points, translation, rotation, scale)
    
    # 计算原石体积
    try:
        orig_hull = ConvexHull(original_vertices)
        orig_volume = orig_hull.volume
    except:
        return 0.0, transformed_points
    
    # 计算钻石体积
    try:
        diamond_hull = ConvexHull(transformed_points)
        diamond_volume = diamond_hull.volume
    except:
        return 0.0, transformed_points
    
    # 检查钻石是否在原石内部
    inside_count = 0
    total_points = len(transformed_points)
    
    # 更彻底地检查内部点，特别是对于梨形钻石
    # 增加采样点数量，确保能够更全面地检测内部点
    sample_size = min(400, total_points)  # 增加采样点数量
    sample_points = transformed_points[np.random.choice(total_points, sample_size, replace=False)]
    
    # 检查每个采样点是否在原石内部
    for point in sample_points:
        if is_inside_convex_hull(point, orig_hull, tolerance=1e-8):  # 减小容差值
            inside_count += 1
    
    # 计算内部点比例
    inside_ratio = inside_count / sample_size
    
    # 特别严格的内部点要求，确保梨形钻石不会超出原石
    if inside_ratio > 0.99:  # 进一步提高要求，确保几乎所有点都在内部（99%）
        volume_ratio = diamond_volume / orig_volume
        return volume_ratio * inside_ratio, transformed_points
    elif inside_ratio > 0.95:  # 如果大部分点在内部，给予较小惩罚
        volume_ratio = diamond_volume / orig_volume
        # 对接近边界的情况进行轻微惩罚
        penalty = 0.8 + (inside_ratio - 0.95) * 4  # 0.95->0.8, 1.0->1.0的线性映射
        return volume_ratio * penalty, transformed_points
    else:
        # 增加对边缘点的惩罚力度，确保优化过程中更重视内部约束
        penalty_factor = inside_ratio ** 3  # 使用立方关系进一步增强惩罚效果
        return penalty_factor * 0.001, transformed_points

def calculate_volume_ratio_double(diamond1_points, diamond2_points, original_vertices, original_faces, params):
    """
    计算两个钻石的总体积比
    
    参数:
    diamond1_points: 第一个钻石的点云
    diamond2_points: 第二个钻石的点云
    original_vertices: 原石顶点
    original_faces: 原石面
    params: 包含两个钻石的参数，形状为(14,)
    
    返回:
    总体积比，以及两个变换后的点云
    """
    # 解析参数
    params1 = params[:7]
    params2 = params[7:]
    
    # 计算第一个钻石
    volume_ratio1, transformed_points1 = calculate_volume_ratio_single(
        diamond1_points, original_vertices, original_faces, params1)
    
    # 计算第二个钻石
    volume_ratio2, transformed_points2 = calculate_volume_ratio_single(
        diamond2_points, original_vertices, original_faces, params2)
    
    # 检查两个钻石是否重叠
    overlap = check_diamond_overlap(transformed_points1, transformed_points2)
    
    # 如果重叠，惩罚适应度
    if overlap:
        return (volume_ratio1 + volume_ratio2) * 0.1, transformed_points1, transformed_points2
    
    # 返回总体积比
    return volume_ratio1 + volume_ratio2, transformed_points1, transformed_points2

def calculate_fitness_multi_diamond(individual, diamond_config, standard_round, standard_pear, original_vertices, original_faces):
    """
    计算多钻石配置的适应度
    
    参数:
    individual: 个体参数
    diamond_config: 钻石配置类型
        - 0: 一个圆形钻石
        - 1: 一个梨形钻石
        - 2: 两个圆形钻石
        - 3: 两个梨形钻石
        - 4: 一个圆形+一个梨形钻石
    standard_round: 标准圆形钻石点云
    standard_pear: 标准梨形钻石点云
    original_vertices: 原石顶点
    original_faces: 原石面
    
    返回:
    适应度值，以及变换后的点云列表
    """
    if diamond_config == 0:  # 一个圆形钻石
        return calculate_volume_ratio_single(standard_round, original_vertices, original_faces, individual)
    
    elif diamond_config == 1:  # 一个梨形钻石
        return calculate_volume_ratio_single(standard_pear, original_vertices, original_faces, individual)
    
    elif diamond_config == 2:  # 两个圆形钻石
        return calculate_volume_ratio_double(standard_round, standard_round, original_vertices, original_faces, individual)
    
    elif diamond_config == 3:  # 两个梨形钻石
        return calculate_volume_ratio_double(standard_pear, standard_pear, original_vertices, original_faces, individual)
    
    elif diamond_config == 4:  # 一个圆形+一个梨形钻石
        return calculate_volume_ratio_double(standard_round, standard_pear, original_vertices, original_faces, individual)
    
    else:
        raise ValueError(f"未知的钻石配置: {diamond_config}")

def initialize_population_single(pop_size, original_vertices, diamond_points, scale_estimate=None):
    """
    初始化单钻石优化的种群
    
    参数:
    pop_size: 种群大小
    original_vertices: 原石顶点
    diamond_points: 钻石点云
    scale_estimate: 缩放上限估计
    
    返回:
    种群数组，形状为(pop_size, 7)
    """
    # 计算原石边界盒
    orig_min_bounds = np.min(original_vertices, axis=0)
    orig_max_bounds = np.max(original_vertices, axis=0)
    orig_center = (orig_min_bounds + orig_max_bounds) / 2
    orig_extent = orig_max_bounds - orig_min_bounds
    
    # 计算钻石边界盒
    diamond_min_bounds = np.min(diamond_points, axis=0)
    diamond_max_bounds = np.max(diamond_points, axis=0)
    diamond_center = (diamond_min_bounds + diamond_max_bounds) / 2
    diamond_extent = diamond_max_bounds - diamond_min_bounds
    
    # 计算缩放上限
    if scale_estimate is None:
        min_scales = orig_extent / diamond_extent
        scale_estimate = min(min_scales) * 2.5  # 允许较大的缩放来探索
    
    print(f"单钻石使用缩放上限: {scale_estimate}")
    
    # 创建种群
    population = []
    
    # 第一个个体: 中心对齐，不旋转，中等缩放
    initial_best = np.zeros(7)
    initial_best[:3] = orig_center - diamond_center * scale_estimate * 0.8
    initial_best[3:6] = [0, 0, 0]
    initial_best[6] = scale_estimate * 0.8
    population.append(initial_best)
    
    # 创建不同缩放的个体
    scaling_factors = [0.6, 0.7, 0.8, 0.9, 1.0]
    for factor in scaling_factors:
        scaled_individual = np.copy(initial_best)
        scaled_individual[6] = scale_estimate * factor
        population.append(scaled_individual)
    
    # 创建不同旋转的个体
    rotation_angles = [
        [np.pi/4, 0, 0], [0, np.pi/4, 0], [0, 0, np.pi/4],
        [np.pi/4, np.pi/4, 0], [np.pi/4, 0, np.pi/4], [0, np.pi/4, np.pi/4],
        [np.pi/4, np.pi/4, np.pi/4], [np.pi/2, np.pi/2, 0]
    ]
    
    for angles in rotation_angles:
        rotated_individual = np.copy(initial_best)
        rotated_individual[3:6] = angles
        population.append(rotated_individual)
    
    # 创建不同位置的个体
    offsets = [
        [0.2, 0, 0], [-0.2, 0, 0], [0, 0.2, 0], [0, -0.2, 0],
        [0, 0, 0.2], [0, 0, -0.2], [0.2, 0.2, 0], [-0.2, -0.2, 0]
    ]
    
    for offset in offsets:
        translated_individual = np.copy(initial_best)
        translated_individual[:3] += np.array(offset) * orig_extent
        population.append(translated_individual)
    
    # 填充其余个体
    remaining_slots = pop_size - len(population)
    for i in range(remaining_slots):
        if i < remaining_slots * 0.3:  # 30%基于最佳个体有小的随机变化
            individual = np.copy(initial_best)
            individual[:3] += np.random.uniform(-0.3, 0.3, 3) * orig_extent
            individual[3:6] += np.random.uniform(-0.5, 0.5, 3)
            individual[6] *= np.random.uniform(0.7, 1.3)
        else:  # 70%完全随机
            translation = orig_center + np.random.uniform(-0.5, 0.5, 3) * orig_extent * 0.5
            rotation = np.random.uniform(0, 2 * np.pi, 3)
            scale = np.random.uniform(0.3, scale_estimate)
            
            individual = np.concatenate([translation, rotation, [scale]])
        
        population.append(individual)
    
    return np.array(population[:pop_size])

def initialize_population_double(pop_size, original_vertices, diamond1_points, diamond2_points, scale_estimate=None):
    """
    初始化双钻石优化的种群
    
    参数:
    pop_size: 种群大小
    original_vertices: 原石顶点
    diamond1_points: 第一个钻石点云
    diamond2_points: 第二个钻石点云
    scale_estimate: 缩放上限估计
    
    返回:
    种群数组，形状为(pop_size, 14)
    """
    # 计算原石边界盒
    orig_min_bounds = np.min(original_vertices, axis=0)
    orig_max_bounds = np.max(original_vertices, axis=0)
    orig_center = (orig_min_bounds + orig_max_bounds) / 2
    orig_extent = orig_max_bounds - orig_min_bounds
    
    # 计算钻石1边界盒
    d1_min_bounds = np.min(diamond1_points, axis=0)
    d1_max_bounds = np.max(diamond1_points, axis=0)
    d1_center = (d1_min_bounds + d1_max_bounds) / 2
    d1_extent = d1_max_bounds - d1_min_bounds
    
    # 计算钻石2边界盒
    d2_min_bounds = np.min(diamond2_points, axis=0)
    d2_max_bounds = np.max(diamond2_points, axis=0)
    d2_center = (d2_min_bounds + d2_max_bounds) / 2
    d2_extent = d2_max_bounds - d2_min_bounds
    
    # 计算缩放上限
    if scale_estimate is None:
        min_scales1 = orig_extent / d1_extent
        min_scales2 = orig_extent / d2_extent
        scale_estimate1 = min(min_scales1) * 1.8  # 双钻石情况下，单个钻石缩放要小一些
        scale_estimate2 = min(min_scales2) * 1.8
        scale_estimate = (scale_estimate1, scale_estimate2)
    else:
        scale_estimate = (scale_estimate, scale_estimate)
    
    print(f"双钻石使用缩放上限: {scale_estimate}")
    
    # 创建种群
    population = []
    
    # 第一个个体: 两个钻石分别位于原石两端
    initial_best = np.zeros(14)
    
    # 第一个钻石位于左侧
    initial_best[:3] = orig_center - d1_center * scale_estimate[0] * 0.7
    initial_best[:3] += [-orig_extent[0]*0.2, 0, 0]  # 向左偏移
    initial_best[3:6] = [0, 0, 0]
    initial_best[6] = scale_estimate[0] * 0.6
    
    # 第二个钻石位于右侧
    initial_best[7:10] = orig_center - d2_center * scale_estimate[1] * 0.7
    initial_best[7:10] += [orig_extent[0]*0.2, 0, 0]  # 向右偏移
    initial_best[10:13] = [0, 0, 0]
    initial_best[13] = scale_estimate[1] * 0.6
    
    population.append(initial_best)
    
    # 创建不同分布方式的个体
    distributions = [
        # 上下分布
        (np.array([0, 0, -orig_extent[2]*0.2]), np.array([0, 0, orig_extent[2]*0.2])),
        # 前后分布
        (np.array([0, -orig_extent[1]*0.2, 0]), np.array([0, orig_extent[1]*0.2, 0])),
        # 对角线分布
        (np.array([-orig_extent[0]*0.15, -orig_extent[1]*0.15, -orig_extent[2]*0.15]), 
         np.array([orig_extent[0]*0.15, orig_extent[1]*0.15, orig_extent[2]*0.15])),
        # 不同方向的对角线
        (np.array([orig_extent[0]*0.15, -orig_extent[1]*0.15, -orig_extent[2]*0.15]), 
         np.array([-orig_extent[0]*0.15, orig_extent[1]*0.15, orig_extent[2]*0.15]))
    ]
    
    for d1_offset, d2_offset in distributions:
        individual = np.copy(initial_best)
        
        # 修改钻石位置
        individual[:3] = orig_center - d1_center * scale_estimate[0] * 0.7 + d1_offset
        individual[7:10] = orig_center - d2_center * scale_estimate[1] * 0.7 + d2_offset
        
        population.append(individual)
    
    # 创建不同旋转组合的个体
    rotations = [
        ([np.pi/4, 0, 0], [0, np.pi/4, 0]),
        ([0, 0, np.pi/4], [np.pi/4, np.pi/4, 0]),
        ([np.pi/2, 0, 0], [0, np.pi/2, 0])
    ]
    
    for r1, r2 in rotations:
        individual = np.copy(initial_best)
        
        # 修改旋转角度
        individual[3:6] = r1
        individual[10:13] = r2
        
        population.append(individual)
    
    # 创建不同缩放比例的个体
    scales = [
        (0.4, 0.8), (0.8, 0.4), (0.6, 0.6), (0.7, 0.7)
    ]
    
    for s1, s2 in scales:
        individual = np.copy(initial_best)
        
        # 修改缩放比例
        individual[6] = scale_estimate[0] * s1
        individual[13] = scale_estimate[1] * s2
        
        population.append(individual)
    
    # 填充其余个体
    remaining_slots = pop_size - len(population)
    for i in range(remaining_slots):
        if i < remaining_slots * 0.3:  # 30%基于最佳个体有小的随机变化
            individual = np.copy(initial_best)
            
            # 第一个钻石随机变化
            individual[:3] += np.random.uniform(-0.2, 0.2, 3) * orig_extent
            individual[3:6] += np.random.uniform(-0.3, 0.3, 3)
            individual[6] *= np.random.uniform(0.8, 1.2)
            
            # 第二个钻石随机变化
            individual[7:10] += np.random.uniform(-0.2, 0.2, 3) * orig_extent
            individual[10:13] += np.random.uniform(-0.3, 0.3, 3)
            individual[13] *= np.random.uniform(0.8, 1.2)
        else:  # 完全随机
            # 第一个钻石随机参数
            translation1 = orig_center + np.random.uniform(-0.4, 0.4, 3) * orig_extent * 0.5
            rotation1 = np.random.uniform(0, 2 * np.pi, 3)
            scale1 = np.random.uniform(0.3, scale_estimate[0])
            
            # 第二个钻石随机参数
            translation2 = orig_center + np.random.uniform(-0.4, 0.4, 3) * orig_extent * 0.5
            rotation2 = np.random.uniform(0, 2 * np.pi, 3)
            scale2 = np.random.uniform(0.3, scale_estimate[1])
            
            # 组合参数
            params1 = np.concatenate([translation1, rotation1, [scale1]])
            params2 = np.concatenate([translation2, rotation2, [scale2]])
            individual = np.concatenate([params1, params2])
        
        population.append(individual)
    
    return np.array(population[:pop_size])

def selection(population, fitness_values, selection_size):
    """
    基于轮盘赌的选择
    
    参数:
    population: 当前种群
    fitness_values: 适应度值
    selection_size: 要选择的个体数量
    
    返回:
    选择后的种群
    """
    # 确保适应度值为正
    adjusted_fitness = np.maximum(fitness_values, 0)
    
    # 如果所有适应度都为零，则随机选择
    if np.sum(adjusted_fitness) == 0:
        indices = np.random.choice(len(population), selection_size)
        return population[indices]
    
    # 计算选择概率
    probs = adjusted_fitness / np.sum(adjusted_fitness)
    
    # 轮盘赌选择
    indices = np.random.choice(len(population), selection_size, p=probs)
    
    return population[indices]

def crossover(parents, crossover_rate, params_per_diamond):
    """
    均匀交叉
    
    参数:
    parents: 父代个体
    crossover_rate: 交叉概率
    params_per_diamond: 每个钻石的参数数量（7）
    
    返回:
    交叉后的后代
    """
    offspring = []
    
    # 打乱父代顺序
    np.random.shuffle(parents)
    
    # 两两配对
    for i in range(0, len(parents), 2):
        if i + 1 >= len(parents):
            # 如果剩下单个个体，直接添加
            offspring.append(parents[i])
            continue
        
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        if np.random.random() < crossover_rate:
            # 为每个钻石分别进行交叉
            # 创建交叉掩码，保持每个钻石参数的一致性
            mask = []
            
            # 对每个钻石参数块创建掩码
            for j in range(0, len(parent1), params_per_diamond):
                # 90%概率整个钻石参数一起交换
                if np.random.random() < 0.9:
                    # 对一个钻石的所有参数使用相同的交叉决定
                    diamond_mask = np.random.random() < 0.5
                    mask.extend([diamond_mask] * params_per_diamond)
                else:
                    # 对钻石内部的参数分别决定交叉
                    diamond_mask = np.random.random(params_per_diamond) < 0.5
                    mask.extend(diamond_mask)
            
            mask = np.array(mask)
            
            # 使用掩码进行交叉
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            
            offspring.extend([child1, child2])
        else:
            # 不交叉，保持父代不变
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def mutation(offspring, mutation_rate, original_vertices, generation=0, max_generations=100, params_per_diamond=7):
    """
    变异操作
    
    参数:
    offspring: 后代个体
    mutation_rate: 变异概率
    original_vertices: 原石顶点，用于确定合理的变异范围
    generation: 当前代数
    max_generations: 最大代数
    params_per_diamond: 每个钻石的参数数量（7）
    
    返回:
    变异后的后代
    """
    # 计算原石边界盒
    min_bounds = np.min(original_vertices, axis=0)
    max_bounds = np.max(original_vertices, axis=0)
    center = (min_bounds + max_bounds) / 2
    extent = max_bounds - min_bounds
    
    # 根据当前代数动态调整变异强度
    progress = generation / max_generations
    translation_range = 0.4 * (1 - progress) + 0.1  # 从0.4逐渐减小到0.1
    rotation_range = np.pi * (1 - progress) + 0.1  # 从π逐渐减小到0.1
    scale_range = 0.3 * (1 - progress) + 0.05  # 从0.3逐渐减小到0.05
    
    for i in range(len(offspring)):
        # 个体变异概率
        if np.random.random() < mutation_rate:
            # 对每个钻石分别进行变异
            for j in range(0, len(offspring[i]), params_per_diamond):
                # 钻石变异概率
                if np.random.random() < 0.5:
                    # 对钻石的每个参数分别变异
                    for k in range(params_per_diamond):
                        # 参数变异概率
                        if np.random.random() < 0.3:
                            param_idx = j + k
                            if k < 3:  # 平移参数
                                offspring[i][param_idx] += np.random.uniform(-translation_range, translation_range) * extent[k]
                            elif k < 6:  # 旋转参数
                                offspring[i][param_idx] += np.random.uniform(-rotation_range, rotation_range)
                                offspring[i][param_idx] = offspring[i][param_idx] % (2 * np.pi)
                            else:  # 缩放参数
                                offspring[i][param_idx] *= (1 + np.random.uniform(-scale_range, scale_range))
                                offspring[i][param_idx] = max(0.1, offspring[i][param_idx])
    
    return offspring

def calculate_fitness_batch(population, diamond_config, standard_round, standard_pear, 
                           original_vertices, original_faces, batch_size=10):
    """
    批量计算适应度
    
    参数:
    population: 种群
    diamond_config: 钻石配置类型
    standard_round: 标准圆形钻石点云
    standard_pear: 标准梨形钻石点云
    original_vertices: 原石顶点
    original_faces: 原石面
    batch_size: 批大小
    
    返回:
    适应度值数组
    """
    fitness_values = np.zeros(len(population))
    
    # 分批处理，避免一次创建太多线程
    for i in range(0, len(population), batch_size):
        batch = population[i:i+batch_size]
        batch_results = []
        
        # 使用线程池处理批次任务，但限制线程数量
        with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as executor:
            batch_results = list(executor.map(
                lambda idx: calculate_fitness_multi_diamond(
                    batch[idx], diamond_config, standard_round, standard_pear, 
                    original_vertices, original_faces
                ),
                range(len(batch))
            ))
        
        # 存储结果
        for j, result in enumerate(batch_results):
            if diamond_config <= 1:  # 单钻石情况
                fitness, _ = result
            else:  # 双钻石情况
                fitness, _, _ = result
            
            fitness_values[i+j] = fitness
    
    return fitness_values

def genetic_algorithm_multi_diamond(diamond_config, standard_round, standard_pear, original_vertices, original_faces,
                                   pop_size=100, generations=100, crossover_rate=0.8, mutation_rate=0.2,
                                   batch_size=10, early_stop_generations=50, scale_estimate=None):
    """
    多钻石配置的遗传算法优化
    
    参数:
    diamond_config: 钻石配置类型
        - 0: 一个圆形钻石
        - 1: 一个梨形钻石
        - 2: 两个圆形钻石
        - 3: 两个梨形钻石
        - 4: 一个圆形+一个梨形钻石
    standard_round: 标准圆形钻石点云
    standard_pear: 标准梨形钻石点云
    original_vertices: 原石顶点
    original_faces: 原石面
    pop_size: 种群大小
    generations: 迭代代数
    crossover_rate: 交叉概率
    mutation_rate: 变异概率
    batch_size: 批处理大小
    early_stop_generations: 提前停止代数
    scale_estimate: 缩放因子估计值
    
    返回:
    最佳个体和适应度值
    """
    # 确定每个钻石的参数数量
    params_per_diamond = 7
    
    # 初始化种群
    if diamond_config == 0:
        # 单个圆形钻石
        population = initialize_population_single(pop_size, original_vertices, standard_round, scale_estimate)
    elif diamond_config == 1:
        # 单个梨形钻石
        population = initialize_population_single(pop_size, original_vertices, standard_pear, scale_estimate)
    elif diamond_config == 2:
        # 两个圆形钻石
        population = initialize_population_double(pop_size, original_vertices, standard_round, standard_round, scale_estimate)
    elif diamond_config == 3:
        # 两个梨形钻石
        population = initialize_population_double(pop_size, original_vertices, standard_pear, standard_pear, scale_estimate)
    elif diamond_config == 4:
        # 一个圆形+一个梨形钻石
        population = initialize_population_double(pop_size, original_vertices, standard_round, standard_pear, scale_estimate)
    else:
        raise ValueError(f"未知的钻石配置: {diamond_config}")
    
    # 记录历史数据
    best_fitness_history = []
    best_individual = None
    best_fitness = 0
    generations_without_improvement = 0
    
    # 记录开始时间
    start_time = time.time()
    
    # 遗传算法主循环
    for generation in range(generations):
        # 计算适应度
        fitness_values = calculate_fitness_batch(
            population, diamond_config, standard_round, standard_pear,
            original_vertices, original_faces, batch_size
        )
        
        # 找出当前代的最佳个体
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        current_best_individual = population[current_best_idx]
        
        # 更新全局最佳个体
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = copy.deepcopy(current_best_individual)
            generations_without_improvement = 0
            
            # 保存最佳个体
            save_best_individual_multi(best_individual, best_fitness, diamond_config)
            
            # 如果找到较好的解，也保存一下
            if best_fitness > 0.3:  # 体积比超过30%
                print(f"找到较好的解，体积比: {best_fitness:.6f}")
                save_best_individual_multi(best_individual, best_fitness, diamond_config, suffix=f"_vr{best_fitness:.4f}")
        else:
            generations_without_improvement += 1
        
        best_fitness_history.append(best_fitness)
        
        # 输出进度
        if (generation + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"代数 {generation + 1}/{generations}, 最佳适应度: {best_fitness:.6f}, 用时: {elapsed_time:.2f}秒")
        
        # 提前停止条件
        if generations_without_improvement >= early_stop_generations:
            print(f"已经 {early_stop_generations} 代没有改进，提前停止")
            break
        
        # 选择
        selected_parents = selection(population, fitness_values, pop_size)
        
        # 交叉
        offspring = crossover(selected_parents, crossover_rate, params_per_diamond)
        
        # 变异
        offspring = mutation(offspring, mutation_rate, original_vertices, generation, generations, params_per_diamond)
        
        # 精英保留
        if best_individual is not None:
            offspring[0] = best_individual
        
        # 周期性注入随机个体
        if generation % 20 == 0 and generation > 0:
            print("注入随机个体以增加多样性...")
            random_count = int(pop_size * 0.1)  # 10%随机个体
            
            if diamond_config <= 1:
                # 单钻石配置
                diamond_points = standard_round if diamond_config == 0 else standard_pear
                random_individuals = initialize_population_single(random_count, original_vertices, diamond_points, scale_estimate)
            else:
                # 双钻石配置
                diamond1_points = standard_round if diamond_config in [0, 2, 4] else standard_pear
                diamond2_points = standard_round if diamond_config == 2 else standard_pear
                random_individuals = initialize_population_double(random_count, original_vertices, diamond1_points, diamond2_points, scale_estimate)
            
            # 替换最差的个体
            worst_indices = np.argsort(fitness_values)[:random_count]
            for i, idx in enumerate(worst_indices):
                if i < len(random_individuals):
                    offspring[idx] = random_individuals[i]
        
        # 更新种群
        population = offspring
    
    # 计算总用时
    total_time = time.time() - start_time
    print(f"优化完成! 总用时: {total_time:.2f}秒")
    print(f"最佳体积比: {best_fitness:.6f}")
    
    # 绘制优化过程，使用非交互模式
    plt.figure(figsize=(10, 6))
    plt.ioff()  # 关闭交互模式
    plt.plot(best_fitness_history)
    plt.title(f'遗传算法优化过程 - {"圆形" if diamond_config == 0 else "梨形" if diamond_config == 1 else "双圆形" if diamond_config == 2 else "双梨形" if diamond_config == 3 else "圆形+梨形"}钻石')
    plt.xlabel('代数')
    plt.ylabel('最佳体积比')
    plt.grid(True)
    # 保存图像并关闭，不显示
    plt.savefig(f'figure/multi_diamond/optimization_history_config_{diamond_config}.png', dpi=300)
    plt.close()
    
    return best_individual, best_fitness

def save_best_individual_multi(individual, fitness, diamond_config, suffix=""):
    """
    保存最佳个体到文件
    
    参数:
    individual: 个体参数
    fitness: 适应度值（体积比）
    diamond_config: 钻石配置类型
    suffix: 文件名后缀
    """
    config_names = [
        "single_round", "single_pear", "double_round", 
        "double_pear", "round_and_pear"
    ]
    
    config_name = config_names[diamond_config]
    filename = f"figure/multi_diamond/best_{config_name}{suffix}.txt"
    
    # 使用UTF-8编码保存文件，确保中文字符正确显示
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"适应度 (体积比): {fitness}\n")
        
        if diamond_config <= 1:  # 单钻石
            f.write(f"平移向量: {individual[:3]}\n")
            f.write(f"旋转角度: {individual[3:6]}\n")
            f.write(f"缩放因子: {individual[6]}\n")
        else:  # 双钻石
            f.write("钻石1:\n")
            f.write(f"  平移向量: {individual[:3]}\n")
            f.write(f"  旋转角度: {individual[3:6]}\n")
            f.write(f"  缩放因子: {individual[6]}\n")
            f.write("钻石2:\n")
            f.write(f"  平移向量: {individual[7:10]}\n")
            f.write(f"  旋转角度: {individual[10:13]}\n")
            f.write(f"  缩放因子: {individual[13]}\n")
        
        f.write(f"原始数组: {individual}\n")

def calculate_geometric_parameters(diamond_points, params):
    """
    计算钻石的几何参数
    
    参数:
    diamond_points: 钻石点云
    params: 变换参数（平移、旋转、缩放）
    
    返回:
    几何参数字典
    """
    # 解析参数
    translation = params[:3]
    rotation = params[3:6]
    scale = params[6]
    
    # 变换点云
    transformed_points = transform_points(diamond_points, translation, rotation, scale)
    
    # 计算质心
    centroid = np.mean(transformed_points, axis=0)
    
    # 找出最高点和最低点的z坐标
    z_coords = transformed_points[:, 2]
    max_z = np.max(z_coords)
    min_z = np.min(z_coords)
    
    # 计算高度
    total_height = max_z - min_z
    
    # 找出腰围（假设腰围位于z坐标中间附近）
    girdle_z = (max_z + min_z) / 2
    girdle_points = transformed_points[np.abs(z_coords - girdle_z) < 0.05 * total_height]
    
    # 计算腰围上的点到对称轴的距离
    # 假设对称轴垂直于xy平面并通过质心
    xy_distances = np.sqrt(np.sum((girdle_points[:, :2] - centroid[:2])**2, axis=1))
    
    # 对腰围点进行椭圆拟合
    if len(girdle_points) > 5:  # 需要足够多的点进行椭圆拟合
        # 将点投影到xy平面
        xy_points = girdle_points[:, :2]
        
        # 计算点的均值
        mean_xy = np.mean(xy_points, axis=0)
        
        # 中心化点
        centered_points = xy_points - mean_xy
        
        # 计算协方差矩阵
        cov = np.cov(centered_points.T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # 确保特征值是按降序排列的
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 计算主半轴和次半轴长度
        major_semi_axis = np.sqrt(eigenvalues[0]) * 2  # 乘以2获取直径
        minor_semi_axis = np.sqrt(eigenvalues[1]) * 2
    else:
        # 如果点不够，使用简单估计
        major_semi_axis = np.max(xy_distances)
        minor_semi_axis = np.mean(xy_distances)
    
    # 计算离心率
    if major_semi_axis > 0:
        eccentricity = np.sqrt(1 - (minor_semi_axis / major_semi_axis)**2)
    else:
        eccentricity = 0
    
    # 计算腰围高度
    if len(girdle_points) > 0:
        girdle_height = np.max(girdle_points[:, 2]) - np.min(girdle_points[:, 2])
    else:
        girdle_height = 0.05 * total_height  # 默认值
    
    # 计算上锥体和下锥体高度
    upper_cone_height = max_z - girdle_z
    lower_cone_height = girdle_z - min_z
    
    # 计算mp和mc参数
    # 在标准钻石中，假设上锥体中间有一个点，下锥体中间也有一个点
    upper_mid_z = (max_z + girdle_z) / 2
    lower_mid_z = (min_z + girdle_z) / 2
    
    # 找出靠近这些中点的点
    upper_mid_points = transformed_points[np.abs(z_coords - upper_mid_z) < 0.05 * upper_cone_height]
    lower_mid_points = transformed_points[np.abs(z_coords - lower_mid_z) < 0.05 * lower_cone_height]
    
    # 计算这些中点到对称轴的距离
    if len(upper_mid_points) > 0:
        upper_mid_distances = np.sqrt(np.sum((upper_mid_points[:, :2] - centroid[:2])**2, axis=1))
        upper_mid_radius = np.mean(upper_mid_distances)
        mc = upper_mid_radius / (major_semi_axis / 2)  # 与主半轴比例
    else:
        mc = 0.5  # 默认值
    
    if len(lower_mid_points) > 0:
        lower_mid_distances = np.sqrt(np.sum((lower_mid_points[:, :2] - centroid[:2])**2, axis=1))
        lower_mid_radius = np.mean(lower_mid_distances)
        mp = lower_mid_radius / (major_semi_axis / 2)  # 与主半轴比例
    else:
        mp = 0.5  # 默认值
    
    # 计算基础半主轴与上下锥体的角度
    # 这需要估计锥体的斜率
    if upper_cone_height > 0:
        tan_bc = ((major_semi_axis / 2) * (1 - mc)) / upper_cone_height
        bc = np.arctan(tan_bc)
    else:
        bc = np.pi / 4  # 默认值
    
    if lower_cone_height > 0:
        tan_bp = ((major_semi_axis / 2) * (1 - mp)) / lower_cone_height
        bp = np.arctan(tan_bp)
    else:
        bp = np.pi / 3  # 默认值
    
    # 计算对称轴方向
    # 在变换后的点云中，对称轴可能不再严格垂直于xy平面
    # 这里我们使用主成分分析(PCA)来估计
    pca = np.linalg.svd(transformed_points - centroid)[2]
    symmetric_axis_direction = pca[0]  # 第一个主成分方向
    
    # 确保对称轴方向朝上
    if symmetric_axis_direction[2] < 0:
        symmetric_axis_direction = -symmetric_axis_direction
    
    # 归一化方向向量
    symmetric_axis_direction = symmetric_axis_direction / np.linalg.norm(symmetric_axis_direction)
    
    # 返回计算的参数
    parameters = {
        "Major_semi_axis_length": major_semi_axis / 2,  # 转换为半轴长度
        "Minor_semi_axis_length": minor_semi_axis / 2,  # 转换为半轴长度
        "Eccentricity": eccentricity,
        "Height_of_girdle": girdle_height,
        "Lower_cone_height": lower_cone_height,
        "mp": mp,
        "Upper_cone_height": upper_cone_height,
        "mc": mc,
        "bc": bc,
        "bp": bp,
        "Centroid": centroid,
        "Symmetric_axis_direction": symmetric_axis_direction,
        "Total_height": total_height
    }
    
    return parameters

def save_parameters_multi(parameters, diamond_config, diamond_index=0):
    """
    保存几何参数到文件
    
    参数:
    parameters: 包含几何参数的字典
    diamond_config: 钻石配置类型
    diamond_index: 在多钻石配置中的钻石索引（0或1）
    """
    config_names = [
        "single_round", "single_pear", "double_round", 
        "double_pear", "round_and_pear"
    ]
    
    config_name = config_names[diamond_config]
    suffix = "" if diamond_config <= 1 else f"_diamond{diamond_index+1}"
    
    filename = f"figure/multi_diamond/parameters_{config_name}{suffix}.txt"
    
    # 使用UTF-8编码保存文件，确保中文字符正确显示
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("钻石几何参数:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Major semi-axis length (a): {parameters['Major_semi_axis_length']:.6f}\n")
        f.write(f"Minor semi-axis length (b): {parameters['Minor_semi_axis_length']:.6f}\n")
        f.write(f"Eccentricity (e): {parameters['Eccentricity']:.6f}\n")
        f.write(f"Height of girdle (D): {parameters['Height_of_girdle']:.6f}\n")
        f.write(f"Lower cone height (Lp): {parameters['Lower_cone_height']:.6f}\n")
        f.write(f"mp: {parameters['mp']:.6f}\n")
        f.write(f"Upper cone height (Lc): {parameters['Upper_cone_height']:.6f}\n")
        f.write(f"mc: {parameters['mc']:.6f}\n")
        f.write(f"Angle bc: {parameters['bc']:.6f} radians = {np.degrees(parameters['bc']):.2f} degrees\n")
        f.write(f"Angle bp: {parameters['bp']:.6f} radians = {np.degrees(parameters['bp']):.2f} degrees\n")
        f.write("-" * 30 + "\n")
        f.write(f"钻石质心位置: {parameters['Centroid']}\n")
        f.write(f"对称轴方向: {parameters['Symmetric_axis_direction']}\n")
        f.write(f"总高度: {parameters['Total_height']:.6f}\n")

def plot_result_multi(original_vertices, original_faces, diamond_config, best_individual, 
                     standard_round, standard_pear, fitness):
    """
    绘制多钻石优化结果
    
    参数:
    original_vertices: 原石顶点
    original_faces: 原石面
    diamond_config: 钻石配置类型
    best_individual: 最佳个体
    standard_round: 标准圆形钻石点云
    standard_pear: 标准梨形钻石点云
    fitness: 最佳适应度值（体积比）
    """
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原石
    for face in original_faces:
        verts = [original_vertices[idx] for idx in face]
        tri = Poly3DCollection([verts], alpha=0.2)
        tri.set_color('lightblue')
        ax.add_collection3d(tri)
    
    # 根据钻石配置生成变换后的点云
    transformed_points = []
    parameters_list = []
    
    if diamond_config <= 1:  # 单钻石
        # 获取要使用的钻石点云
        diamond_points = standard_round if diamond_config == 0 else standard_pear
        diamond_type = "圆形" if diamond_config == 0 else "梨形"
        
        # 解析参数
        params = best_individual
        
        # 变换点云
        transformed_point_cloud = transform_points(diamond_points, params[:3], params[3:6], params[6])
        transformed_points.append(transformed_point_cloud)
        
        # 计算几何参数
        parameters = calculate_geometric_parameters(diamond_points, params)
        parameters_list.append(parameters)
        
        # 保存参数
        save_parameters_multi(parameters, diamond_config)
        
        # 绘制变换后的点云
        ax.scatter(transformed_point_cloud[:, 0], transformed_point_cloud[:, 1], transformed_point_cloud[:, 2],
                  c='red', s=10, label=f'标准{diamond_type}钻石')
        
        # 绘制点云的凸包
        try:
            hull = ConvexHull(transformed_point_cloud)
            for simplex in hull.simplices:
                verts = transformed_point_cloud[simplex]
                tri = Poly3DCollection([verts], alpha=0.3)
                tri.set_color('red')
                ax.add_collection3d(tri)
        except:
            print("无法创建钻石表面，使用原始点云显示")
        
        # 绘制质心和对称轴
        centroid = parameters['Centroid']
        axis_direction = parameters['Symmetric_axis_direction']
        
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                  c='green', s=100, marker='o', label='钻石质心')
        
        # 绘制对称轴
        total_height = parameters['Total_height']
        axis_length = total_height * 1.2
        
        axis_start = centroid - axis_direction * (axis_length / 2)
        axis_end = centroid + axis_direction * (axis_length / 2)
        
        ax.plot([axis_start[0], axis_end[0]],
               [axis_start[1], axis_end[1]],
               [axis_start[2], axis_end[2]],
               'g-', linewidth=2, label='对称轴')
    
    else:  # 双钻石
        # 确定两个钻石的类型和点云
        if diamond_config == 2:  # 两个圆形
            diamond1_points = standard_round
            diamond2_points = standard_round
            diamond1_type = "圆形"
            diamond2_type = "圆形"
        elif diamond_config == 3:  # 两个梨形
            diamond1_points = standard_pear
            diamond2_points = standard_pear
            diamond1_type = "梨形"
            diamond2_type = "梨形"
        else:  # 一个圆形+一个梨形
            diamond1_points = standard_round
            diamond2_points = standard_pear
            diamond1_type = "圆形"
            diamond2_type = "梨形"
        
        # 解析两个钻石的参数
        params1 = best_individual[:7]
        params2 = best_individual[7:]
        
        # 变换两个钻石点云
        transformed_point_cloud1 = transform_points(diamond1_points, params1[:3], params1[3:6], params1[6])
        transformed_point_cloud2 = transform_points(diamond2_points, params2[:3], params2[3:6], params2[6])
        
        transformed_points.extend([transformed_point_cloud1, transformed_point_cloud2])
        
        # 计算几何参数
        parameters1 = calculate_geometric_parameters(diamond1_points, params1)
        parameters2 = calculate_geometric_parameters(diamond2_points, params2)
        
        parameters_list.extend([parameters1, parameters2])
        
        # 保存参数
        save_parameters_multi(parameters1, diamond_config, 0)
        save_parameters_multi(parameters2, diamond_config, 1)
        
        # 绘制第一个钻石
        ax.scatter(transformed_point_cloud1[:, 0], transformed_point_cloud1[:, 1], transformed_point_cloud1[:, 2],
                  c='red', s=10, label=f'第一个{diamond1_type}钻石')
        
        # 绘制第一个钻石的凸包
        try:
            hull1 = ConvexHull(transformed_point_cloud1)
            for simplex in hull1.simplices:
                verts = transformed_point_cloud1[simplex]
                tri = Poly3DCollection([verts], alpha=0.3)
                tri.set_color('red')
                ax.add_collection3d(tri)
        except:
            print("无法创建第一个钻石表面，使用原始点云显示")
        
        # 绘制第二个钻石
        ax.scatter(transformed_point_cloud2[:, 0], transformed_point_cloud2[:, 1], transformed_point_cloud2[:, 2],
                  c='blue', s=10, label=f'第二个{diamond2_type}钻石')
        
        # 绘制第二个钻石的凸包
        try:
            hull2 = ConvexHull(transformed_point_cloud2)
            for simplex in hull2.simplices:
                verts = transformed_point_cloud2[simplex]
                tri = Poly3DCollection([verts], alpha=0.3)
                tri.set_color('blue')
                ax.add_collection3d(tri)
        except:
            print("无法创建第二个钻石表面，使用原始点云显示")
        
        # 绘制两个钻石的质心和对称轴
        centroid1 = parameters1['Centroid']
        axis_direction1 = parameters1['Symmetric_axis_direction']
        
        centroid2 = parameters2['Centroid']
        axis_direction2 = parameters2['Symmetric_axis_direction']
        
        ax.scatter([centroid1[0]], [centroid1[1]], [centroid1[2]],
                  c='green', s=100, marker='o', label='第一个钻石质心')
        ax.scatter([centroid2[0]], [centroid2[1]], [centroid2[2]],
                  c='cyan', s=100, marker='o', label='第二个钻石质心')
        
        # 绘制对称轴
        total_height1 = parameters1['Total_height']
        axis_length1 = total_height1 * 1.2
        
        axis_start1 = centroid1 - axis_direction1 * (axis_length1 / 2)
        axis_end1 = centroid1 + axis_direction1 * (axis_length1 / 2)
        
        ax.plot([axis_start1[0], axis_end1[0]],
               [axis_start1[1], axis_end1[1]],
               [axis_start1[2], axis_end1[2]],
               'g-', linewidth=2, label='第一个钻石对称轴')
        
        total_height2 = parameters2['Total_height']
        axis_length2 = total_height2 * 1.2
        
        axis_start2 = centroid2 - axis_direction2 * (axis_length2 / 2)
        axis_end2 = centroid2 + axis_direction2 * (axis_length2 / 2)
        
        ax.plot([axis_start2[0], axis_end2[0]],
               [axis_start2[1], axis_end2[1]],
               [axis_start2[2], axis_end2[2]],
               'c-', linewidth=2, label='第二个钻石对称轴')
    
    # 设置坐标轴标签
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    
    # 配置名称
    config_names = [
        "单个圆形钻石", "单个梨形钻石", "两个圆形钻石",
        "两个梨形钻石", "一个圆形+一个梨形钻石"
    ]
    
    # 设置标题
    plt.title(f'优化后的{config_names[diamond_config]}在原石中的位置 (体积比: {fitness:.4f})')
    
    # 添加图例
    ax.legend()
    
    # 在图上添加体积比信息
    param_text = f"体积比: {fitness:.4f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存图像并关闭，不显示
    plt.savefig(f'figure/multi_diamond/optimized_{config_file_names[diamond_config]}.png', dpi=300)
    plt.close()
    
    return transformed_points, parameters_list

def compare_all_configurations(results):
    """
    比较所有钻石配置的结果
    
    参数:
    results: 包含每种配置结果的字典
    {
        config_id: (best_individual, best_fitness),
        ...
    }
    """
    # 配置名称
    config_names = [
        "单个圆形钻石", "单个梨形钻石", "两个圆形钻石",
        "两个梨形钻石", "一个圆形+一个梨形钻石"
    ]
    
    # 提取体积比
    fitness_values = [results[i][1] for i in range(len(results))]
    
    # 找出最佳配置
    best_config_idx = np.argmax(fitness_values)
    best_config_fitness = fitness_values[best_config_idx]
    
    # 绘制比较图
    plt.ioff()  # 关闭交互模式
    plt.figure(figsize=(12, 6))
    bars = plt.bar(config_names, fitness_values, color='skyblue')
    bars[best_config_idx].set_color('red')
    
    plt.title('不同钻石配置的体积比比较')
    plt.xlabel('钻石配置')
    plt.ylabel('体积比')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for i, v in enumerate(fitness_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    # 保存图像并关闭，不显示
    plt.savefig('figure/multi_diamond/configuration_comparison.png', dpi=300)
    plt.close()
    
    # 打印比较结果
    print("\n==== 钻石配置比较 ====")
    for i, config in enumerate(config_names):
        prefix = ">>> " if i == best_config_idx else "    "
        print(f"{prefix}{config}: 体积比 = {fitness_values[i]:.6f}")
    
    print(f"\n最佳钻石配置: {config_names[best_config_idx]}")
    print(f"最佳体积比: {best_config_fitness:.6f}")
    
    # 将比较结果保存到文件
    with open('figure/multi_diamond/configuration_comparison.txt', 'w') as f:
        f.write("钻石配置体积比比较:\n")
        f.write("-" * 40 + "\n")
        for i, config in enumerate(config_names):
            f.write(f"{config}: {fitness_values[i]:.6f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"最佳钻石配置: {config_names[best_config_idx]}\n")
        f.write(f"最佳体积比: {best_config_fitness:.6f}\n")
    
    return best_config_idx, best_config_fitness

# 主程序入口点
if __name__ == "__main__":
    # 设置matplotlib为非交互模式，避免多线程问题
    plt.ioff()
    
    print("多钻石切割优化程序")
    print("=" * 50)
    
    # 读取数据
    print("正在读取数据...")
    original_vertices, original_faces = read_original_diamond()
    standard_round_points, _ = read_standard_round_diamond()
    standard_pear_points, _ = read_standard_pear_diamond()
    
    print(f"原石顶点数: {len(original_vertices)}")
    print(f"原石面数: {len(original_faces)}")
    print(f"标准圆形钻石点数: {len(standard_round_points)}")
    print(f"标准梨形钻石点数: {len(standard_pear_points)}")
    
    # 遗传算法参数
    pop_size = 300
    generations = 300
    crossover_rate = 0.8
    mutation_rate = 0.2
    batch_size = 20
    early_stop_generations = 50
    
    # 配置菜单
    config_names = [
        "单个圆形钻石", 
        "单个梨形钻石", 
        "两个圆形钻石",
        "两个梨形钻石", 
        "一个圆形+一个梨形钻石"
    ]
    
    # 创建菜单
    print("\n请选择要运行的钻石配置:")
    print("-" * 50)
    for i, name in enumerate(config_names):
        print(f"{i}. {name}")
    print(f"{len(config_names)}. 运行所有配置")
    print("-" * 50)
    
    # 获取用户选择
    while True:
        try:
            choice = int(input("请输入选择 [0-5]: "))
            if 0 <= choice <= len(config_names):
                break
            else:
                print(f"无效的选择，请输入0-{len(config_names)}之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    # 存储所有配置的结果
    all_results = {}
    
    # 根据用户选择运行相应的配置
    if choice < len(config_names):
        # 运行单个配置
        config_id = choice
        config_name = config_names[config_id]
        
        print(f"\n正在优化 {config_name}...")
        print("-" * 50)
        
        # 运行遗传算法
        best_individual, best_fitness = genetic_algorithm_multi_diamond(
            config_id, standard_round_points, standard_pear_points,
            original_vertices, original_faces,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            batch_size=batch_size,
            early_stop_generations=early_stop_generations
        )
        
        # 存储结果
        all_results[config_id] = (best_individual, best_fitness)
        
        # 绘制最佳结果
        transformed_points, parameters = plot_result_multi(
            original_vertices, original_faces, config_id,
            best_individual, standard_round_points, standard_pear_points,
            best_fitness
        )
        
        print(f"\n优化完成!")
        print(f"配置: {config_name}")
        print(f"最佳体积比: {best_fitness:.6f}")
        
    else:
        # 运行所有配置
        for config_id, config_name in enumerate(config_names):
            print(f"\n正在优化 {config_name}...")
            print("-" * 50)
            
            # 运行遗传算法
            best_individual, best_fitness = genetic_algorithm_multi_diamond(
                config_id, standard_round_points, standard_pear_points,
                original_vertices, original_faces,
                pop_size=pop_size,
                generations=generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                batch_size=batch_size,
                early_stop_generations=early_stop_generations
            )
            
            # 存储结果
            all_results[config_id] = (best_individual, best_fitness)
            
            # 绘制最佳结果
            transformed_points, parameters = plot_result_multi(
                original_vertices, original_faces, config_id,
                best_individual, standard_round_points, standard_pear_points,
                best_fitness
            )
        
        # 比较所有配置
        best_config, best_fitness = compare_all_configurations(all_results)
        
        print("\n所有优化完成!")
        print(f"最佳钻石配置: {config_names[best_config]}")
        print(f"最佳体积比: {best_fitness:.6f}")