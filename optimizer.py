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

# 确保输出目录存在
if not os.path.exists('multiple'):
    os.makedirs('multiple')
if not os.path.exists('multiple/results'):
    os.makedirs('multiple/results')

# 设置随机种子以保证结果可重复性
np.random.seed(42)
random.seed(42)

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

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

def calculate_scale_limits(original_vertices, round_points, pear_points):

    """计算更合理的缩放上限"""
    # 计算原石和钻石的尺寸
    orig_size = np.ptp(original_vertices, axis=0)
    round_size = np.ptp(round_points, axis=0)
    pear_size = np.ptp(pear_points, axis=0)
    
    # 计算最大可能缩放 - 对梨形钻石更加保守
    round_scale = min(orig_size / round_size) * 0.7
    pear_scale = min(orig_size / pear_size) * 0.4  # 降低到0.4的保守值
    
    return round_scale, pear_scale
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
        if distance > tolerance:
            return False
    return True

def is_overlapping(points1, points2, hull1=None, hull2=None):
    """
    判断两个钻石是否重叠
    
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
        hull1 = ConvexHull(points1)
    if hull2 is None:
        hull2 = ConvexHull(points2)
    
    # 检查第一个钻石的点是否在第二个钻石内部
    for point in points1:
        if is_inside_convex_hull(point, hull2):
            return True
    
    # 检查第二个钻石的点是否在第一个钻石内部
    for point in points2:
        if is_inside_convex_hull(point, hull1):
            return True
    
    # 没有重叠的点
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
        hull = ConvexHull(points)
        simplices = hull.simplices
    else:
        simplices = faces
    
    # 计算体积
    volume = 0
    for simplex in simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # 计算四面体体积：V = (1/6) * abs(v1·(v2×v3))
        volume += abs(np.dot(v1, np.cross(v2, v3))) / 6
    
    return volume

def calculate_volume_ratio_with_penalty(transformed_points1, transformed_points2, original_vertices, original_faces, 
                                      individual, use_second_diamond=True):
    """添加额外的边界惩罚项的适应度函数"""
    # 获取钻石类型
    type1 = individual[14]
    type2 = individual[15]
    
    # 点云已经提前变换好了，直接使用
    
    # 计算原石体积和凸包
    original_volume = calculate_volume(original_vertices, original_faces)
    orig_hull = ConvexHull(original_vertices)
    
    # 检查点是否在原石内并计算体积
    diamond_volume1 = calculate_volume(transformed_points1)
    diamond_volume2 = calculate_volume(transformed_points2)
    
    # 检查第一个钻石的点
    points_outside1 = 0
    for p in transformed_points1:
        if not is_inside_convex_hull(p, orig_hull):
            points_outside1 += 1
    
    # 检查第二个钻石的点
    points_outside2 = 0
    for p in transformed_points2:
        if not is_inside_convex_hull(p, orig_hull):
            points_outside2 += 1
    
    # 如果有点在外面，添加惩罚项
    volume_ratio = 0
    if points_outside1 == 0 and points_outside2 == 0:
        # 检查重叠
        if is_overlapping(transformed_points1, transformed_points2):
            # 重叠情况，只计算其中一个钻石的体积
            volume_ratio = max(diamond_volume1, diamond_volume2) / original_volume
        else:
            # 没有重叠，计算总体积
            volume_ratio = (diamond_volume1 + diamond_volume2) / original_volume
    else:
        # 梨形钻石有更严格的惩罚
        penalty1 = points_outside1 / len(transformed_points1) * (0.5 if type1 == 0 else 0.8)
        penalty2 = points_outside2 / len(transformed_points2) * (0.5 if type2 == 0 else 0.8)
        
        # 根据外部点的比例减少体积比
        valid_volume1 = diamond_volume1 * (1 - penalty1) if points_outside1 > 0 else diamond_volume1
        valid_volume2 = diamond_volume2 * (1 - penalty2) if points_outside2 > 0 else diamond_volume2
        
        # 非常严重的情况直接返回0
        if points_outside1 > len(transformed_points1) * 0.1 or points_outside2 > len(transformed_points2) * 0.1:
            volume_ratio = 0
        else:
            # 否则返回惩罚后的体积比
            volume_ratio = (valid_volume1 + valid_volume2) / original_volume
    
    return volume_ratio

def initialize_population(pop_size, original_vertices, round_points, pear_points, 
                        round_scale_limit, pear_scale_limit):
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
    
    # 初始化种群
    population = []
    for _ in range(pop_size):
        # 圆形钻石使用较大范围
        tx1 = orig_center[0] + np.random.uniform(-orig_size[0]/5, orig_size[0]/5)
        ty1 = orig_center[1] + np.random.uniform(-orig_size[1]/5, orig_size[1]/5)
        tz1 = orig_center[2] + np.random.uniform(-orig_size[2]/5, orig_size[2]/5)
        
        # 梨形钻石使用更小范围，更靠近中心
        tx2 = orig_center[0] + np.random.uniform(-orig_size[0]/6, orig_size[0]/6)
        ty2 = orig_center[1] + np.random.uniform(-orig_size[1]/6, orig_size[1]/6)
        tz2 = orig_center[2] + np.random.uniform(-orig_size[2]/6, orig_size[2]/6)
        
        # 旋转角度
        rx1 = np.random.uniform(0, 2 * np.pi)
        ry1 = np.random.uniform(0, 2 * np.pi)
        rz1 = np.random.uniform(0, 2 * np.pi)
        
        rx2 = np.random.uniform(0, 2 * np.pi)
        ry2 = np.random.uniform(0, 2 * np.pi)
        rz2 = np.random.uniform(0, 2 * np.pi)
        
        # 更保守的缩放
        s1 = np.random.uniform(0.1, round_scale_limit * 0.9)  # 圆形钻石
        s2 = np.random.uniform(0.1, pear_scale_limit * 0.8)   # 梨形钻石
        
        # 钻石类型
        type1 = 0  # 圆形
        type2 = 1  # 梨形
        
        individual = [tx1, ty1, tz1, rx1, ry1, rz1, s1, 
                     tx2, ty2, tz2, rx2, ry2, rz2, s2, 
                     type1, type2]
        population.append(individual)
    
    return np.array(population)

def selection(population, fitness_values, selection_size):
    """
    选择操作 - 锦标赛选择
    
    参数:
    population: 当前种群
    fitness_values: 适应度值
    selection_size: 选择的个体数
    
    返回:
    选择的父代
    """
    selected = []
    pop_size = len(population)
    
    for _ in range(selection_size):
        # 随机选择几个个体进行锦标赛
        tournament_size = 5
        tournament_idx = np.random.choice(pop_size, tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament_idx]
        
        # 选择适应度最高的个体
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx])
    
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

def mutation(offspring, mutation_rate, original_vertices, generation=0, max_generations=100):
    """
    变异操作
    
    参数:
    offspring: 后代个体
    mutation_rate: 变异概率
    original_vertices: 原石的顶点
    generation: 当前代数
    max_generations: 最大代数
    
    返回:
    变异后的后代个体
    """
    num_offspring = len(offspring)
    
    # 计算原石的中心和边界框
    orig_center = np.mean(original_vertices, axis=0)
    orig_min = np.min(original_vertices, axis=0)
    orig_max = np.max(original_vertices, axis=0)
    orig_size = orig_max - orig_min
    
    # 随着代数增加，减小变异步长
    mutation_scale = 1.0 - 0.9 * (generation / max_generations)
    
    # 从全局函数调用中获取圆形和梨形钻石的点云
    try:
        round_points_global = globals().get('round_points')
        pear_points_global = globals().get('pear_points')
        if round_points_global is not None and pear_points_global is not None:
            round_scale_limit, pear_scale_limit = calculate_scale_limits(original_vertices, round_points_global, pear_points_global)
        else:
            # 保守估计
            round_scale_limit = 0.6
            pear_scale_limit = 0.3
    except:
        # 出错时使用保守估计
        round_scale_limit = 0.6
        pear_scale_limit = 0.3
    
    for i in range(num_offspring):
        # 获取当前个体的钻石类型
        diamond1_type = offspring[i, 14]  # 0=圆形, 1=梨形
        diamond2_type = offspring[i, 15]  # 0=圆形, 1=梨形
        
        # 对每个参数单独进行变异
        for j in range(14):  # 只变异前14个参数，不变异钻石类型(14和15)
            if np.random.random() < mutation_rate:
                if j < 3:  # 第一个钻石的平移
                    # 梨形钻石使用更小的变异范围
                    scale_factor = 8 if diamond1_type == 0 else 10
                    # 在原石范围内小幅度变异
                    if j == 0:
                        offspring[i, j] += np.random.normal(0, orig_size[0]/scale_factor) * mutation_scale
                    elif j == 1:
                        offspring[i, j] += np.random.normal(0, orig_size[1]/scale_factor) * mutation_scale
                    else:
                        offspring[i, j] += np.random.normal(0, orig_size[2]/scale_factor) * mutation_scale
                    
                    # 确保平移后的位置不会太靠近边界
                    if diamond1_type == 1:  # 梨形钻石
                        # 强制将位置拉回中心一些
                        offspring[i, j] = 0.8 * offspring[i, j] + 0.2 * orig_center[j]
                elif j < 6:  # 第一个钻石的旋转
                    # 随机变异旋转角度
                    offspring[i, j] += np.random.normal(0, np.pi/8) * mutation_scale
                    # 确保角度在0到2π之间
                    offspring[i, j] = offspring[i, j] % (2 * np.pi)
                elif j == 6:  # 第一个钻石的缩放
                    # 随机变异缩放因子
                    offspring[i, j] *= (1 + np.random.normal(0, 0.1) * mutation_scale)
                    # 确保缩放因子不会太小且不会超过限制
                    if diamond1_type == 0:  # 圆形钻石
                        offspring[i, j] = max(0.1, min(offspring[i, j], round_scale_limit * 0.9))
                    else:  # 梨形钻石
                        offspring[i, j] = max(0.1, min(offspring[i, j], pear_scale_limit * 0.6))
                elif j < 10:  # 第二个钻石的平移
                    # 梨形钻石使用更小的变异范围
                    scale_factor = 8 if diamond2_type == 0 else 10
                    if j == 7:
                        offspring[i, j] += np.random.normal(0, orig_size[0]/scale_factor) * mutation_scale
                    elif j == 8:
                        offspring[i, j] += np.random.normal(0, orig_size[1]/scale_factor) * mutation_scale
                    else:
                        offspring[i, j] += np.random.normal(0, orig_size[2]/scale_factor) * mutation_scale
                    
                    # 确保平移后的位置不会太靠近边界
                    if diamond2_type == 1:  # 梨形钻石
                        # 强制将位置拉回中心一些
                        offspring[i, j] = 0.8 * offspring[i, j] + 0.2 * orig_center[j-7]
                elif j < 13:  # 第二个钻石的旋转
                    offspring[i, j] += np.random.normal(0, np.pi/8) * mutation_scale
                    offspring[i, j] = offspring[i, j] % (2 * np.pi)
                elif j == 13:  # 第二个钻石的缩放
                    offspring[i, j] *= (1 + np.random.normal(0, 0.1) * mutation_scale)
                    # 确保缩放因子不会太小且不会超过限制
                    if diamond2_type == 0:  # 圆形钻石
                        offspring[i, j] = max(0.1, min(offspring[i, j], round_scale_limit * 0.9))
                    else:  # 梨形钻石
                        offspring[i, j] = max(0.1, min(offspring[i, j], pear_scale_limit * 0.6))
    
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
    orig_hull = ConvexHull(original_vertices)
    
    # 分批计算适应度
    for i in range(0, pop_size, batch_size):
        batch = population[i:i+batch_size]
        batch_fitness = np.zeros(len(batch))
        
        # 使用多线程计算适应度
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for j, individual in enumerate(batch):
                # 根据钻石类型选择不同的点云
                d1_points = diamond1_points if individual[14] == 0 else diamond2_points
                d2_points = diamond1_points if individual[15] == 0 else diamond2_points
                
                # 提取参数
                translation1 = individual[:3]
                rotation1 = individual[3:6]
                scale1 = individual[6]
                
                translation2 = individual[7:10]
                rotation2 = individual[10:13]
                scale2 = individual[13]
                
                # 变换点云
                transformed_points1 = transform_points(d1_points, translation1, rotation1, scale1)
                transformed_points2 = transform_points(d2_points, translation2, rotation2, scale2)
                
                # 对梨形钻石应用特殊处理，将超出原石的点推回内部
                if individual[14] == 1:  # 第一个是梨形钻石
                    transformed_points1 = push_inside_original(transformed_points1, original_vertices, orig_hull)
                
                if individual[15] == 1:  # 第二个是梨形钻石
                    transformed_points2 = push_inside_original(transformed_points2, original_vertices, orig_hull)
                
                futures.append(
                    executor.submit(
                        calculate_volume_ratio_with_penalty, 
                        transformed_points1, transformed_points2, 
                        original_vertices, original_faces, 
                        individual, True
                    )
                )
            
            # 获取结果
            for j, future in enumerate(futures):
                batch_fitness[j] = future.result()
        
        # 存储结果
        fitness_values[i:i+len(batch)] = batch_fitness
    
    return fitness_values

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
    # 计算更合理的缩放限制
    round_scale_limit, pear_scale_limit = calculate_scale_limits(original_vertices, round_points, pear_points)
    
    # 根据钻石类型组合设置初始种群
    initial_population = initialize_population(pop_size, original_vertices, round_points, pear_points, 
                                             round_scale_limit, pear_scale_limit)
    
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
    best_fitness = 0
    generations_without_improvement = 0
    
    start_time = time.time()
    
    # 遗传算法主循环
    for generation in range(generations):
        # 计算每个个体的适应度
        fitness_values = calculate_fitness_batch(
            population, round_points, pear_points, original_vertices, original_faces, batch_size
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
            
            # 保存当前最佳个体到文件
            save_best_individual(best_individual, best_fitness, diamond_type_combination)
            
            # 如果找到了比较好的解，也保存一下
            if best_fitness > 0.5:  # 如果体积比超过50%，保存这个解
                print(f"找到较好的解，体积比为: {best_fitness:.6f}")
                save_best_individual(best_individual, best_fitness, f"{diamond_type_combination}_vr{best_fitness:.4f}")
        else:
            generations_without_improvement += 1
        
        best_fitness_history.append(best_fitness)
        
        # 输出进度
        if (generation + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"类型: {diamond_type_combination}, 代数: {generation + 1}/{generations}, 最佳适应度: {best_fitness:.6f}, 用时: {elapsed_time:.2f}秒")
        
        # 提前停止条件
        if generations_without_improvement >= early_stop_generations:
            print(f"已经 {early_stop_generations} 代没有改进，提前停止优化")
            break
        
        # 选择
        selected_parents = selection(population, fitness_values, pop_size)
        
        # 交叉
        offspring = crossover(selected_parents, crossover_rate)
        
        # 变异 - 加入代数信息用于动态调整变异强度
        offspring = mutation(offspring, mutation_rate, original_vertices, generation, generations)
        
        # 精英保留策略：保留最佳个体
        if best_individual is not None:
            offspring[0] = best_individual
        
        # 更新种群
        population = offspring
    
    total_time = time.time() - start_time
    print(f"优化完成！类型: {diamond_type_combination}, 总用时: {total_time:.2f}秒")
    print(f"最佳体积比: {best_fitness:.6f}")
    
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

def push_inside_original(points, original_vertices, hull=None):
    """将超出原石的点推回原石内部"""
    if hull is None:
        hull = ConvexHull(original_vertices)
    
    orig_center = np.mean(original_vertices, axis=0)
    adjusted_points = points.copy()
    
    for i, point in enumerate(points):
        if not is_inside_convex_hull(point, hull):
            # 计算从中心到这个点的方向
            direction = point - orig_center
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # 进行二分搜索找到边界上的点
            dist_min = 0
            dist_max = np.linalg.norm(point - orig_center)
            max_iter = 20
            current_iter = 0
            
            while current_iter < max_iter:
                mid_dist = (dist_min + dist_max) / 2
                test_point = orig_center + direction * mid_dist
                
                if is_inside_convex_hull(test_point, hull):
                    dist_min = mid_dist
                else:
                    dist_max = mid_dist
                    
                current_iter += 1
            
            # 将点稍微推入内部
            adjusted_points[i] = orig_center + direction * (dist_min * 0.9)
    
    return adjusted_points
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
    
    # 估算合理的缩放范围
    round_scale_estimate = min(orig_size / round_size) * 0.9
    pear_scale_estimate = min(orig_size / pear_size) * 0.9
    
    print(f"圆形钻石建议的缩放上限: {round_scale_estimate}")
    print(f"梨形钻石建议的缩放上限: {pear_scale_estimate}")
    
    # 设置优化参数
    pop_size = 200
    generations = 200
    crossover_rate = 0.8
    mutation_rate = 0.2
    batch_size = 20
    early_stop_generations = 50
    # 计算原石体积
    original_volume = calculate_volume(original_vertices, original_faces)
    print(f"原石体积: {original_volume:.6f} 立方厘米")
    # 运行不同钻石组合的优化
    combinations = ["round_round", "pear_pear", "round_pear"]
    best_results = {}
    
    for diamond_type_combination in combinations:
        print(f"\n开始优化 {diamond_type_combination} 组合...")
        
        # 运行遗传算法
        best_individual, best_fitness = genetic_algorithm_multi_diamond(
            original_vertices, original_faces,
            round_points, pear_points,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            batch_size=batch_size,
            early_stop_generations=early_stop_generations,
            diamond_type_combination=diamond_type_combination
        )
        
        best_results[diamond_type_combination] = (best_individual, best_fitness)
        
       # 计算钻石参数
        parameters = calculate_parameters(best_individual, round_points, pear_points, diamond_type_combination)
        save_parameters(parameters, diamond_type_combination, original_volume)
        
        # 绘制结果 - Matplotlib
        plot_result(original_vertices, original_faces, round_points, pear_points, 
                   best_individual, diamond_type_combination, parameters)
        
        # 绘制结果 - Plotly交互式HTML
        try:
            import plotly.graph_objects as go
            plot_result_plotly(original_vertices, original_faces, round_points, pear_points,
                             best_individual, diamond_type_combination, parameters)
            print(f"已生成交互式HTML文件: multiple/results/multi_diamond_result_{diamond_type_combination}.html")
        except ImportError:
            print("未安装plotly库，跳过生成交互式HTML文件")
    
    # 比较不同组合的结果
    print("\n不同钻石组合的最佳体积比:")
    for combo, (individual, fitness) in best_results.items():
        print(f"{combo}: {fitness:.6f}")
    
    # 找出最佳组合
    best_combo = max(best_results.items(), key=lambda x: x[1][1])
    print(f"\n最佳钻石组合: {best_combo[0]}, 体积比: {best_combo[1][1]:.6f}")
    
    # 计算并保存最佳组合的详细参数
    best_individual, best_fitness = best_combo[1]
    parameters = calculate_parameters(best_individual, round_points, pear_points, best_combo[0])
    save_parameters(parameters, "best_combination", original_volume)  # 添加原石体积参数
    
    
    # 保存总结信息
    with open('multiple/results/optimization_summary.txt', 'w') as f:
        f.write("双钻石切割优化结果汇总\n")
        f.write("========================\n\n")
        
        for combo, (individual, fitness) in best_results.items():
            f.write(f"{combo} 组合:\n")
            f.write(f"  体积比: {fitness:.6f}\n")
            f.write(f"  详细参数: 见 diamond_parameters_{combo}.txt\n\n")
        
        f.write(f"最佳组合: {best_combo[0]}\n")
        f.write(f"最佳体积比: {best_combo[1][1]:.6f}\n")

if __name__ == "__main__":
    main()