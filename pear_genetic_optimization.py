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
import matplotlib.font_manager as fm
import os

# 确保multiple目录存在
if not os.path.exists('multiple'):
    os.makedirs('multiple')

# 设置随机种子以保证结果可重复性
np.random.seed(43)
random.seed(43)

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

def read_pear_diamond():
    """读取梨形钻石数据"""
    # 读取CSV文件
    file_path = 'data/attachment_3_standarded_pear_diamond_geometry_data_file.csv'
    
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

def transform_points(points, translation, rotation, scale):
    """
    变换点云：先旋转，再缩放，最后平移
    
    参数:
    points: 原始点云数组，形状为(n, 3)
    translation: 平移向量，形状为(3,)
    rotation: 欧拉角（x, y, z）或四元数，用于旋转
    scale: 缩放因子
    
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
    
    # 应用缩放
    scale = torch.tensor(scale, dtype=torch.float32, device=device)
    scaled_points = rotated_points * scale
    
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

def calculate_volume_ratio(pear_points, original_vertices, original_faces, individual):
    """
    计算梨形钻石和原石的体积比
    
    参数:
    pear_points: 梨形钻石的点云
    original_vertices: 原石的顶点
    original_faces: 原石的面
    individual: 包含平移、旋转和缩放参数的个体
    
    返回:
    体积比，如果梨形钻石不完全在原石内部，则返回低适应度
    """
    # 解析个体参数
    translation = individual[:3]
    rotation = individual[3:6]
    scale = individual[6]
    
    # 变换梨形钻石点云
    transformed_points = transform_points(pear_points, translation, rotation, scale)
    
    # 计算原石的凸包
    try:
        orig_hull = ConvexHull(original_vertices)
        orig_volume = orig_hull.volume
    except:
        return 0.0
    
    # 计算梨形钻石的凸包
    try:
        pear_hull = ConvexHull(transformed_points)
        pear_volume = pear_hull.volume
    except:
        return 0.0
    
    # 检查梨形钻石是否在原石内部
    # 检查所有点是否都在内部
    inside_count = 0
    for point in transformed_points:
        if is_inside_convex_hull(point, orig_hull, tolerance=1e-6):
            inside_count += 1
    
    # 计算内部点比例
    inside_ratio = inside_count / len(transformed_points)
    
    # 如果所有点都在内部，则计算实际体积比
    if inside_ratio == 1.0:
        volume_ratio = pear_volume / orig_volume
        return volume_ratio
    else:
        # 返回一个很小但非零的适应度值，用内部点比例来区分
        return inside_ratio * 0.001  # 按照内部点比例给予小的适应度值

def initialize_population(pop_size, original_vertices, pear_points, scale_estimate=None):
    """
    初始化种群 - 改进版
    
    每个个体包含:
    - 平移向量 (3)
    - 旋转角度 (3)
    - 缩放因子 (1)
    
    参数:
    pop_size: 种群大小
    original_vertices: 原石的顶点
    pear_points: 梨形钻石的点云
    scale_estimate: 估算的合理缩放上限，如果为None则自动计算
    
    返回形状为 (pop_size, 7) 的数组
    """
    # 计算原石的边界盒以确定平移范围
    orig_min_bounds = np.min(original_vertices, axis=0)
    orig_max_bounds = np.max(original_vertices, axis=0)
    orig_center = (orig_min_bounds + orig_max_bounds) / 2
    orig_extent = orig_max_bounds - orig_min_bounds
    
    # 计算梨形钻石的边界盒
    pear_min_bounds = np.min(pear_points, axis=0)
    pear_max_bounds = np.max(pear_points, axis=0)
    pear_center = (pear_min_bounds + pear_max_bounds) / 2
    pear_extent = pear_max_bounds - pear_min_bounds
    
    # 计算原石的凸包体积
    orig_hull = ConvexHull(original_vertices)
    orig_volume = orig_hull.volume
    
    # 计算梨形钻石的凸包体积
    pear_hull = ConvexHull(pear_points)
    pear_volume = pear_hull.volume
    
    # 计算理论上可能的最大缩放因子（假设钻石完全填充原石）
    theory_scale = (orig_volume / pear_volume) ** (1/3)
    print(f"理论最大缩放因子: {theory_scale:.6f}")
    
    # 计算缩放范围，考虑理论最大值
    if scale_estimate is None:
        # 考虑几何形状不同，保守估计为理论值的80%
        scale_estimate = theory_scale * 0.8
    
    # 确保scale_estimate不超过原石最小尺寸与梨形钻石最大尺寸之比
    min_scales = orig_extent / pear_extent
    scale_bound = min(min_scales) * 1.5
    scale_estimate = min(scale_estimate, scale_bound)
    
    print(f"使用缩放上限: {scale_estimate}")
    
    population = []
    
    # 第一个个体：使用最佳估计参数（钻石中心对齐，标准旋转，合理缩放）
    initial_best = np.zeros(7)
    initial_best[:3] = orig_center - pear_center * scale_estimate * 0.8  # 平移以对齐中心
    initial_best[3:6] = [0, 0, 0]  # 不旋转
    initial_best[6] = scale_estimate * 0.8  # 设置为估计最大缩放的80%
    population.append(initial_best)
    
    # 创建不同缩放比例的初始个体
    scaling_factors = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    for factor in scaling_factors:
        # 创建不同缩放的个体
        scaled_individual = np.copy(initial_best)
        scaled_individual[6] = scale_estimate * factor  # 使用不同的缩放因子
        population.append(scaled_individual)
    
    # 使用更多不同的旋转角度组合
    for x_angle in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
        for y_angle in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            for z_angle in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
                if len(population) >= pop_size * 0.7:  # 控制生成的个体数量
                    break
                    
                # 如果角度都为0，则跳过（已经有一个这样的个体）
                if x_angle == 0 and y_angle == 0 and z_angle == 0:
                    continue
                    
                # 创建不同旋转的个体
                rotated_individual = np.copy(initial_best)
                rotated_individual[3:6] = [x_angle, y_angle, z_angle]
                population.append(rotated_individual)
    
    # 在原石内创建均匀分布的平移位置
    translation_grid = []
    grid_steps = 3  # 每个维度的步数
    
    # 在原石内部创建均匀分布的平移位置网格
    for x_pos in np.linspace(orig_min_bounds[0] + 0.2 * orig_extent[0], 
                            orig_max_bounds[0] - 0.2 * orig_extent[0], grid_steps):
        for y_pos in np.linspace(orig_min_bounds[1] + 0.2 * orig_extent[1], 
                                orig_max_bounds[1] - 0.2 * orig_extent[1], grid_steps):
            for z_pos in np.linspace(orig_min_bounds[2] + 0.2 * orig_extent[2], 
                                    orig_max_bounds[2] - 0.2 * orig_extent[2], grid_steps):
                translation = np.array([x_pos, y_pos, z_pos])
                translation_grid.append(translation - pear_center * scale_estimate * 0.8)
    
    # 从网格中随机选择平移位置
    np.random.shuffle(translation_grid)
    for translation in translation_grid[:min(len(translation_grid), pop_size // 10)]:
        translated_individual = np.copy(initial_best)
        translated_individual[:3] = translation
        population.append(translated_individual)
    
    # 其余个体：添加随机变化
    remaining_slots = pop_size - len(population)
    if remaining_slots > 0:
        print(f"添加 {remaining_slots} 个随机个体")
        
        # 30%的个体基于最佳估计参数，但有小的随机变化
        for i in range(int(remaining_slots * 0.3)):
            individual = np.copy(initial_best)
            # 小范围随机平移
            individual[:3] += np.random.uniform(-0.2, 0.2, 3) * orig_extent
            # 小范围随机旋转
            individual[3:6] += np.random.uniform(-0.5, 0.5, 3)
            # 小范围随机缩放
            individual[6] *= np.random.uniform(0.8, 1.2)
            population.append(individual)
        
        # 其余个体完全随机
        for i in range(remaining_slots - int(remaining_slots * 0.3)):
            # 随机平移在原石范围内
            translation = orig_center + np.random.uniform(-0.5, 0.5, 3) * orig_extent * 0.5
            # 随机旋转
            rotation = np.random.uniform(0, 2 * np.pi, 3)
            # 随机缩放，但在合理范围内
            scale = np.random.uniform(0.6 * scale_estimate, scale_estimate)
            
            individual = np.concatenate([translation, rotation, [scale]])
            population.append(individual)
    
    # 确保返回的种群大小正确
    if len(population) > pop_size:
        population = population[:pop_size]
    
    # 如果种群大小不足，复制现有个体填补
    while len(population) < pop_size:
        idx = np.random.randint(0, len(population))
        population.append(np.copy(population[idx]))
    
    return np.array(population)

def selection(population, fitness_values, selection_size):
    """
    基于轮盘赌的选择
    
    参数:
    population: 当前种群
    fitness_values: 对应的适应度值
    selection_size: 要选择的个体数
    
    返回:
    选择后的种群
    """
    # 确保适应度值为正
    adjusted_fitness = np.maximum(fitness_values, 0)
    
    # 如果所有适应度都为0，则随机选择
    if np.sum(adjusted_fitness) == 0:
        indices = np.random.choice(len(population), selection_size)
        return population[indices]
    
    # 计算选择概率
    probs = adjusted_fitness / np.sum(adjusted_fitness)
    
    # 轮盘赌选择
    indices = np.random.choice(len(population), selection_size, p=probs)
    
    return population[indices]

def crossover(parents, crossover_rate):
    """
    使用多种交叉方法进行染色体交换 - 改进版
    
    参数:
    parents: 父代个体
    crossover_rate: 交叉概率
    
    返回:
    交叉后产生的新个体
    """
    offspring = []
    
    # 打乱父代顺序
    np.random.shuffle(parents)
    
    # 两两配对
    for i in range(0, len(parents), 2):
        if i + 1 >= len(parents):
            offspring.append(parents[i])
            continue
        
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        if np.random.random() < crossover_rate:
            # 随机选择交叉方法
            cross_method = np.random.choice(['uniform', 'blx_alpha', 'sbx'])
            
            if cross_method == 'uniform':
                # 均匀交叉
                mask = np.random.random(len(parent1)) < 0.5
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            
            elif cross_method == 'blx_alpha':
                # BLX-alpha交叉 (用于实数编码)
                alpha = 0.5
                # 对每个参数单独处理
                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent2)
                
                for j in range(len(parent1)):
                    # 确定范围
                    min_val = min(parent1[j], parent2[j])
                    max_val = max(parent1[j], parent2[j])
                    range_val = max_val - min_val
                    
                    # 扩展范围
                    min_bound = min_val - alpha * range_val
                    max_bound = max_val + alpha * range_val
                    
                    # 在扩展范围内随机生成子代值
                    child1[j] = np.random.uniform(min_bound, max_bound)
                    child2[j] = np.random.uniform(min_bound, max_bound)
                    
                    # 特殊处理旋转角度，确保在0-2π范围内
                    if 3 <= j < 6:
                        child1[j] = child1[j] % (2 * np.pi)
                        child2[j] = child2[j] % (2 * np.pi)
                    
                    # 确保缩放因子为正
                    if j == 6:
                        child1[j] = max(0.1, child1[j])
                        child2[j] = max(0.1, child2[j])
            
            elif cross_method == 'sbx':
                # 模拟二进制交叉 (SBX)
                eta = 15  # 分布指数，越大子代越接近父代
                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent2)
                
                for j in range(len(parent1)):
                    # 确保父代不完全相同
                    if abs(parent1[j] - parent2[j]) > 1e-10:
                        if parent1[j] < parent2[j]:
                            y1, y2 = parent1[j], parent2[j]
                        else:
                            y1, y2 = parent2[j], parent1[j]
                        
                        # 生成随机数
                        u = np.random.random()
                        
                        # 计算beta
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        
                        # 生成子代
                        child1[j] = 0.5 * ((1 + beta) * y1 + (1 - beta) * y2)
                        child2[j] = 0.5 * ((1 - beta) * y1 + (1 + beta) * y2)
                    else:
                        # 如果父代相同，直接复制
                        child1[j] = parent1[j]
                        child2[j] = parent2[j]
                    
                    # 特殊处理旋转角度，确保在0-2π范围内
                    if 3 <= j < 6:
                        child1[j] = child1[j] % (2 * np.pi)
                        child2[j] = child2[j] % (2 * np.pi)
                    
                    # 确保缩放因子为正
                    if j == 6:
                        child1[j] = max(0.1, child1[j])
                        child2[j] = max(0.1, child2[j])
            
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def mutation(offspring, mutation_rate, original_vertices, generation=0, max_generations=100, best_fitness=0):
    """
    对后代进行变异 - 改进版，支持自适应变异
    
    参数:
    offspring: 后代个体
    mutation_rate: 变异概率
    original_vertices: 原石顶点，用于确定合理的平移范围
    generation: 当前代数
    max_generations: 最大代数
    best_fitness: 当前最佳适应度值，用于自适应变异
    
    返回:
    变异后的后代
    """
    # 计算原石的边界盒
    min_bounds = np.min(original_vertices, axis=0)
    max_bounds = np.max(original_vertices, axis=0)
    center = (min_bounds + max_bounds) / 2
    extent = max_bounds - min_bounds
    
    # 根据当前代数动态调整变异强度
    # 早期探索更广泛的空间，后期进行微调
    progress = generation / max_generations
    
    # 如果当前最佳适应度较低，增加变异强度以提高探索能力
    # 基础变异强度随进度减小，但如果适应度低，则保持较高变异强度
    if best_fitness < 0.2:  # 适应度阈值可调整
        # 如果适应度低，保持较高的变异强度，以增强搜索能力
        exploration_factor = max(0.8, 1 - progress * 0.5)
    else:
        # 如果适应度已经不错，逐渐减小变异强度，进行微调
        exploration_factor = 1 - progress * 0.8
    
    # 基于当前阶段和适应度调整变异参数
    translation_range = 0.5 * exploration_factor + 0.1  # 从0.5逐渐减小
    rotation_range = np.pi * exploration_factor + 0.1   # 从π逐渐减小
    scale_range = 0.3 * exploration_factor + 0.05      # 从0.3逐渐减小
    
    # 确定每个个体的自适应变异率
    for i in range(len(offspring)):
        # 个体变异概率 - 可能根据个体相对适应度调整
        individual_mutation_rate = mutation_rate
        
        if np.random.random() < individual_mutation_rate:
            # 使用三种变异策略之一
            mutation_strategy = np.random.choice(['uniform', 'gaussian', 'cauchy'])
            
            if mutation_strategy == 'uniform':
                # 均匀变异 - 每个参数单独变异
                for j in range(len(offspring[i])):
                    if np.random.random() < 0.3:  # 每个参数的变异概率
                        if j < 3:  # 平移参数
                            # 在原石范围内变异
                            offspring[i][j] += np.random.uniform(-translation_range, translation_range) * extent[j]
                        elif j < 6:  # 旋转参数
                            # 随机旋转角度变异
                            offspring[i][j] += np.random.uniform(-rotation_range, rotation_range)
                            # 保持角度在0到2π范围内
                            offspring[i][j] = offspring[i][j] % (2 * np.pi)
                        else:  # 缩放参数
                            # 随机缩放因子变异，但保持在合理范围内
                            offspring[i][j] *= (1 + np.random.uniform(-scale_range, scale_range))
                            # 确保缩放因子不会太小
                            offspring[i][j] = max(0.1, offspring[i][j])
            
            elif mutation_strategy == 'gaussian':
                # 高斯变异 - 适合局部搜索
                for j in range(len(offspring[i])):
                    if np.random.random() < 0.3:  # 每个参数的变异概率
                        if j < 3:  # 平移参数
                            # 在原石范围内使用高斯变异
                            offspring[i][j] += np.random.normal(0, translation_range * 0.5) * extent[j]
                        elif j < 6:  # 旋转参数
                            # 高斯旋转角度变异
                            offspring[i][j] += np.random.normal(0, rotation_range * 0.5)
                            # 保持角度在0到2π范围内
                            offspring[i][j] = offspring[i][j] % (2 * np.pi)
                        else:  # 缩放参数
                            # 高斯缩放因子变异
                            offspring[i][j] *= (1 + np.random.normal(0, scale_range * 0.5))
                            # 确保缩放因子不会太小
                            offspring[i][j] = max(0.1, offspring[i][j])
            
            elif mutation_strategy == 'cauchy':
                # 柯西变异 - 产生较大变化的机会更多，有助于跳出局部最优
                for j in range(len(offspring[i])):
                    if np.random.random() < 0.3:  # 每个参数的变异概率
                        if j < 3:  # 平移参数
                            # 在原石范围内使用柯西变异
                            cauchy_noise = np.random.standard_cauchy() * translation_range * 0.3
                            # 限制柯西噪声的大小，避免过大扰动
                            cauchy_noise = np.clip(cauchy_noise, -translation_range, translation_range)
                            offspring[i][j] += cauchy_noise * extent[j]
                        elif j < 6:  # 旋转参数
                            # 柯西旋转角度变异
                            cauchy_noise = np.random.standard_cauchy() * rotation_range * 0.3
                            cauchy_noise = np.clip(cauchy_noise, -rotation_range, rotation_range)
                            offspring[i][j] += cauchy_noise
                            # 保持角度在0到2π范围内
                            offspring[i][j] = offspring[i][j] % (2 * np.pi)
                        else:  # 缩放参数
                            # 柯西缩放因子变异
                            cauchy_noise = np.random.standard_cauchy() * scale_range * 0.3
                            cauchy_noise = np.clip(cauchy_noise, -scale_range, scale_range)
                            offspring[i][j] *= (1 + cauchy_noise)
                            # 确保缩放因子不会太小
                            offspring[i][j] = max(0.1, offspring[i][j])
    
    return offspring

def calculate_fitness_batch(population, pear_points, original_vertices, original_faces, batch_size=10):
    """
    批量计算适应度，可以充分利用GPU并行计算能力
    
    参数:
    population: 种群
    pear_points: 梨形钻石的点云
    original_vertices: 原石的顶点
    original_faces: 原石的面
    batch_size: 批大小
    
    返回:
    每个个体的适应度值数组
    """
    fitness_values = np.zeros(len(population))
    
    # 分批处理
    for i in range(0, len(population), batch_size):
        batch = population[i:i+batch_size]
        
        # 并行计算适应度
        with ThreadPoolExecutor() as executor:
            batch_fitness = list(executor.map(
                lambda idx: calculate_volume_ratio(
                    pear_points, original_vertices, original_faces, batch[idx]
                ),
                range(len(batch))
            ))
        
        # 存储结果
        fitness_values[i:i+len(batch)] = batch_fitness
    
    return fitness_values

def plot_original_and_pear_diamonds(original_vertices, original_faces, pear_points, pear_triangles):
    """
    绘制原石和梨形钻石的形状，帮助直观理解数据
    
    参数:
    original_vertices: 原石的顶点
    original_faces: 原石的面
    pear_points: 梨形钻石的点云
    pear_triangles: 梨形钻石的三角面
    """
    # 创建3D图形
    fig = plt.figure(figsize=(15, 6))
    
    # 绘制原石
    ax1 = fig.add_subplot(121, projection='3d')
    for face in original_faces:
        verts = [original_vertices[idx] for idx in face]
        tri = Poly3DCollection([verts], alpha=0.2)
        tri.set_color('lightblue')
        ax1.add_collection3d(tri)
    
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_zlabel('Z (cm)')
    ax1.set_title('原石形状')
    
    # 计算原石的范围以设置坐标轴
    min_bounds = np.min(original_vertices, axis=0)
    max_bounds = np.max(original_vertices, axis=0)
    
    # 设置坐标轴范围
    ax1.set_xlim(min_bounds[0], max_bounds[0])
    ax1.set_ylim(min_bounds[1], max_bounds[1])
    ax1.set_zlim(min_bounds[2], max_bounds[2])
    
    # 绘制梨形钻石
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 绘制梨形钻石的面
    for triangle in pear_triangles:
        verts = [pear_points[idx] for idx in triangle]
        tri = Poly3DCollection([verts], alpha=0.5)
        tri.set_color('red')
        ax2.add_collection3d(tri)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('梨形钻石')
    
    # 计算梨形钻石的范围以设置坐标轴
    pear_min_bounds = np.min(pear_points, axis=0)
    pear_max_bounds = np.max(pear_points, axis=0)
    
    # 设置坐标轴范围
    ax2.set_xlim(pear_min_bounds[0], pear_max_bounds[0])
    ax2.set_ylim(pear_min_bounds[1], pear_max_bounds[1])
    ax2.set_zlim(pear_min_bounds[2], pear_max_bounds[2])
    
    plt.tight_layout()
    plt.savefig('multiple/original_and_pear_diamonds.png', dpi=300)
    plt.show()

def calculate_parameters(best_individual, pear_points):
    """
    计算最佳个体对应的几何参数
    
    参数:
    best_individual: 包含最佳平移、旋转和缩放参数的个体
    pear_points: 梨形钻石的点云
    
    返回:
    包含所需几何参数的字典
    """
    # 解析个体参数
    translation = best_individual[:3]
    rotation = best_individual[3:6]
    scale = best_individual[6]
    
    # 变换梨形钻石点云
    transformed_points = transform_points(pear_points, translation, rotation, scale)
    
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
    
    # 保存参数到文件
    save_parameters(parameters, scale_factor=scale)
    
    return parameters

def save_parameters(parameters, scale_factor):
    """
    保存几何参数到文件
    
    参数:
    parameters: 包含几何参数的字典
    scale_factor: 缩放因子
    """
    with open('multiple/pear_geometric_parameters.txt', 'w', encoding='utf-8') as f:
        f.write("梨形钻石几何参数:\n")
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
        f.write(f"缩放比例 (Scale): {scale_factor:.6f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"钻石质心位置: {parameters['Centroid']}\n")
        f.write(f"对称轴方向: {parameters['Symmetric_axis_direction']}\n")
        f.write(f"总高度: {parameters['Total_height']:.6f}\n")

def plot_result(original_vertices, original_faces, pear_points, pear_triangles, best_individual, best_fitness, parameters=None):
    """
    绘制最佳结果
    
    参数:
    original_vertices: 原石的顶点
    original_faces: 原石的面
    pear_points: 梨形钻石的点云
    pear_triangles: 梨形钻石的三角面
    best_individual: 包含最佳平移、旋转和缩放参数的个体
    best_fitness: 最佳体积比
    parameters: 几何参数字典
    """
    # 解析个体参数
    translation = best_individual[:3]
    rotation = best_individual[3:6]
    scale = best_individual[6]
    
    # 变换梨形钻石点云
    transformed_points = transform_points(pear_points, translation, rotation, scale)
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始钻石
    for face in original_faces:
        verts = [original_vertices[idx] for idx in face]
        tri = Poly3DCollection([verts], alpha=0.2)
        tri.set_color('lightblue')
        ax.add_collection3d(tri)
    
    # 绘制变换后的梨形钻石
    for triangle in pear_triangles:
        # 获取变换后的顶点
        verts = [transformed_points[idx] for idx in triangle]
        tri = Poly3DCollection([verts], alpha=0.5)
        tri.set_color('red')
        ax.add_collection3d(tri)
    
    # 如果有参数，绘制对称轴和中心
    if parameters:
        centroid = parameters['Centroid']
        axis_direction = parameters['Symmetric_axis_direction']
        
        # 绘制质心
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], 
                   c='green', s=100, marker='o', label='钻石质心')
        
        # 绘制对称轴
        # 计算轴的起点和终点，使其穿过整个钻石
        total_height = parameters['Total_height']
        axis_length = total_height * 1.2  # 稍微长一点以便看清
        
        # 计算轴的起点和终点
        axis_start = centroid - axis_direction * (axis_length / 2)
        axis_end = centroid + axis_direction * (axis_length / 2)
        
        # 绘制对称轴
        ax.plot([axis_start[0], axis_end[0]], 
                [axis_start[1], axis_end[1]], 
                [axis_start[2], axis_end[2]], 
                'g-', linewidth=2, label='对称轴')
    
    # 设置坐标轴标签
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    
    # 设置标题
    plt.title('优化后的梨形钻石在原石中的位置')
    
    # 添加图例
    ax.legend()
    
    # 在图上添加参数信息
    param_text = (
        f"体积比: {best_fitness:.4f}\n"
        f"主半轴长度: {parameters['Major_semi_axis_length']:.4f}\n"
        f"次半轴长度: {parameters['Minor_semi_axis_length']:.4f}\n"
        f"离心率: {parameters['Eccentricity']:.4f}\n"
        f"总高度: {parameters['Total_height']:.4f}"
    )
    plt.figtext(0.02, 0.02, param_text, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.savefig('multiple/pear_diamond_optimization_result.png', dpi=300)
    
    # 显示图形
    plt.show()

def save_best_individual(individual, fitness, suffix=""):
    """
    保存最佳个体到文件
    
    参数:
    individual: 个体参数数组
    fitness: 适应度值
    suffix: 文件名后缀，用于保存多个解
    """
    filename = f"multiple/pear_best_individual{suffix}.txt"
    with open(filename, 'w') as f:
        f.write(f"适应度 (体积比): {fitness}\n")
        f.write(f"平移向量: {individual[:3]}\n")
        f.write(f"旋转角度: {individual[3:6]}\n")
        f.write(f"缩放因子: {individual[6]}\n")
        f.write(f"原始数组: {individual}\n")

def genetic_algorithm(pear_points, pear_triangles, original_vertices, original_faces, 
                     pop_size=100, generations=100, 
                     crossover_rate=0.8, mutation_rate=0.1,
                     batch_size=10, early_stop_generations=50,
                     scale_estimate=None):
    """
    使用遗传算法寻找最优解 - 改进版
    
    参数:
    pear_points: 梨形钻石的点云
    pear_triangles: 梨形钻石的三角面
    original_vertices: 原石的顶点
    original_faces: 原石的面
    pop_size: 种群大小
    generations: 迭代代数
    crossover_rate: 交叉概率
    mutation_rate: 变异概率
    batch_size: 批处理大小
    early_stop_generations: 如果适应度在这么多代内没有提升，则提前停止
    scale_estimate: 缩放因子的估计上限
    
    返回:
    最佳个体和其适应度值
    """
    # 岛屿模型参数
    num_islands = 5  # 岛屿数量
    migration_interval = 30  # 迁移间隔
    migration_size = int(pop_size * 0.1)  # 迁移个体数量
    
    # 每个岛的种群大小
    island_pop_size = pop_size // num_islands
    
    # 初始化每个岛的种群
    islands = []
    for i in range(num_islands):
        island_pop = initialize_population(island_pop_size, original_vertices, pear_points, scale_estimate)
        islands.append(island_pop)
    
    # 记录全局最佳个体和适应度
    best_individual = None
    best_fitness = 0
    
    # 记录每代的最佳适应度
    best_fitness_history = []
    
    # 每个岛的最佳个体和适应度
    island_best_individuals = [None] * num_islands
    island_best_fitness = [0] * num_islands
    
    # 记录没有改进的代数
    generations_without_improvement = 0
    
    start_time = time.time()
    
    # 遗传算法主循环
    for generation in range(generations):
        # 对每个岛进行演化
        for island_idx in range(num_islands):
            # 当前岛的种群
            population = islands[island_idx]
            
            # 计算适应度
            fitness_values = calculate_fitness_batch(
                population, pear_points, original_vertices, original_faces, batch_size
            )
            
            # 找出当前代的最佳个体
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_individual = population[current_best_idx]
            
            # 更新岛内最佳个体
            if current_best_fitness > island_best_fitness[island_idx]:
                island_best_fitness[island_idx] = current_best_fitness
                island_best_individuals[island_idx] = copy.deepcopy(current_best_individual)
            
            # 更新全局最佳个体
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = copy.deepcopy(current_best_individual)
                generations_without_improvement = 0
                
                # 保存当前最佳个体到文件
                save_best_individual(best_individual, best_fitness)
                
                # 如果找到了比较好的解，也保存一下
                if best_fitness > 0.3:  # 如果体积比超过30%，保存这个解
                    print(f"找到较好的解，体积比为: {best_fitness:.6f}")
                    save_best_individual(best_individual, best_fitness, suffix=f"_vr{best_fitness:.4f}")
            else:
                generations_without_improvement += 1
            
            # 选择
            selected_parents = selection(population, fitness_values, island_pop_size)
            
            # 交叉
            offspring = crossover(selected_parents, crossover_rate)
            
            # 变异 - 加入更多信息以实现自适应变异
            offspring = mutation(
                offspring, 
                mutation_rate, 
                original_vertices, 
                generation, 
                generations, 
                island_best_fitness[island_idx]
            )
            
            # 精英保留策略：保留岛内最佳个体
            if island_best_individuals[island_idx] is not None:
                offspring[0] = island_best_individuals[island_idx]
            
            # 更新岛的种群
            islands[island_idx] = offspring
        
        # 迁移：在岛之间交换个体
        if generation > 0 and generation % migration_interval == 0:
            print(f"代数 {generation + 1}: 岛屿间进行迁移...")
            
            # 各岛最佳个体列表
            all_island_best_fitness = []
            for island_idx in range(num_islands):
                # 计算每个岛的种群适应度
                fitness_values = calculate_fitness_batch(
                    islands[island_idx], pear_points, original_vertices, original_faces, batch_size
                )
                all_island_best_fitness.append((island_idx, np.max(fitness_values)))
            
            # 按适应度排序岛屿
            all_island_best_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # 从最好的岛迁移到其他岛
            best_island_idx = all_island_best_fitness[0][0]
            best_island_pop = islands[best_island_idx]
            
            # 计算最佳岛的适应度
            best_island_fitness = calculate_fitness_batch(
                best_island_pop, pear_points, original_vertices, original_faces, batch_size
            )
            
            # 获取最佳岛中最好的个体
            best_indices = np.argsort(best_island_fitness)[-migration_size:]
            migrants = [copy.deepcopy(best_island_pop[i]) for i in best_indices]
            
            # 将最佳个体迁移到其他岛
            for i, (island_idx, _) in enumerate(all_island_best_fitness[1:]):
                # 计算目标岛的适应度
                target_fitness = calculate_fitness_batch(
                    islands[island_idx], pear_points, original_vertices, original_faces, batch_size
                )
                
                # 替换目标岛中最差的个体
                worst_indices = np.argsort(target_fitness)[:migration_size]
                for j, migrant_idx in enumerate(worst_indices):
                    if j < len(migrants):
                        islands[island_idx][migrant_idx] = migrants[j]
        
        # 每隔一段时间添加新的随机个体以增加多样性
        if generation % 20 == 0 and generation > 0:
            print(f"代数 {generation + 1}: 添加随机个体以增加多样性...")
            for island_idx in range(num_islands):
                random_count = int(island_pop_size * 0.1)  # 10%的随机个体
                random_individuals = initialize_population(random_count, original_vertices, pear_points, scale_estimate)
                
                # 计算当前岛的适应度
                fitness_values = calculate_fitness_batch(
                    islands[island_idx], pear_points, original_vertices, original_faces, batch_size
                )
                
                # 替换掉一些较差的个体
                worst_indices = np.argsort(fitness_values)[:random_count]
                for i, idx in enumerate(worst_indices):
                    if i < len(random_individuals):
                        islands[island_idx][idx] = random_individuals[i]
        
        # 对全局最优解进行局部搜索
        if generation % 25 == 0 and best_individual is not None and best_fitness > 0.1:
            print(f"代数 {generation + 1}: 对全局最优解进行局部搜索...")
            
            # 计算原石的边界盒
            min_bounds = np.min(original_vertices, axis=0)
            max_bounds = np.max(original_vertices, axis=0)
            
            # 创建多个微扰版本的最佳个体
            local_pop_size = 20
            local_population = np.zeros((local_pop_size, len(best_individual)))
            
            for i in range(local_pop_size):
                local_individual = np.copy(best_individual)
                
                # 对平移向量进行小幅度扰动
                local_individual[:3] += np.random.normal(0, 0.05, 3) * (max_bounds - min_bounds)
                
                # 对旋转角度进行小幅度扰动
                local_individual[3:6] += np.random.normal(0, 0.1, 3)
                local_individual[3:6] = local_individual[3:6] % (2 * np.pi)
                
                # 对缩放因子进行小幅度扰动
                local_individual[6] *= (1 + np.random.normal(0, 0.05))
                local_individual[6] = max(0.1, local_individual[6])
                
                local_population[i] = local_individual
            
            # 评估局部种群
            local_fitness = calculate_fitness_batch(
                local_population, pear_points, original_vertices, original_faces, batch_size
            )
            
            # 找出最佳局部解
            local_best_idx = np.argmax(local_fitness)
            local_best_fitness = local_fitness[local_best_idx]
            
            # 如果局部搜索找到更好的解，更新全局最佳解
            if local_best_fitness > best_fitness:
                best_fitness = local_best_fitness
                best_individual = copy.deepcopy(local_population[local_best_idx])
                generations_without_improvement = 0
                
                print(f"局部搜索找到更好的解！体积比: {best_fitness:.6f}")
                save_best_individual(best_individual, best_fitness, suffix="_local")
        
        # 记录每代的最佳适应度
        best_fitness_history.append(best_fitness)
        
        # 输出进度
        if (generation + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"代数 {generation + 1}/{generations}, 最佳适应度: {best_fitness:.6f}, 用时: {elapsed_time:.2f}秒")
            
            # 输出每个岛的最佳适应度
            island_fitness_str = ", ".join([f"岛{i}: {fitness:.6f}" for i, fitness in enumerate(island_best_fitness)])
            print(f"各岛最佳适应度: {island_fitness_str}")
        
        # 提前停止条件
        if generations_without_improvement >= early_stop_generations:
            print(f"已经 {early_stop_generations} 代没有改进，提前停止优化")
            break
    
    total_time = time.time() - start_time
    print(f"优化完成！总用时: {total_time:.2f}秒")
    print(f"最佳体积比: {best_fitness:.6f}")
    
    # 绘制适应度历史
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.title('梨形钻石遗传算法优化过程')
    plt.xlabel('代数')
    plt.ylabel('最佳体积比')
    plt.grid(True)
    plt.savefig('multiple/pear_diamond_fitness_history.png', dpi=300)
    plt.show()
    
    return best_individual, best_fitness

if __name__ == "__main__":
    # 读取原钻石数据
    original_vertices, original_faces = read_original_diamond()
    
    # 读取梨形钻石数据
    pear_points, pear_triangles = read_pear_diamond()
    
    print(f"原石顶点数: {len(original_vertices)}")
    print(f"原石面数: {len(original_faces)}")
    print(f"梨形钻石点数: {len(pear_points)}")
    print(f"梨形钻石面数: {len(pear_triangles)}")
    
    # 可视化原石和梨形钻石的形状
    plot_original_and_pear_diamonds(original_vertices, original_faces, pear_points, pear_triangles)
    
    # 计算梨形钻石的中心和边界框
    pear_center = np.mean(pear_points, axis=0)
    pear_min = np.min(pear_points, axis=0)
    pear_max = np.max(pear_points, axis=0)
    pear_size = pear_max - pear_min
    
    print(f"梨形钻石中心: {pear_center}")
    print(f"梨形钻石尺寸: {pear_size}")
    
    # 计算原石的中心和边界框
    orig_center = np.mean(original_vertices, axis=0)
    orig_min = np.min(original_vertices, axis=0)
    orig_max = np.max(original_vertices, axis=0)
    orig_size = orig_max - orig_min
    
    print(f"原石中心: {orig_center}")
    print(f"原石尺寸: {orig_size}")
    
    # 估算合理的缩放范围
    # 采用更激进的缩放上限，允许算法探索更大的解空间
    scale_estimate = min(orig_size / pear_size) * 2.5  # 进一步提高到2.5倍
    print(f"建议的缩放上限: {scale_estimate}")
    
    # 运行遗传算法
    best_individual, best_fitness = genetic_algorithm(
        pear_points, pear_triangles, original_vertices, original_faces,
        pop_size=1000,  # 增加种群大小以提高搜索能力
        generations=1000,  # 增加迭代次数
        crossover_rate=0.85, 
        mutation_rate=0.4,  # 增加变异率以提高多样性
        batch_size=50,  # 增加批处理大小
        early_stop_generations=150,  # 增加早停阈值
        scale_estimate=scale_estimate * 1.2  # 进一步增大缩放上限
    )
    
    # 检查是否找到有效解
    if best_individual is not None and best_fitness > 0.001:
        # 验证最佳解是否真的在原石内部
        transformed_points = transform_points(pear_points, best_individual[:3], 
                                             best_individual[3:6], best_individual[6])
        orig_hull = ConvexHull(original_vertices)
        all_inside = all(is_inside_convex_hull(p, orig_hull) for p in transformed_points)
        
        if not all_inside:
            print("警告：最佳解中有点超出了原石边界，将缩小缩放因子...")
            # 稍微缩小缩放因子以确保所有点都在内部
            scale_factor = 0.99
            while not all_inside and scale_factor > 0.9:
                best_individual[6] *= scale_factor
                transformed_points = transform_points(pear_points, best_individual[:3], 
                                                     best_individual[3:6], best_individual[6])
                all_inside = all(is_inside_convex_hull(p, orig_hull) for p in transformed_points)
                scale_factor -= 0.01
                
            # 重新计算体积比
            try:
                pear_hull = ConvexHull(transformed_points)
                best_fitness = pear_hull.volume / orig_hull.volume
                print(f"调整后的缩放因子: {best_individual[6]}")
                print(f"调整后的体积比: {best_fitness:.6f}")
            except:
                print("无法计算调整后的体积比")
        
        # 计算最佳个体的几何参数
        parameters = calculate_parameters(best_individual, pear_points)
        
        # 输出结果
        print("\n最佳参数:")
        print(f"平移向量: {best_individual[:3]}")
        print(f"旋转角度: {best_individual[3:6]}")
        print(f"缩放因子: {best_individual[6]}")
        print(f"体积比: {best_fitness}")
        
        print("\n几何参数:")
        for param_name, param_value in parameters.items():
            print(f"{param_name}: {param_value}")
        
        # 绘制结果
        plot_result(original_vertices, original_faces, pear_points, pear_triangles, best_individual, best_fitness, parameters)
    else:
        print("\n未找到有效解。请尝试调整参数或优化算法。") 