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

# 设置随机种子以保证结果可重复性
np.random.seed(42)
random.seed(42)

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

def read_standard_diamond():
    """读取标准圆形钻石数据"""
    # 使用与diamond_visualization.py相同的读取方法
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

def plot_original_and_standard_diamonds(original_vertices, original_faces, standard_points):
    """
    绘制原石和标准钻石的形状，帮助直观理解数据
    
    参数:
    original_vertices: 原石的顶点
    original_faces: 原石的面
    standard_points: 标准钻石的点云
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
    
    # 绘制标准钻石
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(standard_points[:, 0], standard_points[:, 1], standard_points[:, 2], 
               c='red', s=10)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('标准钻石点云')
    
    # 计算标准钻石的范围以设置坐标轴
    std_min_bounds = np.min(standard_points, axis=0)
    std_max_bounds = np.max(standard_points, axis=0)
    
    # 设置坐标轴范围
    ax2.set_xlim(std_min_bounds[0], std_max_bounds[0])
    ax2.set_ylim(std_min_bounds[1], std_max_bounds[1])
    ax2.set_zlim(std_min_bounds[2], std_max_bounds[2])
    
    plt.tight_layout()
    plt.savefig('figure/original_and_standard_diamonds.png', dpi=300)
    plt.show()

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

def calculate_volume_ratio(standard_points, original_vertices, original_faces, individual):
    """
    计算标准钻石和原石的体积比
    
    参数:
    standard_points: 标准钻石的点云
    original_vertices: 原石的顶点
    original_faces: 原石的面
    individual: 包含平移、旋转和缩放参数的个体
    
    返回:
    体积比，如果标准钻石不完全在原石内部，则返回低适应度
    """
    # 解析个体参数
    translation = individual[:3]
    rotation = individual[3:6]
    scale = individual[6]
    
    # 变换标准钻石点云
    transformed_points = transform_points(standard_points, translation, rotation, scale)
    
    # 计算原石的凸包
    try:
        orig_hull = ConvexHull(original_vertices)
        orig_volume = orig_hull.volume
    except:
        return 0.0
    
    # 计算标准钻石的凸包
    try:
        std_hull = ConvexHull(transformed_points)
        std_volume = std_hull.volume
    except:
        return 0.0
    
    # 检查标准钻石是否在原石内部
    # 检查所有点而不仅仅是采样点，确保完全在内部
    inside_count = 0
    for point in transformed_points:
        if is_inside_convex_hull(point, orig_hull, tolerance=1e-6):
            inside_count += 1
    
    # 计算内部点比例
    inside_ratio = inside_count / len(transformed_points)
    
    # 如果所有点都在内部，则计算实际体积比
    if inside_ratio == 1.0:
        volume_ratio = std_volume / orig_volume
        return volume_ratio
    else:
        # 返回一个很小但非零的适应度值，用内部点比例来区分
        return inside_ratio * 0.001  # 按照内部点比例给予小的适应度值

def initialize_population(pop_size, original_vertices, standard_points, scale_estimate=None):
    """
    初始化种群
    
    每个个体包含:
    - 平移向量 (3)
    - 旋转角度 (3)
    - 缩放因子 (1)
    
    参数:
    pop_size: 种群大小
    original_vertices: 原石的顶点
    standard_points: 标准钻石的点云
    scale_estimate: 估算的合理缩放上限，如果为None则自动计算
    
    返回形状为 (pop_size, 7) 的数组
    """
    # 计算原石的边界盒以确定平移范围
    orig_min_bounds = np.min(original_vertices, axis=0)
    orig_max_bounds = np.max(original_vertices, axis=0)
    orig_center = (orig_min_bounds + orig_max_bounds) / 2
    orig_extent = orig_max_bounds - orig_min_bounds
    
    # 计算标准钻石的边界盒
    std_min_bounds = np.min(standard_points, axis=0)
    std_max_bounds = np.max(standard_points, axis=0)
    std_center = (std_min_bounds + std_max_bounds) / 2
    std_extent = std_max_bounds - std_min_bounds
    
    # 计算缩放范围
    if scale_estimate is None:
        min_scales = orig_extent / std_extent
        scale_estimate = min(min_scales) * 1.5  # 进一步增加到1.5以允许更大的缩放
    
    print(f"使用缩放上限: {scale_estimate}")
    
    population = []
    
    # 第一个个体：使用最佳估计参数（钻石中心对齐，标准旋转，合理缩放）
    initial_best = np.zeros(7)
    initial_best[:3] = orig_center - std_center * scale_estimate * 0.8  # 平移以对齐中心
    initial_best[3:6] = [0, 0, 0]  # 不旋转
    initial_best[6] = scale_estimate * 0.8  # 设置为估计最大缩放的80%
    population.append(initial_best)
    
    # 创建不同缩放比例的初始个体
    scaling_factors = [0.7, 0.9, 1.0, 1.1]
    for factor in scaling_factors:
        # 创建不同缩放的个体
        scaled_individual = np.copy(initial_best)
        scaled_individual[6] = scale_estimate * factor  # 使用不同的缩放因子
        population.append(scaled_individual)
    
    # 创建不同旋转的个体
    rotation_angles = [
        [np.pi/4, 0, 0],  # 绕x轴旋转45度
        [0, np.pi/4, 0],  # 绕y轴旋转45度
        [0, 0, np.pi/4],  # 绕z轴旋转45度
        [np.pi/4, np.pi/4, 0],  # 绕x和y轴旋转45度
        [np.pi/4, 0, np.pi/4],  # 绕x和z轴旋转45度
        [0, np.pi/4, np.pi/4],  # 绕y和z轴旋转45度
        [np.pi/4, np.pi/4, np.pi/4]  # 绕所有轴旋转45度
    ]
    
    for angles in rotation_angles:
        # 创建不同旋转的个体
        rotated_individual = np.copy(initial_best)
        rotated_individual[3:6] = angles
        population.append(rotated_individual)
    
    # 创建不同平移的个体
    translation_offsets = [
        [0.1, 0, 0], [-0.1, 0, 0],
        [0, 0.1, 0], [0, -0.1, 0],
        [0, 0, 0.1], [0, 0, -0.1]
    ]
    
    for offset in translation_offsets:
        # 创建不同平移的个体
        translated_individual = np.copy(initial_best)
        translated_individual[:3] += np.array(offset) * orig_extent
        population.append(translated_individual)
    
    # 其余个体：添加随机变化
    remaining_slots = pop_size - len(population)
    for i in range(remaining_slots):
        if i < remaining_slots * 0.3:  # 30%的个体基于最佳估计参数，但有小的随机变化
            individual = np.copy(initial_best)
            # 小范围随机平移
            individual[:3] += np.random.uniform(-0.2, 0.2, 3) * orig_extent
            # 小范围随机旋转
            individual[3:6] += np.random.uniform(-0.5, 0.5, 3)
            # 小范围随机缩放
            individual[6] *= np.random.uniform(0.8, 1.2)
        else:  # 其余个体完全随机
            # 随机平移在原石范围内
            translation = orig_center + np.random.uniform(-0.5, 0.5, 3) * orig_extent * 0.5
            # 随机旋转
            rotation = np.random.uniform(0, 2 * np.pi, 3)
            # 随机缩放，但在合理范围内
            scale = np.random.uniform(0.5, scale_estimate)
            
            individual = np.concatenate([translation, rotation, [scale]])
        
        population.append(individual)
    
    return np.array(population[:pop_size])  # 确保正好返回pop_size个个体

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
    使用均匀交叉进行染色体交换
    
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
            # 均匀交叉
            mask = np.random.random(len(parent1)) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    
    return np.array(offspring)

def mutation(offspring, mutation_rate, original_vertices, generation=0, max_generations=100):
    """
    对后代进行变异
    
    参数:
    offspring: 后代个体
    mutation_rate: 变异概率
    original_vertices: 原石顶点，用于确定合理的平移范围
    generation: 当前代数
    max_generations: 最大代数
    
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
    translation_range = 0.5 * (1 - progress) + 0.1  # 从0.5逐渐减小到0.1
    rotation_range = np.pi * (1 - progress) + 0.1  # 从π逐渐减小到0.1
    scale_range = 0.3 * (1 - progress) + 0.05  # 从0.3逐渐减小到0.05
    
    for i in range(len(offspring)):
        # 个体变异概率
        if np.random.random() < mutation_rate:
            # 参数变异概率，每个参数都有独立的变异机会
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
    
    return offspring

def calculate_parameters(best_individual, standard_points):
    """
    计算最佳个体对应的几何参数
    
    参数:
    best_individual: 包含最佳平移、旋转和缩放参数的个体
    standard_points: 标准钻石的点云
    
    返回:
    包含所需几何参数的字典
    """
    # 解析个体参数
    translation = best_individual[:3]
    rotation = best_individual[3:6]
    scale = best_individual[6]
    
    # 变换标准钻石点云
    transformed_points = transform_points(standard_points, translation, rotation, scale)
    
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
    save_parameters(parameters)
    
    return parameters

def save_parameters(parameters):
    """
    保存几何参数到文件
    
    参数:
    parameters: 包含几何参数的字典
    """
    with open('figure/geometric_parameters.txt', 'w') as f:
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

def plot_result(original_vertices, original_faces, standard_points, best_individual, parameters=None):
    """
    绘制最佳结果
    
    参数:
    original_vertices: 原石的顶点
    original_faces: 原石的面
    standard_points: 标准钻石的点云
    best_individual: 包含最佳平移、旋转和缩放参数的个体
    parameters: 几何参数字典
    """
    # 解析个体参数
    translation = best_individual[:3]
    rotation = best_individual[3:6]
    scale = best_individual[6]
    
    # 变换标准钻石点云
    transformed_points = transform_points(standard_points, translation, rotation, scale)
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始钻石
    for face in original_faces:
        verts = [original_vertices[idx] for idx in face]
        tri = Poly3DCollection([verts], alpha=0.2)
        tri.set_color('lightblue')
        ax.add_collection3d(tri)
    
    # 绘制变换后的标准钻石点云 - 更美观的表示方式
    # 首先绘制点
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], 
               c='red', s=10, label='标准钻石')
    
    # 然后尝试构建和绘制钻石的表面
    try:
        # 创建凸包
        hull = ConvexHull(transformed_points)
        
        # 为每个面创建一个多边形
        for simplex in hull.simplices:
            verts = transformed_points[simplex]
            # 创建一个三角形面
            tri = Poly3DCollection([verts], alpha=0.3)
            tri.set_color('red')
            ax.add_collection3d(tri)
    except:
        # 如果凸包创建失败，使用原始点云
        print("无法创建钻石表面，使用原始点云显示")
    
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
    plt.title('优化后的标准钻石在原石中的位置')
    
    # 添加图例
    ax.legend()
    
    # 在图上添加参数信息
    if parameters:
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
    plt.savefig('figure/optimized_diamond_position.png', dpi=300)
    
    # 显示图形
    plt.show()

def calculate_fitness_batch(population, standard_points, original_vertices, original_faces, batch_size=10):
    """
    批量计算适应度，可以充分利用GPU并行计算能力
    
    参数:
    population: 种群
    standard_points: 标准钻石的点云
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
                    standard_points, original_vertices, original_faces, batch[idx]
                ),
                range(len(batch))
            ))
        
        # 存储结果
        fitness_values[i:i+len(batch)] = batch_fitness
    
    return fitness_values

def genetic_algorithm(standard_points, original_vertices, original_faces, 
                     pop_size=100, generations=100, 
                     crossover_rate=0.8, mutation_rate=0.1,
                     batch_size=10, early_stop_generations=50,
                     scale_estimate=None):
    """
    使用遗传算法寻找最优解
    
    参数:
    standard_points: 标准钻石的点云
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
    # 初始化种群
    population = initialize_population(pop_size, original_vertices, standard_points, scale_estimate)
    
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
            population, standard_points, original_vertices, original_faces, batch_size
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
            save_best_individual(best_individual, best_fitness)
            
            # 如果找到了比较好的解，也保存一下
            if best_fitness > 0.3:  # 如果体积比超过30%，保存这个解
                print(f"找到较好的解，体积比为: {best_fitness:.6f}")
                save_best_individual(best_individual, best_fitness, suffix=f"_vr{best_fitness:.4f}")
        else:
            generations_without_improvement += 1
        
        best_fitness_history.append(best_fitness)
        
        # 输出进度
        if (generation + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"代数 {generation + 1}/{generations}, 最佳适应度: {best_fitness:.6f}, 用时: {elapsed_time:.2f}秒")
        
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
        
        # 添加一些随机个体以增加多样性
        if generation % 20 == 0 and generation > 0:
            print("添加随机个体以增加多样性...")
            random_count = int(pop_size * 0.1)  # 10%的随机个体
            random_individuals = initialize_population(random_count, original_vertices, standard_points, scale_estimate)
            # 替换掉一些较差的个体
            worst_indices = np.argsort(fitness_values)[:random_count]
            for i, idx in enumerate(worst_indices):
                if i < len(random_individuals):
                    offspring[idx] = random_individuals[i]
        
        # 更新种群
        population = offspring
    
    total_time = time.time() - start_time
    print(f"优化完成！总用时: {total_time:.2f}秒")
    print(f"最佳体积比: {best_fitness:.6f}")
    
    # 绘制适应度历史
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.title('遗传算法优化过程')
    plt.xlabel('代数')
    plt.ylabel('最佳体积比')
    plt.grid(True)
    plt.savefig('figure/optimization_history.png', dpi=300)
    plt.show()
    
    return best_individual, best_fitness

def save_best_individual(individual, fitness, suffix=""):
    """
    保存最佳个体到文件
    
    参数:
    individual: 个体参数数组
    fitness: 适应度值
    suffix: 文件名后缀，用于保存多个解
    """
    filename = f"figure/best_individual{suffix}.txt"
    with open(filename, 'w') as f:
        f.write(f"适应度 (体积比): {fitness}\n")
        f.write(f"平移向量: {individual[:3]}\n")
        f.write(f"旋转角度: {individual[3:6]}\n")
        f.write(f"缩放因子: {individual[6]}\n")
        f.write(f"原始数组: {individual}\n")

if __name__ == "__main__":
    # 读取原钻石数据
    original_vertices, original_faces = read_original_diamond()
    
    # 读取标准圆形钻石数据
    standard_points, _ = read_standard_diamond()
    
    print(f"原石顶点数: {len(original_vertices)}")
    print(f"原石面数: {len(original_faces)}")
    print(f"标准钻石点数: {len(standard_points)}")
    
    # 可视化原石和标准钻石的形状
    plot_original_and_standard_diamonds(original_vertices, original_faces, standard_points)
    
    # 计算标准钻石的中心和边界框
    std_center = np.mean(standard_points, axis=0)
    std_min = np.min(standard_points, axis=0)
    std_max = np.max(standard_points, axis=0)
    std_size = std_max - std_min
    
    print(f"标准钻石中心: {std_center}")
    print(f"标准钻石尺寸: {std_size}")
    
    # 计算原石的中心和边界框
    orig_center = np.mean(original_vertices, axis=0)
    orig_min = np.min(original_vertices, axis=0)
    orig_max = np.max(original_vertices, axis=0)
    orig_size = orig_max - orig_min
    
    print(f"原石中心: {orig_center}")
    print(f"原石尺寸: {orig_size}")
    
    # 估算合理的缩放范围 - 根据论文，将上限调整到能获得约0.4体积比的值
    # 采用更激进的缩放上限，允许算法探索更大的解空间
    scale_estimate = min(orig_size / std_size) * 2.5  # 进一步提高到2.5倍
    print(f"建议的缩放上限: {scale_estimate}")
    
    # 运行遗传算法
    best_individual, best_fitness = genetic_algorithm(
        standard_points, original_vertices, original_faces,
        pop_size=500,  # 进一步增加种群大小以提高搜索能力
        generations=500,  # 增加迭代次数
        crossover_rate=0.8, 
        mutation_rate=0.3,
        batch_size=20, 
        early_stop_generations=100,
        scale_estimate=scale_estimate
    )
    
    # 检查是否找到有效解
    if best_individual is not None and best_fitness > 0.001:
        # 验证最佳解是否真的在原石内部
        transformed_points = transform_points(standard_points, best_individual[:3], 
                                             best_individual[3:6], best_individual[6])
        orig_hull = ConvexHull(original_vertices)
        all_inside = all(is_inside_convex_hull(p, orig_hull) for p in transformed_points)
        
        if not all_inside:
            print("警告：最佳解中有点超出了原石边界，将缩小缩放因子...")
            # 稍微缩小缩放因子以确保所有点都在内部
            scale_factor = 0.99
            while not all_inside and scale_factor > 0.9:
                best_individual[6] *= scale_factor
                transformed_points = transform_points(standard_points, best_individual[:3], 
                                                     best_individual[3:6], best_individual[6])
                all_inside = all(is_inside_convex_hull(p, orig_hull) for p in transformed_points)
                scale_factor -= 0.01
                
            # 重新计算体积比
            try:
                std_hull = ConvexHull(transformed_points)
                best_fitness = std_hull.volume / orig_hull.volume
                print(f"调整后的缩放因子: {best_individual[6]}")
                print(f"调整后的体积比: {best_fitness:.6f}")
            except:
                print("无法计算调整后的体积比")
        
        # 计算最佳个体的几何参数
        parameters = calculate_parameters(best_individual, standard_points)
        
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
        plot_result(original_vertices, original_faces, standard_points, best_individual, parameters)
    else:
        print("\n未找到有效解。请尝试调整参数或优化算法。") 