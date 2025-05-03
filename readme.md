# 梨形钻石最优放置算法

## 项目介绍

本项目使用遗传算法寻找将标准梨形钻石放入原石模型中的最佳位置、旋转角度和缩放比例，使得体积比最大化。程序会计算最优情况下的几何参数，并生成3D可视化结果。

## 功能特性

1. 读取原石和梨形钻石的几何数据
2. 使用遗传算法寻找最佳放置方案（旋转、平移和等比缩放）
3. 利用GPU加速计算（如果可用）
4. 计算最优情况下的几何参数：
   - a: 主半轴长度
   - b: 次半轴长度
   - e: 离心率
   - D: 腰围高度
   - Lp: 下锥高度
   - mp: 下锥参数
   - Lc: 上锥高度
   - mc: 上锥参数
   - bc: 基础半长轴与上锥的夹角
   - bp: 基础半长轴与下锥的夹角
5. 使用Plotly生成交互式3D可视化HTML文件

## 环境需求

- Python 3.7+
- 依赖包：
  - numpy
  - pandas
  - scipy
  - deap (用于遗传算法)
  - plotly (用于3D可视化)
  - torch (用于GPU加速)

## 安装依赖

```bash
pip install numpy pandas scipy deap plotly torch
```

## 使用方法

1. 确保数据文件存放在`data`目录下
2. 运行主程序：

```bash
python diamond_optimization.py
```

3. 程序会输出优化过程、最佳参数和几何参数
4. 结果会保存为HTML文件和CSV文件

## 数据文件说明

- `attachment_1_diamond_original_geometry_data_file.csv`: 原石几何数据
- `attachment_3_standarded_pear_diamond_geometry_data_file.csv`: 标准梨形钻石几何数据

## 输出结果

1. 控制台输出：
   - 优化过程中的进度
   - 最佳放置参数和体积比
   - 计算得到的各项几何参数
  
2. 文件输出：
   - `diamond_optimization_result_YYYYMMDD_HHMMSS.html`: 3D可视化结果
   - `diamond_parameters_YYYYMMDD_HHMMSS.csv`: 几何参数数据

## 算法说明

1. 遗传算法使用参数：
   - 种群大小：200
   - 杂交概率：0.8
   - 变异概率：0.24
   - 迭代次数：200

2. 优化参数：
   - 缩放比例
   - 三维旋转角度 (X, Y, Z轴)
   - 三维平移距离 (X, Y, Z轴)

3. 适应度函数定义为梨形钻石体积与原石体积的比值，当且仅当梨形钻石完全在原石内部时有效。
##核心代码文件

   -diamond_visualization.py 标准圆形钻石的显示文件
   
   -visualize_diamond.py 原石的显示文件
   
   -enhanced_pear_diamond_3d.py 3d梨形钻石显示文件
   
   -genetic_optimization.py 标准圆形钻石利用遗传算法进行优化的文件
   
   -pear_genetic_optimization.py 标准梨形钻石利用遗传算法进行优化的文件
   
   -optimizer.py 两个钻石利用遗传算法进行优化的文件
