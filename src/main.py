import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_loader import DataLoader
from geometry.diamond_model import DiamondModel
from optimization.cutting_optimizer import CuttingOptimizer

def main():
    # 1. 加载数据
    print("正在加载数据...")
    loader = DataLoader()
    raw_data = loader.process_raw_diamond_data()
    
    # 2. 创建原始钻石模型
    print("正在创建原始钻石模型...")
    diamond_model = DiamondModel()
    raw_diamond = diamond_model.create_raw_diamond_model(raw_data)
    
    # 3. 优化切割方案
    print("正在优化切割方案...")
    optimizer = CuttingOptimizer(diamond_model)
    cutting_plan = optimizer.generate_cutting_plan()
    
    # 4. 输出结果
    print("\n切割方案结果：")
    print(f"标准圆形钻石参数：{cutting_plan['parameters'][:3]}")
    print(f"梨型钻石参数：{cutting_plan['parameters'][3:]}")
    print(f"方案价值：{cutting_plan['value']}")
    
    # 5. 可视化结果
    print("\n正在生成可视化结果...")
    cutting_plan['standard_diamond'].visualize()
    cutting_plan['pear_diamond'].visualize()

if __name__ == "__main__":
    main() 