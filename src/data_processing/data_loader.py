import pandas as pd
from docx import Document
import numpy as np
import os
from typing import Dict, List, Tuple, Union

class DataLoader:
    def __init__(self):
        """初始化数据加载器"""
        # 获取项目根目录的绝对路径
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, 'data')
        print(f"数据目录: {self.data_dir}")
        
    def load_excel_data(self, filename: str) -> pd.DataFrame:
        """加载Excel文件数据
        
        Args:
            filename: Excel文件名
            
        Returns:
            pandas DataFrame对象
        """
        file_path = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_excel(file_path)
            print(f"成功加载Excel文件: {filename}")
            print(f"列名: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"加载Excel文件 {filename} 时出错: {str(e)}")
            return pd.DataFrame()
    
    def load_docx_data(self, filename: str) -> Document:
        """加载Word文档数据
        
        Args:
            filename: Word文档文件名
            
        Returns:
            python-docx Document对象
        """
        file_path = os.path.join(self.data_dir, filename)
        try:
            doc = Document(file_path)
            print(f"成功加载Word文档: {filename}")
            return doc
        except Exception as e:
            print(f"加载Word文档 {filename} 时出错: {str(e)}")
            return None
    
    def process_raw_diamond_data(self) -> Dict[str, np.ndarray]:
        """处理原始钻石数据
        
        Returns:
            包含顶点坐标和面信息的字典
        """
        raw_data = self.load_excel_data('附件1：钻石原石几何模型.xlsx')
        
        if raw_data.empty:
            return {'vertices': np.array([]), 'faces': np.array([])}
            
        # 打印数据信息以便调试
        print("\n原始钻石数据信息:")
        print(f"数据形状: {raw_data.shape}")
        print(f"列名: {raw_data.columns.tolist()}")
        
        # 提取顶点坐标和面信息
        # 注意：这里需要根据实际Excel文件的列名进行调整
        try:
            vertices = raw_data[['X坐标', 'Y坐标', 'Z坐标']].values
            faces = raw_data[['面1', '面2', '面3']].values
        except KeyError as e:
            print(f"警告：未找到预期的列名 - {str(e)}")
            print("可用的列名:", raw_data.columns.tolist())
            vertices = np.array([])
            faces = np.array([])
        
        return {
            'vertices': vertices,
            'faces': faces
        }
    
    def process_standard_diamond_data(self) -> Dict[str, float]:
        """处理标准圆形钻石数据
        
        Returns:
            包含标准圆形钻石参数的字典
        """
        doc = self.load_docx_data('附件2：标准圆形钻石几何数据.docx')
        
        if doc is None:
            return {}
            
        # 初始化参数字典
        parameters = {
            'table_size': 0.0,
            'crown_height': 0.0,
            'pavilion_depth': 0.0,
            'girdle_thickness': 0.0,
            'culet_size': 0.0
        }
        
        # 从文档中提取参数
        for paragraph in doc.paragraphs:
            text = paragraph.text.lower()
            print(f"处理文本: {text}")  # 调试信息
            for param in parameters.keys():
                if param in text:
                    # 尝试提取数值
                    try:
                        value = float(text.split(':')[-1].strip())
                        parameters[param] = value
                    except:
                        continue
        
        return parameters
    
    def process_pear_diamond_data(self) -> Dict[str, Union[np.ndarray, float]]:
        """处理梨型钻石数据
        
        Returns:
            包含梨型钻石参数的字典
        """
        pear_data = self.load_excel_data('附件3：梨型钻石几何模型.xlsx')
        
        if pear_data.empty:
            return {}
            
        # 打印数据信息以便调试
        print("\n梨型钻石数据信息:")
        print(f"数据形状: {pear_data.shape}")
        print(f"列名: {pear_data.columns.tolist()}")
        
        # 提取关键参数
        parameters = {
            'length_to_width_ratio': 0.0,
            'crown_angle': 0.0,
            'pavilion_angle': 0.0,
            'girdle_thickness': 0.0
        }
        
        # 从数据中提取参数
        for param in parameters.keys():
            if param in pear_data.columns:
                parameters[param] = pear_data[param].iloc[0]
            else:
                print(f"警告：未找到参数 {param}")
        
        return parameters

if __name__ == "__main__":
    # 测试数据加载
    loader = DataLoader()
    
    # 测试原始钻石数据加载
    raw_data = loader.process_raw_diamond_data()
    print("\n原始钻石数据:")
    print(f"顶点数量: {len(raw_data['vertices'])}")
    print(f"面数量: {len(raw_data['faces'])}")
    
    # 测试标准圆形钻石数据加载
    standard_data = loader.process_standard_diamond_data()
    print("\n标准圆形钻石参数:")
    for param, value in standard_data.items():
        print(f"{param}: {value}")
    
    # 测试梨型钻石数据加载
    pear_data = loader.process_pear_diamond_data()
    print("\n梨型钻石参数:")
    for param, value in pear_data.items():
        print(f"{param}: {value}") 