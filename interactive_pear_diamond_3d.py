import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# 设置Plotly主题
pio.templates.default = "plotly_white"

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

# 创建交互式3D模型
def create_interactive_3d_model(nodes, triangles):
    # 从节点中提取坐标
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    
    # 创建三角面的索引，以Plotly格式
    i, j, k = [], [], []
    for triangle in triangles:
        i.append(triangle[0])
        j.append(triangle[1])
        k.append(triangle[2])
    
    # 创建一个带有子图的图形
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'mesh3d'}]],
        subplot_titles=["梨形钻石3D交互模型"]
    )
    
    # 创建3D表面网格
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.8,
        colorscale=[[0, 'rgb(200, 230, 255)'], [1, 'rgb(100, 180, 255)']],
        intensity=z,  # 使用z坐标作为颜色强度
        lighting=dict(
            ambient=0.5,    # 环境光
            diffuse=0.8,    # 漫反射
            roughness=0.1,  # 粗糙度
            specular=1.0,   # 镜面反射
            fresnel=1.0,    # 菲涅耳效应
        ),
        flatshading=False, # 平滑着色
        name="梨形钻石",
        showscale=False,
    )
    
    # 添加网格到图形
    fig.add_trace(mesh, row=1, col=1)
    
    # 设置图形布局
    fig.update_layout(
        title=dict(
            text="梨形钻石3D交互模型",
            font=dict(size=24, family="Arial")
        ),
        scene=dict(
            xaxis=dict(title="X (cm)", showbackground=False),
            yaxis=dict(title="Y (cm)", showbackground=False),
            zaxis=dict(title="Z (cm)", showbackground=False),
            aspectmode='data',  # 保持实际比例
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # 设置视角
                up=dict(x=0, y=0, z=1)
            ),
            dragmode='turntable'  # 设置拖动模式为转盘
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgb(250, 250, 255)',  # 浅蓝色背景
    )
    
    # 添加水印信息
    fig.add_annotation(
        text="交互提示: 点击并拖动以旋转，滚轮放大缩小",
        xref="paper", yref="paper",
        x=0.5, y=0.01,
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig

# 创建交互式3D模型的第二个变种（透明效果）
def create_alternative_3d_model(nodes, triangles):
    # 从节点中提取坐标
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    
    # 创建主图形
    fig = go.Figure()
    
    # 添加点云作为顶点
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,  # 使用z坐标作为颜色
            colorscale='Blues',
            opacity=0.7
        ),
        name="顶点"
    ))
    
    # 创建三角面的索引，以Plotly格式
    i, j, k = [], [], []
    for triangle in triangles:
        i.append(triangle[0])
        j.append(triangle[1])
        k.append(triangle[2])
    
    # 添加透明的3D表面网格
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='rgba(100, 200, 255, 0.3)',  # 半透明蓝色
        opacity=0.6,
        lighting=dict(
            ambient=0.7,
            diffuse=0.5,
            roughness=0.2,
            specular=0.8,
            fresnel=0.8,
        ),
        flatshading=True,
        name="钻石表面"
    ))
    
    # 设置图形布局
    fig.update_layout(
        title="梨形钻石透明3D模型",
        scene=dict(
            xaxis=dict(title="X (cm)"),
            yaxis=dict(title="Y (cm)"),
            zaxis=dict(title="Z (cm)"),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
            )
        ),
        width=900,
        height=700,
    )
    
    return fig

# 主函数
def main():
    # CSV文件路径
    file_path = 'data/attachment_3_standarded_pear_diamond_geometry_data_file.csv'
    print("正在读取数据文件...")
    
    # 读取数据
    nodes, triangles = read_diamond_data(file_path)
    print(f"已读取 {len(nodes)} 个节点和 {len(triangles)} 个三角面")
    
    # 创建交互式3D模型
    print("正在生成交互式梨形钻石3D模型...")
    
    # 创建主模型
    fig = create_interactive_3d_model(nodes, triangles)
    
    # 保存为HTML文件，可以在浏览器中打开
    fig.write_html("html/pear_diamond_interactive.html")
    print("已保存交互式3D模型到 pear_diamond_interactive.html")
    
    # 创建替代模型（透明效果）
    fig_alt = create_alternative_3d_model(nodes, triangles)
    fig_alt.write_html("html/pear_diamond_transparent.html")
    print("已保存透明效果3D模型到 pear_diamond_transparent.html")
    
    # 显示模型（在线打开图形）
    fig.show()
    
    print("提示: 您可以在浏览器中打开保存的HTML文件，获得完整的交互体验")

if __name__ == "__main__":
    main() 