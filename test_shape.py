import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_image_3(file_path, sample_index=0):
    # 检查文件路径是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    # 从 .npy 文件中加载数据
    try:
        data = np.load(file_path)
    except Exception as e:
        raise ValueError(f"无法加载文件 {file_path}，错误信息：{e}")
    
    # 确保数据是 NumPy 数组
    if not isinstance(data, np.ndarray):
        raise ValueError("数据必须是 NumPy 数组")
    
    # 检查样本索引是否在有效范围内
    if sample_index < 0 or sample_index >= data.shape[0]:
        raise IndexError(f"样本索引 {sample_index} 超出范围")
    
    # 获取指定样本的帧数据
    frame_data = data[sample_index, :, :, :, 0]
    
    # 检查 frame_data 的形状是否正确
    if frame_data.shape[0] != 3 or frame_data.shape[1] % 10 != 0:
        raise ValueError("frame_data 的形状不正确")
    
    # 创建一个 3x3 的子图网格，每个子图都是 3D 投影
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), subplot_kw=dict(projection='3d'))
    
    # 定义颜色
    colors = ['r', 'b']
    
    # 遍历每个子图
    for i in range(3):
        m = 3 * i
        for j in range(3):
            # 获取 x, y, z 坐标
            x_coords = frame_data[0, (m + j) * 10, :]
            y_coords = frame_data[1, (m + j) * 10, :]
            z_coords = frame_data[2, (m + j) * 10, :]
            
            # 绘制 3D 散点图
            axs[i][j].scatter3D(x_coords, y_coords, z_coords, color=colors[0])
            
            # 设置子图的标题
            axs[i][j].set_title(f"Subplot {i+1}-{j+1}")
            
            # 设置坐标轴标签
            axs[i][j].set_xlabel('X')
            axs[i][j].set_ylabel('Y')
            axs[i][j].set_zlabel('Z')
    
    # 显示图形
    plt.show()

# 调用函数，传入 .npy 文件路径和样本索引
plot_image_3(r"data\train_joint.npy", sample_index=0)

