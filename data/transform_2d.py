import numpy as np

def convert_npy_to_npz(train_file, label_file, output_file):
    # 加载训练数据.npy文件
    x_data = np.load(train_file)
    
    # 加载标签数据.npy文件
    y_data = np.load(label_file)
    
    # 检查数据维度是否符合预期
    if x_data.ndim != 5:
        raise ValueError("训练数据必须具有5个维度: N C T V M")
    
    # 重新排列维度从 N C T V M 到 N T M V C
    x_data = x_data.transpose((0, 2, 4, 3, 1))
    
    # 保留前两个通道（x和y坐标）
    x_data = x_data[:, :, :, :, :2]
    print(x_data.shape)
    # 保存为.npz文件
    np.savez(output_file, x_test=x_data, y_test=y_data) # 注意对于train和test，这里需要修改

# 使用示例
train_npy = r'data_C\test_joint.npy'
label_npy = r'data_C\test_label.npy'
output_npz = r'data_C\test_2d.npz'
convert_npy_to_npz(train_npy, label_npy, output_npz)

# N: 样本数量 (Number of samples)
# C: 二维空间中的坐标通道 (Channels for XY coordinates)
# T: 帧数 (Time steps or frames)
# V: 节点 (Vertices)
# M: 人数 (Number of people)
