import numpy as np

def load_and_combine_npy(files, rates, output_file):
    """
    加载多个.npy文件，按照给定的比例融合为一个.npy文件。

    参数:
    files (list): 包含.npy文件路径的列表。
    rates (list): 每个文件对应的权重比例列表。
    output_file (str): 输出文件的路径。
    """
    # 初始化最终数组
    final_array = np.zeros((2000, 155))

    # 遍历文件和对应的权重
    for file, rate in zip(files, rates):
        # 加载.npy文件
        data = np.load(file)
        # 按照权重融合数据
        final_array += rate * data

    # 保存融合后的数组到新的.npy文件
    np.save(output_file, final_array)
    print(f"Combined array saved to {output_file}")

# 示例用法
files = [
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_B_3D\B_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_BM\BM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_BM_3D\BM_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_J_3D\J_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_JM\JM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\ctrgcn_V2_JM_3D\JM_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\mstgcn_V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\mstgcn_V2_BM\BM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\mstgcn_V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\mstgcn_V2_JM\JM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_BM\BM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_JM\JM.npy'
]
rates = [0.7, 0.1, 0.28, 0.29,
         0.65, 0.3, 0.3, 0.3,
         0.05, 0.05, 0.05, 0.05,
         0.7, 0.3, 0.7, 0.3]  # 根据需要调整权重比例
output_file = 'ICMEW2024-Track10-main/combined.npy'

load_and_combine_npy(files, rates, output_file)