import numpy as np
import argparse

def load_and_combine_npy(files, rates, output_file):
    """
    加载多个.npy文件，按照给定的比例融合为一个.npy文件。

    参数:
    files (list): 包含.npy文件路径的列表。
    rates (list): 每个文件对应的权重比例列表。
    output_file (str): 输出文件的路径。
    """
    # 初始化最终数组
    ###############################################################################
    final_array = np.zeros((4307, 155)) 
    ###############################################################################

    # 遍历文件和对应的权重
    for file, rate in zip(files, rates):
        # 加载.npy文件
        data = np.load(file)
        # 按照权重融合数据
        final_array += rate * data

    # 保存融合后的数组到新的.npy文件
    np.save(output_file, final_array)
    print(f"Combined array saved to {output_file}")

    return output_file

def calculate_accuracy(label_path, pred_path):
    # 加载标签和预测
    label = np.load(label_path)
    pred = np.load(pred_path).argmax(axis=1)

    # 计算正确预测的数量
    correct = (pred == label).sum()
    # 计算总样本数
    total = len(label)

    # 打印准确度
    print('Top1 Acc: {:.5f}%'.format(correct / total * 100))

# 设置命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='ICMEW2024-Track10-main/combined.npy')

# 示例用法
files = [
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_B\B_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_B_3D\B_3D_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_BM\BM_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_BM_3D\BM_3D_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_J\J_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_J_3D\J_3D_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_JM\JM_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_JM_3D\JM_3D_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_B\B_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_BM\BM_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_J\J_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_JM\JM_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\tdgcn_V2_B\B_b.npy',
    # r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_BM\BM.npy',
    # r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_J\J.npy',
    # r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_JM\JM.npy'
    r'TE-GCN-main\work_dir\2996\epoch1_test_score.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_B\B_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2\K2_b.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2M\K2M_b.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_BM\BM.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_JM\V2_JM_b.npy',
]
rates = [ 0.64740818,  0.82908343,  0.52259517, -0.02600348,
          0.29760336,  0.41879526,  -0.75960172,  0.2362587,
          -0.13957313, -0.42938177, -0.05887759, -0.13832274,
          1.65843405,  
          2.10272236, -0.07652921,  0.86116095, -0.13640361,
          0.13170087]
# rates = [0.75, 0.95, 0.62,
#         0.68, 0.23,
#         #  0.01, 0.01, 0.0001, 0.0001,
#         #  0.001, 0.001, 0.0001, 0.0001]
#         0.99]
output_file = 'ICMEW2024-Track10-main/combined.npy'

if __name__ == "__main__":
    args = parser.parse_args()

    # 首先，加载并合并.npy文件
    combined_file = load_and_combine_npy(files, rates, output_file)

    # 然后，计算准确度
    label_path = r'data_C\test_label.npy'  # 假设标签文件的路径
    calculate_accuracy(label_path, combined_file)