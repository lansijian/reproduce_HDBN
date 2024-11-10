import numpy as np
import argparse
from scipy.optimize import minimize
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
    return correct / total * 100

# 定义目标函数
def f1(rates):
    label_path = r'data_C\val_label.npy'  # 假设标签文件的路径
    combined_file = load_and_combine_npy(files, rates, output_file) #加载并合并置信度文件
    return -calculate_accuracy(label_path, combined_file)

# 设置命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='ICMEW2024-Track10-main/combined.npy')

# 示例用法
files = [
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_B_3D\B_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_BM\BM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_BM_3D\BM_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_J_3D\J_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_JM\JM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_JM_3D\JM_3D.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_BM\BM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_JM\JM.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\tdgcn_V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\tdgcn_V2_BM\BM.npy',
    # r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_J\J.npy',
    # r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_JM\JM.npy'
    r'TE-GCN-main\work_dir\2995\epoch1_test_score.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2\K2.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2M\K2M.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_BM\BM.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_JM\V2_JM.npy',
]
rates = [-1.14570081,  0.73093037,  0.50504466,  0.10773738,
          0.61150915,  0.53455875, -0.56590946,  0.46124499,
         -0.06959502,  0.06440321, -0.38478471, -0.0254499,
          7.02123438, -0.24064246,
          2.45840128, -1.05865881,  0.81761517,  0.00712087, -0.06325559]
output_file = 'combined.npy'

if __name__ == "__main__":
    args = parser.parse_args()
    x = rates
    result = minimize(f1, x, method='COBYLA')
    '''
    最优的参数列表 方法名+rates参数数组
    #powell 
    #COBYLA 
    '''
    print('Optimization terminated successfully.')
    print('Success:', result.success)
    print('Status:', result.status)
    print('Function value:', result.fun)
    print('Optimized parameters (x):', result.x)  # 打印优化后的参数列表
    print('Number of iterations:', result.nit)
    print('Direction:', result.direc)
    print('Number of function evaluations:', result.nfev)
    print('---------------------------------')
    print('最后的Top1 Acc: {:.5f}%'.format(-f1(result.x)))
