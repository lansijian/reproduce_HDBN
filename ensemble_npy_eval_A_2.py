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
    label_path = r'data\val_label.npy'  # 假设标签文件的路径
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
rates=[ 0.25295188 , 0.9906784 , 0.50188838 , -0.01512917 ,
        0.64582545 , 0.54793991 , -0.62225464 , 0.1760889 ,  #ctrgcn
        0.00920295 , 0.00805803 , 0.09316419 , -0.03026022 , #mstrgcn
        6.12391752 , -0.0001     , -1.60516981 ,              #tdgcn
        2.11799451 ,                                         #TEGCN
        -0.00777379 ,-0.69977849 ,-0.05539037 ,-0.05393019] #mixformer
'''
rates=[ 0.25295188 , 0.95371117 , 0.51420676 , -0.00909629 ,
        0.65327953 , 0.55331973 , -0.62225464 , 0.46124499 ,
        0.00920295 , 0.00805803 , 0.00703888 , -0.00831144 ,
        -0.001, 0.01,0.01,
        1.317      , -0.00777379 , 0.91497434 , 0.008603,
        0.01 ]
'''
'''
rates=[ 2.807e-01,9.783e-01,5.178e-01,6.186e-03,
       6.700e-01,6.053e-01,-5.866e-01,4.614e-01,
        0.01,0.01,0.01,0.01,
        1.317e+00,-5.792e-03,9.231e-01,8.603e-03]
'''
#rates = [0.44, 1.03, 0.62, 0.009,
#        0.67, 0.001, 0.001, 0.23,
        # 0.01, 0.01, 0.0001, 0.0001,
        # 0.001, 0.001, 0.0001, 0.0001,
#        0.99, 0.1, 0.95, 0.001]
output_file = 'combinedA.npy'

if __name__ == "__main__":
    args = parser.parse_args()
    result = minimize(f1, rates, method='powell')
    np.set_printoptions(threshold=np.inf)
    print(result)
    print('---------------------------------')

    print(result.x)
    print('最后的Top1 Acc: {:.5f}%'.format(-f1(result.x)))
