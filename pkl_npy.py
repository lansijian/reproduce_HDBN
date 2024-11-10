import pickle
import numpy as np

# 读取 pkl 文件
with open(r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_BM\epoch1_test_score.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 提取所有数据
data_array = np.array([value for value in data_dict.values()])

# 保存为 npy 文件
np.save(r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_BM\BM_b.npy', data_array)