import argparse
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        r_11 = r1[i]
        r_22 = r2[i]
        r_33 = r3[i]
        r_44 = r4[i]
        r_55 = r5[i]
        r_66 = r6[i]
        r_77 = r7[i]
        r_88 = r8[i]
        r_99 = r9[i]
        r_1010 = r10[i]
        r_1111 = r11[i]
        r_1212 = r12[i]
        r_1313 = r13[i]
        #r_1414 = r14[i]
        #r_1515 = r15[i]
        # r_1616 = r16[i]
        # r_1717 = r17[i]
        # r_1818 = r18[i]
        r_1919 = r19[i]
        r_2020 = r20[i]
        r_2121 = r21[i]
        r_2222 = r22[i]
        
        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8] \
            + r_1010 * weights[9] \
            + r_1111 * weights[10] \
            + r_1212 * weights[11] \
            + r_1313 * weights[12] \
        # + r_1616 * weights[11] \
        # + r_1717 * weights[12] \
        # + r_1818 * weights[13] \
        + r_1919 * weights[13] \
        + r_2020 * weights[14] \
        + r_2121 * weights[15] \
        + r_2222 * weights[16]
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mixformer_J_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.npy')
    parser.add_argument('--mixformer_B_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_B\B.npy')
    parser.add_argument('--mixformer_JM_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_JM\V2_JM.npy')
    # parser.add_argument('--mixformer_BM_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.npy')
    parser.add_argument('--mixformer_k2_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2\K2.npy')
    parser.add_argument('--mixformer_k2M_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2M\K2M.npy')
    parser.add_argument('--ctrgcn_J2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_J\J.npy')
    parser.add_argument('--ctrgcn_B2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_B\B.npy')
    parser.add_argument('--ctrgcn_JM2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_JM\JM.npy')
    parser.add_argument('--ctrgcn_BM2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_BM\BM.npy')
    parser.add_argument('--ctrgcn_J3d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_J_3D\J_3D.npy')
    parser.add_argument('--ctrgcn_B3d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_B_3D\B_3D.npy')
    parser.add_argument('--ctrgcn_JM3d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_JM_3D\JM_3D.npy')
    parser.add_argument('--ctrgcn_BM3d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\ctrgcn_V2_BM_3D\BM_3D.npy')
    parser.add_argument('--TE-GCN', default = r'TE-GCN-main\work_dir\2995\epoch1_test_score.npy')
    # parser.add_argument('--tdgcn_J2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.npy')
    # parser.add_argument('--tdgcn_B2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.npy')
    # parser.add_argument('--tdgcn_JM2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.npy')
    # parser.add_argument('--tdgcn_BM2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.npy')
    parser.add_argument('--mstgcn_J2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_J\J.npy')
    parser.add_argument('--mstgcn_B2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_B\B.npy')
    parser.add_argument('--mstgcn_JM2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_JM\JM.npy')
    parser.add_argument('--mstgcn_BM2d_Score', default = r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output2\output\mstgcn_V2_BM\BM.npy')
    arg = parser.parse_args()

    labels = np.load(r'data_C\val_label.npy')
    label = labels.tolist()  # 将 NumPy 数组转换为 Python 列表

    # r1 = np.load(arg.mixformer_J_Score)
    r10 = np.load(arg.mixformer_B_Score)
    r11 = np.load(arg.mixformer_JM_Score)
    # r4 = np.load(arg.mixformer_BM_Score)
    r12 = np.load(arg.mixformer_k2_Score)
    r13 = np.load(arg.mixformer_k2M_Score)
    r5 = np.load(arg.ctrgcn_J2d_Score)
    r1 = np.load(arg.ctrgcn_B2d_Score)
    r7 = np.load(arg.ctrgcn_JM2d_Score)
    r3 = np.load(arg.ctrgcn_BM2d_Score)
    r6 = np.load(arg.ctrgcn_J3d_Score)
    r2 = np.load(arg.ctrgcn_B3d_Score)
    r8 = np.load(arg.ctrgcn_JM3d_Score)
    r4 = np.load(arg.ctrgcn_BM3d_Score)
    r9 = np.load(arg.TE_GCN)
    # r15 = np.load(arg.tdgcn_J2d_Score)
    # r16 = np.load(arg.tdgcn_B2d_Score)
    # r17 = np.load(arg.tdgcn_JM2d_Score)
    # r18 = np.load(arg.tdgcn_BM2d_Score)
    r19 = np.load(arg.mstgcn_J2d_Score)
    r20 = np.load(arg.mstgcn_B2d_Score)
    r21 = np.load(arg.mstgcn_JM2d_Score)
    r22 = np.load(arg.mstgcn_BM2d_Score)

    space = [(0.0, 1.2) for i in range(17)]
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))
