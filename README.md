https://github.com/lansijian/my_tegcn
https://github.com/lansijian/reproduce_HDBN
在仓库中有使用说明，模型权重与对应的准确度已压缩，训练集和测试集太大无法上传需要根据仓库readme转化

均在master中 

使用1：对于TEGCN是直接参照readme进行训练然后用准确度最高的模型得出测试集的置信度文件

使用2：对于HDBN则是下面的步骤：

首先对于原来的训练集和测试集进行转化，使用比赛方提供的训练集，然后使用data中的transform_2d.py和transform.py将原来的train、vel与test的npy文件和对应的label文件（test的label是自己创建的全部是0的文件）转化为适合训练集的npz文件，然后再于Mix_GCN中修改config文件夹中对应的训练和测试的文件路径，然后就进行训练，得到每一个训练的pkl文件后，用主文件夹下中的pkl_npy.py将每一个对应的模型转化为对应的置信度文件npy，之后利用主文件夹中的ensemble_npy_eval_A_2.py对置信度文件进行融合的权重分配（PS：我们使用了两个仓库中的置信度文件进行融合，ensemble_npy_eval_A.py是无操作的数据融合，ensemble_npy_eval_A_3.py是基于作者使用的高斯优化的方法得到大概的权重），之后找到最佳的权重分配，用训练得到的最优的模型pt文件测试生成test测试集对应的pkl文件，再同理转换为npy文件，在ensemble_npy_eval_B.py中利用最佳的权重分配对B测试集的置信度文件进行融合得到最终的置信度文件combined.npy，最后修改置信度名字为pred.npy，至此结束。

训练日志和模型权重均在output文件当中，环境配置为mix_GCN.yml中，修改的部分代码已经放在仓库中，训练参数设置均在config中，训练的文件的路径则按照各自的路径进行设置，训练文件为data文件夹 ，运行的方式则参考官方的readme文件，无较大改动。训练的npz文件过大无法上传GitHub仓库需要下载后用data中的python文件进行转化，output2（国赛的数据均在output2中）其中含有较大的文件全部进行了压缩，需解压！

最终版本采用了单目标优化模型来搜寻最佳的模型融合权重，我们以准确度最大为优化目标，采用powell方法对问题求解，使用的文件和参数如下（注意修改每一个置信度文件的路径）
```
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
    r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_J\J.npy',
    # r'ICMEW2024-Track10-main\Model_inference\Mix_GCN\output\tdgcn_V2_JM\JM.npy'
    r'TE-GCN-main\work_dir\2995\epoch1_test_score.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_B\B.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2\K2.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_K2M\K2M.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_BM\BM.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_J\J.npy',
    r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output2\output\V2_JM\V2_JM.npy',
]
```

```
rates=[ 0.25295188 , 0.9906784 , 0.50188838 , -0.01512917 ,
        0.64582545 , 0.54793991 , -0.62225464 , 0.1760889 ,  #ctrgcn
        0.00920295 , 0.00805803 , 0.09316419 , -0.03026022 , #mstrgcn
        6.12391752 , -0.0001     , -1.60516981 ,              #tdgcn
        2.11799451 ,                                         #TEGCN
        -0.00777379 ,-0.69977849 ,-0.05539037 ,-0.05393019] #mixformer
```

```
files = [
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_B\B.npy',
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_B_3D\B_3D.npy',
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_BM\BM.npy',
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_BM_3D\BM_3D.npy',

    r'Model_inference\Mix_GCN\output\ctrgcn_V2_J\J.npy',
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_J_3D\J_3D.npy',
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_JM\JM.npy',
    r'Model_inference\Mix_GCN\output\ctrgcn_V2_JM_3D\JM_3D.npy',


    #r'Model_inference\Mix_GCN\output\mstgcn_V2_B\B.npy',
    r'Model_inference\Mix_GCN\output\mstgcn_V2_BM\BM.npy',
    r'Model_inference\Mix_GCN\output\mstgcn_V2_J\J.npy',
    r'Model_inference\Mix_GCN\output\mstgcn_V2_JM\JM.npy',

    r'Model_inference\Mix_GCN\output\tdgcn_V2_B\B.npy',
    #r'Model_inference\Mix_GCN\output\tdgcn_V2_BM\BM.npy',
    r'Model_inference\Mix_GCN\output\tdgcn_V2_J\J.npy',
    #r'Model_inference\Mix_GCN\output\tdgcn_V2_JM\JM.npy',

    r'TE-GCN\pred.npy',
    r'Model_inference\Mix_Former\output\V2_B\B.npy',
    r'Model_inference\Mix_Former\output\V2_K2\K2.npy',
    r'Model_inference\Mix_Former\output\V2_K2M\K2M.npy',
    #r'ICMEW2024-Track10-main\Model_inference\Mix_Former\output\V2_K2\K2.npy',
    r'Model_inference\Mix_Former\output\V2_JM\V2_JM.npy',
    #r'Model_inference\Mix_Former\output\V2_BM\BM.npy'
]
rates=[ 0.24533144 ,0.98529858 ,0.52528801 ,-0.01512917  ,0.64582545  ,0.54781795,
 -0.62225464  ,0.1760889   ,0.03534935  ,0.06965514 ,-0.07291386  ,6.07546296,
 -1.63986077  ,2.11220217 ,-0.00777379 ,-0.72594603 ,-0.0592807  ,-0.05676209]
```
以上是两版得分最高的权重配置

# ICMEW2024-Track10
This is the official repo of **HDBN** and our work is one of the **top solutions** in the Multi-Modal Video Reasoning and Analyzing Competition (**MMVRAC**) of **2024 ICME** Grand Challenge **Track10**. <br />
Our work "[HDBN: A Novel Hybrid Dual-branch Network for Robust Skeleton-based Action Recognition](https://arxiv.org/abs/2404.15719)" is accepted by **2024 IEEE International Conference on Multimedia and Expo Workshop (ICMEW)**. <br />
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2404.15719) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hdbn-a-novel-hybrid-dual-branch-network-for/skeleton-based-action-recognition-on-uav)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-uav?p=hdbn-a-novel-hybrid-dual-branch-network-for)
# Framework
![image](https://github.com/liujf69/ICMEW2024-Track10/blob/main/framework.png)
Please install the environment based on **mix_GCN.yml**. <br />

# Dataset
Download official training and testing datasets, as well as A-list and B-list datasets
Download the data to the Data folder, and then convert the original data set into the model adaptation file format using **transform.py** and **transform_2d.py**,It should be noted that when converting the training set and the test set, you need to modify the corresponding **x_train, y_train,x_test,y_test** in the above two files to ensure that the training set corresponds to the train and the test set corresponds to the test, Change the correct file path.
```
python transform.py
python transform_2d.py
```

# Model inference
## Run Mix_GCN
Copy the **data** folder to **Model_inference/Mix_GCN/data**:
```
cp -r data Model_inference/Mix_GCN/data
cd ./Model_inference/Mix_GCN
pip install -e torchlight
```

**1. Run the following code to train the corresponding model.** <br />
# Change the configuration file (.yaml) of the corresponding modality.
# Mix_GCN Example
```
cd ./Model_inference/Mix_GCN
python main.py --config ./config/ctrgcn_V2_J.yaml --device 0
python main.py --config ./config/ctrgcn_V2_B.yaml --device 0
python main.py --config ./config/ctrgcn_V2_JM.yaml --device 0
python main.py --config ./config/ctrgcn_V2_BM.yaml --device 0
python main.py --config ./config/ctrgcn_V2_J_3d.yaml --device 0
python main.py --config ./config/ctrgcn_V2_B_3d.yaml --device 0
python main.py --config ./config/ctrgcn_V2_JM_3d.yaml --device 0
python main.py --config ./config/ctrgcn_V2_BM_3d.yaml --device 0
python main.py --config ./config/mstgcn_V2_J.yaml --device 0
python main.py --config ./config/mstgcn_V2_B.yaml --device 0
python main.py --config ./config/mstgcn_V2_JM.yaml --device 0
python main.py --config ./config/mstgcn_V2_BM.yaml --device 0
python main.py --config ./config/tdgcn_V2_J.yaml --device 0
python main.py --config ./config/tdgcn_V2_B.yaml --device 0
python main.py --config ./config/tdgcn_V2_JM.yaml --device 0
python main.py --config ./config/tdgcn_V2_BM.yaml --device 0
```

**Run the following code separately to obtain classification scores using different model weights.**
♥ Note that the file path in weight should be changed to the path of one of the .pt model weights in your trained model
**CSv2:**
```
python main.py --config ./config/ctrgcn_V2_J.yaml --phase test --save-score True --weights ./  
      /ctrgcn_V2_J.pt --device 0
python main.py --config ./config/ctrgcn_V2_B.yaml --phase test --save-score True --weights ./  
      /ctrgcn_V2_B.pt --device 0
python main.py --config ./config/ctrgcn_V2_JM.yaml --phase test --save-score True --weights ./ 
      /ctrgcn_V2_JM.pt --device 0
python main.py --config ./config/ctrgcn_V2_BM.yaml --phase test --save-score True --weights ./ 
      /ctrgcn_V2_BM.pt --device 0
python main.py --config ./config/ctrgcn_V2_J_3d.yaml --phase test --save-score True --weights ./ 
      /ctrgcn_V2_J_3d.pt --device 0
python main.py --config ./config/ctrgcn_V2_B_3d.yaml --phase test --save-score True --weights ./ 
      /ctrgcn_V2_B_3d.pt --device 0
python main.py --config ./config/ctrgcn_V2_JM_3d.yaml --phase test --save-score True --weights ./ 
      /ctrgcn_V2_JM_3d.pt --device 0
python main.py --config ./config/ctrgcn_V2_BM_3d.yaml --phase test --save-score True --weights ./ 
      /ctrgcn_V2_BM_3d.pt --device 0
###
python main.py --config ./config/tdgcn_V2_J.yaml --phase test --save-score True --weights ./    
      /tdgcn_V2_J.pt --device 0
python main.py --config ./config/tdgcn_V2_B.yaml --phase test --save-score True --weights ./    
      /tdgcn_V2_B.pt --device 0
python main.py --config ./config/tdgcn_V2_JM.yaml --phase test --save-score True --weights ./  
      /tdgcn_V2_JM.pt --device 0
python main.py --config ./config/tdgcn_V2_BM.yaml --phase test --save-score True --weights ./  
      /tdgcn_V2_BM.pt --device 0
###
python main.py --config ./config/mstgcn_V2_J.yaml --phase test --save-score True --weights ./  
      /mstgcn_V2_J.pt --device 0
python main.py --config ./config/mstgcn_V2_B.yaml --phase test --save-score True --weights ./  
      /mstgcn_V2_B.pt --device 0
python main.py --config ./config/mstgcn_V2_JM.yaml --phase test --save-score True --weights ./ 
      /mstgcn_V2_JM.pt --device 0
python main.py --config ./config/mstgcn_V2_BM.yaml --phase test --save-score True --weights ./ 
      /mstgcn_V2_BM.pt --device 0
```

**2. Verification report of the UAV dataset** <br />
To verify the correctness of your handling of the dataset, you can use the validation set from the original UAV-Human dataset to test the checkpoints above, and we provide the corresponding recognition accuracy below. <br />
**CSv1:**
```
ctrgcn_V1_J.pt: 43.52%
ctrgcn_V1_B.pt: 43.32%
ctrgcn_V1_JM.pt: 36.25%
ctrgcn_V1_BM.pt: 35.86%
ctrgcn_V1_J_3d.pt: 35.14%
ctrgcn_V1_B_3d.pt: 35.66%
ctrgcn_V1_JM_3d.pt: 31.08%
ctrgcn_V1_BM_3d.pt: 30.89%
###
tdgcn_V1_J.pt: 43.21%
tdgcn_V1_B.pt: 43.33%
tdgcn_V1_JM.pt: 35.74%
tdgcn_V1_BM.pt: 35.56%
###
mstgcn_V1_J.pt: 41.48%
mstgcn_V1_B.pt: 41.57%
mstgcn_V1_JM.pt: 33.82%
mstgcn_V1_BM.pt: 34.74%
```
**CSv2:**
```
ctrgcn_V2_J.pt: 69.00%
ctrgcn_V2_B.pt: 68.68%
ctrgcn_V2_JM.pt: 57.93%
ctrgcn_V2_BM.pt: 58.45%
ctrgcn_V2_J_3d.pt: 64.60%
ctrgcn_V2_B_3d.pt: 63.25%
ctrgcn_V2_JM_3d.pt: 55.80%
ctrgcn_V2_BM_3d.pt: 54.67%
###
tdgcn_V2_J.pt: 69.50%
tdgcn_V2_B.pt: 69.30%
tdgcn_V2_JM.pt: 57.74%
tdgcn_V2_BM.pt: 55.14%
###
mstgcn_V2_J.pt: 67.48%
mstgcn_V2_B.pt: 67.30%
mstgcn_V2_JM.pt: 54.43%
mstgcn_V2_BM.pt: 52.13%
```
## Run Mix_Former
Copy the **data**  folder to **Model_inference/Mix_Former/data**:
```
cd ./Model_inference/Mix_Former
```
**1. Run the following code to train the corresponding model.**
You have to change the corresponding **data-path** in the **config file**, just like：**data_path: data/train_2d.npz**
# Mix_Former Example
```
cd ./Model_inference/Mix_Former
python main.py --config ./config/mixformer_V2_J.yaml --device 0
python main.py --config ./config/mixformer_V2_JM.yaml --device 0
python main.py --config ./config/mixformer_V2_B.yaml --device 0
python main.py --config ./config/mixformer_V2_BM.yaml --device 0
python main.py --config ./config/mixformer_V2_k2.yaml --device 0
python main.py --config ./config/mixformer_V2_k2M.yaml --device 0
```

**Run the following code separately to obtain classification scores using different model weights.**
♥ Note that the file path in weight should be changed to the path of one of the .pt model weights in your trained model
```
python main.py --config ./config/mixformer_V2_J.yaml --phase test --save-score True --weights ./ 
      /mixformer_V2_J.pt --device 0 
python main.py --config ./config/mixformer_V2_B.yaml --phase test --save-score True --weights ./ 
      /mixformer_V2_B.pt --device 0 
python main.py --config ./config/mixformer_V2_JM.yaml --phase test --save-score True --weights ./ 
      /mixformer_V2_JM.pt --device 0 
python main.py --config ./config/mixformer_V2_BM.yaml --phase test --save-score True --weights ./ 
      /mixformer_V2_BM.pt --device 0 
python main.py --config ./config/mixformer_V2_k2.yaml --phase test --save-score True --weights ./ 
      /mixformer_V2_k2.pt --device 0 
python main.py --config ./config/mixformer_V2_k2M.yaml --phase test --save-score True --weights ./ 
      /mixformer_V2_k2M.pt --device 0 
```

**2. Verification report of the UAV dataset** <br />
**CSv1:**
```
mixformer_V1_J.pt: 41.43%
mixformer_V1_B.pt: 37.40%
mixformer_V1_JM.pt: 33.41%
mixformer_V1_BM.pt: 30.24%
mixformer_V1_k2.pt: 39.21%
mixformer_V1_k2M.pt: 32.60%
```
**CSv2:**
```
mixformer_V2_J.pt: 66.03%
mixformer_V2_B.pt: 64.89%
mixformer_V2_JM.pt: 54.58%
mixformer_V2_BM.pt: 52.95%
mixformer_V2_k2.pt: 65.56%
mixformer_V2_k2M.pt: 55.01%
```

# Ensemble
## Ensemble Mix_GCN
**1.** After running the code of model inference, we will obtain classification score files corresponding to each weight in the **output folder** named **epoch1_test_score.pkl**. <br />
**2.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
python Ensemble_MixGCN.py \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V1.txt \
--benchmark V1
```
**3.** You can obtain the final classification accuracy of CSv2 by running the following code:
```
python Ensemble_MixGCN.py \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V2.txt \
--benchmark V2
```
Please note that when running the above code, you may need to carefully **check the paths** for each **epoch1_test_score.pkl** file and the **val_sample** to prevent errors. <br />
```
**CSv1:** Emsemble Mix_GCN: 46.73%
**CSv2:** Emsemble Mix_GCN: 74.06%
```
## Ensemble Mix_Former
**1.** After running the code of model inference, we will obtain classification score files corresponding to each weight in the **output folder** named **epoch1_test_score.pkl**. <br />
**2.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
pip install scikit-optimize
```
```
python Ensemble_MixFormer.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl \
--benchmark V1
```

**3.** You can obtain the final classification accuracy of CSv2 by running the following code:
```
python Ensemble_MixFormer.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V2_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V2_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V2_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V2_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2M/epoch1_test_score.pkl \
--benchmark V2
```
Please note that when running the above code, you may need to carefully **check the paths** for each **epoch1_test_score.pkl** file and the **val_sample** to prevent errors. <br />
```
**CSv1:** Emsemble MixFormer: 47.23%
**CSv2:** Emsemble MixFormer: 73.47%
```
## Ensemble Mix_GCN and Mix_Former

**1.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
python Ensemble.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V1.txt \
--benchmark V1
```
or
```
 python Ensemble2.py --benchmark V1 \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl \
--mixformer_JM_Score  ./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl
```
**2.** You can obtain the final classification accuracy of CSv2 by running the following code:
```
python Ensemble.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V2_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V2_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V2_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V2_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V2.txt \
--benchmark V2
```
or
```
python Ensemble2.py --benchmark V2 \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V2_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V2_B/epoch1_test_score.pkl \
--mixformer_JM_Score  ./Model_inference/Mix_Former/output/skmixf__V2_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V2_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl
```

```
**CSv1:** Emsemble: 47.95%
**CSv2:** Emsemble: 75.36%
```

**Run the following to convert the accuracy pkl of each model prediction into a confidence file npy.**
Note that you change the path of the file you want to convert in your code.
```
python pkl_npy.py
```

**Run the following to find the best paramters ,  merge the confidence files obtained from the different model training tests and output a final confidence file.**
Note that you change the path of the file you want to convert in your code. 
After running the code, the fused confidence file and accuracy are output. For A list test set.
```
python ensemble_npy_eval_A.py
python ensemble_npy_eval_A_2.py
```
For B list test set
```
python ensemble_npy_eval_B.py
```

# Suggestion
We recommend comprehensively considering the three ensemble results **Ensemble Mix_GCN**, **Ensemble Mix_Former**, and **Ensemble Mix_GCN and Mix_Former**.

# Thanks
Our work is based on the [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [TD-GCN](https://github.com/liujf69/TD-GCN-Gesture), [MotionBERT](https://github.com/Walter0807/MotionBERT), [Ske-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer).
# Citation
```
@inproceedings{liu2024HDBN,
  author={Liu, Jinfu and Yin, Baiqiao and Lin, Jiaying and Wen, Jiajun and Li, Yue and Liu, Mengyuan},
  title={HDBN: A Novel Hybrid Dual-branch Network for Robust Skeleton-based Action Recognition}, 
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo Workshop (ICMEW)}, 
  year={2024}
}
```
# Contact
For any questions, feel free to contact: liujf69@mail2.sysu.edu.cn
