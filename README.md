# Partially Relevant Video Retrieval
Source code of our ACM MM'2022 paper [Partially Relevant Video Retrieval](https://arxiv.org/abs/2208.12510).

Homepage of our paper [http://danieljf24.github.io/prvr/](http://danieljf24.github.io/prvr/).

<img src="https://github.com/HuiGuanLab/ms-sl/blob/main/figures/pvr_model.png" width="1100px">

## Table of Contents

* [Environments](#environments)
* [MS-SL on TVR](#MS-SL-on-TVR)
  * [Required Data](#Required-Data)
  * [Model Training](#Training)
  * [Model Evaluation](#Evaluation)
  * [Expected Performance](#Expected-Performance)
* [MS-SL on Activitynet](#MS-SL-on-activitynet)
  * [Required Data](#Required-Data-1)
  * [Model Training](#Training-1)
  * [Model Evaluation](#Evaluation-1)
  * [Expected Performance](#Expected-Performance-1)
* [MS-SL on Charades-STA](#MS-SL-on-Charades-STA)
  * [Required Data](#Required-Data-2)
  * [Model Training](#Training-2)
  * [Model Evaluation](#Evaluation-2)
  * [Expected Performance](#Expected-Performance-2)
* [Reference](#Reference)
* [Acknowledgement](#Acknowledgement)

## Environments 
* **python 3.8**
* **pytorch 1.9.0**
* **torchvision 0.10.0**
* **tensorboard 2.6.0**
* **tqdm 4.62.0**
* **easydict 1.9**
* **h5py 2.10.0**
* **cuda 11.1**

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.
```
conda create --name ms_sl python=3.8
conda activate ms_sl
git clone https://github.com/HuiGuanLab/ms-sl.git
cd ms-sl
pip install -r requirements.txt
conda deactivate
```

## MS-SL on TVR

### Required Data
Run the following script to download the video feature and text feature of the TVR dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
# download the data of TVR
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
wget http://8.210.46.84:8787/prvr/data/tvr.tar
tar -xvf tvr.tar
```

### Training
Run the following script to train `MS-SL` network on TVR. It will save the chechpoint that performs best on the validation set as the final model.


```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
RUN_ID=runs_0
GPU_DEVICE_ID=0

./do_tvr.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
```
`$RUN_ID` is the name of the folder where the model is saved in.

`$GPU_DEVICE_ID` is the index of the GPU where we train on.
### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=tvr
FEATURE=i3d_resnet
ROOTPATH=$HOME/VisualSearch
MODELDIR=tvr-runs_0-2022_07_11_20_27_02 

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

We also provide the trained checkpoint on TVR, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.baidu.com/s/1d70cecBvwVqYwmvobJpbGw?pwd=zxzk). 
```
DATASET=tvr
FEATURE=i3d_resnet
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_tvr

wget http://8.210.46.84:8787/prvr/checkpoints/checkpoint_tvr.tar
tar -xvf checkpoint_tvr.tar -C $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```
`$DATASET` is the dataset that the model trained and evaluate on.

`$FEATURE` is the video feature corresponding to the dataset.

`$MODELDIR` is the path of checkpoints saved.
### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 13.5 | 32.1 | 43.4 | 83.4  | 172.3 |

## MS-SL on Activitynet
### Required Data
Run the following script to download the video feature and text feature of the Activitynet dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
wget http://8.210.46.84:8787/prvr/data/activitynet.tar
tar -xvf activitynet.tar
```

### Training
Run the following script to train `MS-SL` network on Activitynet.
```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
RUN_ID=runs_0
GPU_DEVICE_ID=0

./do_activitynet.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
```

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=activitynet-runs_0-2022_07_11_20_27_02

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

We also provide the trained checkpoint on Activitynet, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.baidu.com/s/10zMvaSGRyJWxGUgSPm2ySg?pwd=omgg).
```
DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_activitynet

wget http://8.210.46.84:8787/prvr/checkpoints/checkpoint_activitynet.tar
tar -xvf checkpoint_activitynet.tar -C $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 7.1 | 22.5 | 34.7 | 75.8  | 140.1 |

## MS-SL on Charades-STA

### Required Data
Run the following script to download the video feature and text feature of the Charades-STA dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
wget http://8.210.46.84:8787/prvr/data/charades.tar
tar -xvf charades.tar
```

### Training
Run the following script to train `MS-SL` network on Charades-STA.

```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
RUN_ID=runs_0
GPU_DEVICE_ID=0

./do_charades.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
```

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=charades
FEATURE=i3d_rgb_lgi
ROOTPATH=$HOME/VisualSearch
MODELDIR=charades-runs_0-2022_07_11_20_27_02

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```
We also provide the trained checkpoint on Charades-STA, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.baidu.com/s/1IuUI1D04gSSmfiHQwedbgg?pwd=w6mk).
```
DATASET=charades
FEATURE=i3d_rgb_lgi
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_charades

wget http://8.210.46.84:8787/prvr/checkpoints/checkpoint_charades.tar
tar -xvf checkpoint_charades.tar -C $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 1.8 | 7.1 | 11.8 | 47.7  | 68.4 |

## Reference
```
@inproceedings{dong2022prvr,
title = {Partially Relevant Video Retrieval},
author = {Jianfeng Dong and Xianke Chen and Minsong Zhang and Xun Yang and Shujie Chen and Xirong Li and Xun Wang},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
year = {2022},
}
```
## Acknowledgement
The codes are modified from [TVRetrieval](https://github.com/jayleicn/TVRetrieval) and [ReLoCLNet](https://github.com/IsaacChanghau/ReLoCLNet).

This work was supported by the National Key R&D Program of China (2018YFB1404102), NSFC (62172420,61902347, 61976188, 62002323), the Public Welfare Technology Research Project of Zhejiang Province (LGF21F020010), the Open Projects Program of the National Laboratory of Pattern Recognition, the Fundamental Research Funds for the Provincial Universities of Zhejiang, and Public Computing Cloud of RUC.
