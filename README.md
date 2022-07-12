# ms-sl
<img src="https://github.com/HuiGuanLab/ms-sl/blob/main/figures/pvr_model.png" width="1100px">



## Table of Contents

* [Environments](#environments)
* [MS-SL on TVR](#MS-SL-on-TVR)
  * [Required Data](#Required Data)
  * [Model Training](#Training)
  * [Model Evaluation](#Evaluation)
  * [Expected Performance](#Expected-Performance)
* [MS-SL on Activitynet](#MS-SL-on-activitynet)
  * [Required Data](#Required Data-1)
  * [Model Training](#Training-1)
  * [Model Evaluation](#Evaluation-1)
  * [Expected Performance](#Expected-Performance-1)
* [MS-SL on Charades-STA](#MS-SL-on-Charades-STA)
  * [Required Data](#Required Data-2)
  * [Model Training](#Training-2)
  * [Model Evaluation](#Evaluation-2)
  * [Expected Performance](#Expected-Performance-2)
* [Reference](#Reference)

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
The video feature and text feature of the TVR dataset can be downloaded [Here](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Run the following script to place the data in the specified path. 

```
# download the data of TVR
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
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

We also provide the trained checkpoint on TVR, it can be downloaded from [Here](). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_tvr.tar -C $ROOTPATH/tvr

DATASET=tvr
FEATURE=i3d_resnet
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_tvr

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```
`$DATASET` is the dataset that the model trained and evaluate on.

`$FEATURE` is the video feature corresponding to the dataset.

`$MODELDIR` is the path of checkpoints saved.
### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 13.8 | 32.8 | 43.7 | 83.2  | 173.5 |

## MS-SL on Activitynet
### Required Data
The video feature and text feature of the Activitynet dataset can be downloaded [Here](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Run the following script to place the data in the specified path. 

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
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

We also provide the trained checkpoint on Activitynet, it can be downloaded from [Here](https://pan.baidu.com/s/10zMvaSGRyJWxGUgSPm2ySg?pwd=omgg). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_activitynet.tar -C $ROOTPATH/results

DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_activitynet

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 7.1 | 22.5 | 34.7 | 75.8  | 140.1 |

## MS-SL on Charades-STA

### Required Data
The video feature and text feature of the Charades-STA dataset can be downloaded [Here](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Run the following script to place the data in the specified path. 

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
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

We also provide the trained checkpoint on Charades-STA, it can be downloaded from [Here](https://pan.baidu.com/s/1IuUI1D04gSSmfiHQwedbgg?pwd=w6mk). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_charades.tar -C $ROOTPATH/results

DATASET=charades
FEATURE=i3d_rgb_lgi
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_charades

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 1.8 | 7.1 | 11.8 | 47.7  | 68.4 |



