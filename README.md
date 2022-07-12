# ms-sl
<img src="https://github.com/HuiGuanLab/ms-sl/blob/main/figures/pvr_model.png" width="1100px">



## Table of Contents

* [Environments](#environments)
* [Required Data](#required-data)
* [MS-SL on TVR](#MS-SL-on-TVR)
  * [Model Training](#Training)
  * [Model Evaluation](#Evaluation)
  * [Expected Performance](#Expected-Performance)
* [MS-SL on Activitynet](#MS-SL-on-activitynet)
  * [Model Training](#Training-1)
  * [Model Evaluation](#Evaluation-1)
  * [Expected Performance](#Expected-Performance-1)
* [MS-SL on Charades-STA](#MS-SL-on-Charades-STA)
  * [Model Training](#Training-2)
  * [Model Evaluation](#Evaluation-2)
  * [Expected Performance](#Expected-Performance-2)
* [MS-SL on DiDemo](#MS-SL-on-DiDemo)
  * [Model Training](#Training-3)
  * [Model Evaluation](#Evaluation-3)
  * [Expected Performance](#Expected-Performance-3)
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

## Required Data
We use four public datasets: TVR, Activitynet, Charades-STA and DiDemo. The extracted feature is placed  in `$HOME/VisualSearch/`.


We unify the data fomat of the four dataset used in our experiments, they can be downloaded [Here](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4).

```
# download the data of TVR
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
tar -xvf tvr.tar

# download the data of Activitynet
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
tar -xvf activitynet.tar

# download the data of Charades-STA
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
tar -xvf charades.tar

# download the data of DiDemo
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
tar -xvf didemo.tar
```

Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)

```
source setup.sh
```

## MS-SL on TVR

### Training
Run the following script to train `MS-SL` network on TVR. It will save the chechpoint that performs best on the validation set as the final model.


```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
#Template:
./do_tvr.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
#Example:
./do_tvr.sh runs_0 $ROOTPATH 0
```
`$RUN_ID` is the name of the folder where the model is saved in.

`$GPU_DEVICE_ID` is the index of the GPU where we train on.
### Evaluation
The model is placed in the directory $/HOME/ms-sl/results/$RUN_ID after training. To evaluate it, please run the following script:
```
#Template:
./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
#Example:
./do_test.sh tvr i3d_resnet $ROOTPATH tvr-runs_0-2022_07_11_20_27_02
```

We also provide the trained checkpoint on TVR, it can be downloaded from [Here](). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_tvr.tar -C results/

./do_test.sh tvr i3d_resnet $ROOTPATH checkpoint_tvr
```
`$DATASET` is the dataset that the trained model evaluate on.

`$FEATURE` is the video feature corresponding to the dataset.
`$MODELDIR` is the path of checkpoints saved.
### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 13.8 | 32.8 | 43.7 | 83.2  | 173.5 |

## MS-SL on Activitynet

### Training
Run the following script to train `MS-SL` network on Activitynet.
```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
#Template:
./do_tvr.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
#Example:
./do_tvr.sh runs_0 $ROOTPATH 0
```

### Evaluation
The model is placed in the directory $/HOME/ms-sl/results/$RUN_ID after training. To evaluate it, please run the following script:
```
#Template:
./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
#Example:
./do_test.sh activitynet i3d $ROOTPATH activitynet-runs_0-2022_07_11_20_27_02
```

We also provide the trained checkpoint on Activitynet, it can be downloaded from [Here](). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_activitynet.tar -C results/

./do_test.sh activitynet i3d $ROOTPATH checkpoint_activitynet
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 7.1 | 22.5 | 34.7 | 75.8  | 140.1 |

## MS-SL on Charades-STA

### Training
Run the following script to train `MS-SL` network on Charades-STA.

```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
#Template:
./do_tvr.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
#Example:
./do_tvr.sh runs_0 $ROOTPATH 0
```

### Evaluation
The model is placed in the directory $/HOME/ms-sl/results/$RUN_ID after training. To evaluate it, please run the following script:
```
#Template:
./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
#Example:
./do_test.sh charades i3d_rgb_lgi $ROOTPATH charades-runs_0-2022_07_11_20_27_02
```

We also provide the trained checkpoint on Charades-STA, it can be downloaded from [Here](). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_charades.tar -C results/

./do_test.sh charades i3d_rgb_lgi $ROOTPATH checkpoint_charades
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 1.8 | 7.1 | 11.8 | 47.7  | 68.4 |

## MS-SL on DiDemo

### Training
Run the following script to train  `MS-SL` network on DiDemo.

```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate ms-sl

ROOTPATH=$HOME/VisualSearch
#Template:
./do_tvr.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
#Example:
./do_tvr.sh didemo_runs_0 $ROOTPATH 0
```

### Evaluation
The model is placed in the directory $/HOME/ms-sl/results/$RUN_ID after training. To evaluate it, please run the following script:
```
#Template:
./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
#Example:
./do_test.sh didemo rgb_flow $ROOTPATH didemo-runs_0-2022_07_11_20_27_02
```

We also provide the trained checkpoint on Activitynet, it can be downloaded from [Here](). Run the following script to evaluate it.
```
ROOTPATH=$HOME/VisualSearch
cd ms-sl
tar -xvf checkpoint_didemo.tar -C results/

./do_test.sh didemo rgb_flow $ROOTPATH checkpoint_didemo
```

### Expected performance 

|             | R@1  | R@5  | R@10 | R@100 | SumR  |
| :---------: | :--: | :--: | :--: | :---: | :---: |
| Text-to-Video | 6.6 | 19.3 | 27.7 | 72.4  | 125.9 |


