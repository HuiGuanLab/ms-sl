# ms-sl
<img src="https://github.com/HuiGuanLab/ms-sl/blob/main/figures/pvr.png" width="600px">
<img src="https://github.com/HuiGuanLab/ms-sl/blob/main/figures/ms-sl.png" width="600px">

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

## Getting Started

1.Download Features

We unify the data fomat of the four dataset used in our experiments, they can be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4).
We recommend extract the data to the directory $/HOME/ms_sl/data. Take TVR dataset as example:
```
cd ms-sl
tar -xvf tvr.tar -C data
```

2.Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)

```
source setup.sh
```

## Training and Inference


To train the model on the TVR, please run the following script:

```
./do_tvr.sh $RUN_ID $DATAPATH $GPU_DEVICE_ID
```
For example:
```
./do_tvr.sh runs_0 data/ 0
```
The model is placed in the directory $/HOME/ms-sl/results/$RUN_ID. To evaluate it, please run the following script:
```
./do_test.sh $DATASET $FEATURE $DATAPATH $RUN_ID
```
For example:
```
./do_test.sh tvr i3d_resnet data/ runs_0
```
