# SimPool: Simple Pooling at the Inference for Efficient Semantic Segmentation
<div align="center">
  <img src="https://user-images.githubusercontent.com/74402562/224751955-567588cb-1760-4217-9b1f-66eaf0a3282d.png" width="90%" height="90%">
</div>

<p align="center">
Figure 1. Overall framework of SimPool methodology
</p>


<div align="center">
  <img src="https://user-images.githubusercontent.com/74402562/224756392-f1a3db5d-40a5-4755-9a66-2380c905bc37.png" width="65%" height="65%">
  <img src="https://user-images.githubusercontent.com/74402562/224756208-1b616692-02d1-48e2-b833-9398124446ca.png" width="25%" height="25%">
</div>
<p align="center">
Figure 2. Overview of SimPoolFormer and Pooling-based Embedding-Free Attention(PEFA) module

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for SimPool methodology and SimPoolFormer.

SimPool is a simple and efficient methodology to reduce the computation costs with little performance degradation.

SimPoolFormer is designed for applying our SimPool effectively.

We use [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as the codebase.


## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```pip install timm==0.3.2```

An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```

## Evaluation

`trained weights` were submitted with a zip file. 
  
```local_configs/``` contains config files. In config files, increase the ```sr_ratios``` of our backbone and ```additive_pooling_ratio``` of our decoder to reduce the computation costs. 

Example: evaluate ```SimPoolFormer-B0``` on ```ADE20K```:

```
# Single-gpu testing
python tools/test.py local_configs/simpoolformer/B0/simpoolformer.b0.512x512.ade.160k.py /path/to/checkpoint_file

# Multi-gpu testing
./tools/dist_test.sh local_configs/simpoolformer/B0/simpoolformer.b0.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>

# Multi-gpu, multi-scale testing
tools/dist_test.sh local_configs/simpoolformer/B0/simpoolformer.b0.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM> --aug-test
```

## Training

Download `ImageNet-1K pretrained weights` [[Pretrained](https://drive.google.com/drive/folders/1Y1JvQqi08NNS6zmAUmBuZnLP_aiWh5Qz?usp=sharing)]

Put them in a folder ```pretrained/```.

Example: train ```SimPoolFormer-B0``` on ```ADE20K```:

```
# Single-gpu training
python tools/train.py local_configs/simpoolformer/B0/simpoolformer.b0.512x512.ade.160k.py 

# Multi-gpu training
./tools/dist_train.sh local_configs/simpoolformer/B0/simpoolformer.b0.512x512.ade.160k.py <GPU_NUM>
```
