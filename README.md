# APCNet

This repository provides an implementation of **APCNet**, described in the paper:

*APCNet: An Attribute-Guided Point Cloud Network for Machining Feature and Geometric-Topological Attribute Recognition*

APCNet is an attribute-guided point cloud network for intersecting machining feature recognition. It jointly performs semantic segmentation, instance segmentation, and geometric-topological attribute recognition.

## Requirements

Anaconda or Miniconda is recommended for environment management.

Create a Conda environment using the following command:

```bash
conda create -n apcnet python=3.10
```

The required dependencies are listed below:

- Python 3.10
- PyTorch 2.8.0 with CUDA 12.8
- tqdm
- torch-ema
- torchmetrics
- wandb
- addict
- torch_scatter ([PyTorch 2.8.0 + CUDA 12.8 wheels](https://pytorch-geometric.com/whl/torch-2.8.0%2Bcu128.html))
- timm

## Dataset and Pretrained Model

The **MFPCA dataset** and the pretrained model weights (`best model`) are available at the following link:

- **Dataset and Model Weights:** [Baidu AI Studio – MFPCA Dataset and Model Weights](https://aistudio.baidu.com/dataset/detail/399462/file)
