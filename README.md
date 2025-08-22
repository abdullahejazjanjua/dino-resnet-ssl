# DINO ResNet50: Self-Supervised Feature Learning

This repository implements ResNet trained using the DINO (self-DIstillation with NO labels) self-supervised learning method. This project explores how DINO's self-supervised pretraining can enhance ResNet's feature learning capabilities compared to traditional supervised training.

## Overview

DINO is a self-supervised learning approach that trains vision transformers and CNNs without labels by using knowledge distillation between different augmented views of the same image. This implementation applies DINO to ResNet to investigate its effectiveness on convolutional architectures.

## Model Architecture

The implementation uses ResNet as the student and teacher networks in the DINO framework:
- **Student network**: ResNet with projection head
- **Teacher network**: Exponential moving average of student parameters
- **Training objective**: Cross-entropy loss between student and teacher outputs

## Installation

```bash
git clone https://github.com/abdullahejazjanjua/dino-resnet-ssl.git
cd dino-resnet-ssl
pip install -r requirements.txt
```

## Usage

### Training
> Refer to `main.py` for a list of availiable arguments.
```bash
python main.py --epochs 30 --model_size "resnet50 or resnet101" --device "cuda"  --dataset_path "path_to_your_dataset" 
```
- You can download the dataset from [here](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

### Training on Custom Dataset
- Your dataset must be in ImageNet format.
- In future, I will switch to COCO style as I find it easier to work with.

## Citation

If you use this work, please cite the original DINO paper:

```bibtex
@inproceedings{caron2021emerging,
  title={Emerging properties in self-supervised vision transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9650--9660},
  year={2021}
}
```
