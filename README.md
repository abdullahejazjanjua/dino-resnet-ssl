# TODO:
- [] Implement Dataloader
- [x] Get model outputs using MLP till softmax
- [x] Calculate centre (Find shape aswell)
- [x] Compute loss
- [x] complete training pipeline 

# DINO ResNet50: Self-Supervised Feature Learning

This repository implements ResNet50 trained using the DINO (self-DIstillation with NO labels) self-supervised learning method. The project explores how DINO's self-supervised pretraining can enhance ResNet50's feature learning capabilities compared to traditional supervised training.

## Overview

DINO is a self-supervised learning approach that trains vision transformers and CNNs without labels by using knowledge distillation between different augmented views of the same image. This implementation applies DINO to ResNet50 to investigate its effectiveness on convolutional architectures.

## Key Features

- **Self-supervised pretraining** using DINO methodology
- **ResNet50 backbone** for feature extraction
- **Comparative analysis** of supervised vs. self-supervised feature learning
- **Evaluation metrics** for downstream tasks

## Model Architecture

The implementation uses ResNet50 as the student and teacher networks in the DINO framework:
- **Student network**: ResNet50 with projection head
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
```bash
python train.py 
```

### Evaluation
```bash
python evaluate.py --model_path checkpoints/dino_resnet50.pth --dataset imagenet
```

## Results

| Method | Top-1 Accuracy | Top-5 Accuracy | Linear Eval |
|--------|---------------|---------------|-------------|
| Supervised ResNet50 | XX.X% | XX.X% | XX.X% |
| DINO ResNet50 | XX.X% | XX.X% | XX.X% |

## Dataset

The model is trained on ImageNet and evaluated on standard benchmarks.

## Experiments

This repository includes experiments comparing:
- Supervised ResNet50 baseline
- DINO-pretrained ResNet50
- Feature quality analysis
- Transfer learning performance

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
