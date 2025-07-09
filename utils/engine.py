import tqdm
import torch
import torch.nn as nn
from .dino import DINOAug
from .data_loader import get_dataloader


def train_one_epoch(student_model, teacher_mode):

    Criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(set="train")
    augment = DINOAug()
    for img_idx, img, labels in enumerate(train_dataloader):
        local_augs, global_augs = augment(img)

    def update_centre(teacher_outs_1, teacher_outs_2, centre, m):
        return m * centre + (1 - m) * torch.concat(
            [teacher_outs_1, teacher_outs_2]
        ).mean(dim=0)


def evaluate():
    pass
