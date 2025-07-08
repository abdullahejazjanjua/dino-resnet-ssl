import tqdm
import torch
import torch.nn as nn
from .data_loader import get_dataloader


def train_one_epoch(student_model, teacher_mode):
    
    Criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(set="train")
    

    




def evaluate():
    pass