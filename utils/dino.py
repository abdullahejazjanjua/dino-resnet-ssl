import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from dino_head import DinoHead


class DINO(nn.Module):
    def __init__(self, model=None, in_dim=1000):
        super().__init__()
        self.model = model or resnet50()
        self.dino_head = DinoHead(in_dim=in_dim)

    def forward(self, x):

        x = self.model(x)

        x = self.dino_head(x)

        return x

if __name__ == "__main__":
    
    model = DINO()  
    x = torch.randn((2, 3, 224, 224))
    x = model(x)
    print(f"x: {x.shape}")

