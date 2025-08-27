import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model

from .dino_head import DinoHead


class DINO(nn.Module):
    def __init__(self, model_id=None, in_dim=2048):
        super().__init__()
        assert model_id in ["resnet50", "resnet101"], f"Only ResNet50 and ResNet101 are supported and not {model_id}"

        model = get_model(model_id, weights=None)
        self.model = nn.Sequential(*model.children())[:-1]

        self.dino_head = DinoHead(in_dim=in_dim)

    def forward(self, x):
        
        x = self.model(x).squeeze(-1).squeeze(-1)
        x = self.dino_head(x)

        return x


if __name__ == "__main__":
    model = DINO(model_id="resnet50")
    print(model)
    x = torch.randn((2, 3, 224, 224))
    x = model(x)
    print(f"x: {x.shape}")
