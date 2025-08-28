import torch
import torch.nn as nn
from torchvision.models import get_model

from utils.misc import initialize_model

class DINOResnet(nn.Module):
    def __init__(self, model_path, model_id="resnet50", num_classes=10):
        super().__init__()

        assert model_id in ["resnet50", "resnet101"], f"Only ResNet50 and ResNet101 are supported and not {model_id}"
        model = get_model(model_id, weights=None)
        model = nn.Sequential(*model.children())[:-1]
        
        self.model = initialize_model(model=model, model_path=model_path)
        self.classification_head = nn.Linear(in_features=2048, out_features=num_classes)

        print(f"Freezing model")
        for n, p in self.model.named_parameters():
            p.requires_grad = False
            print(f"{n} is frozen")

    def forward(self, x):

        x = self.model(x).squeeze(-1).squeeze(-1)
        out = self.classification_head(x)

        return out
        