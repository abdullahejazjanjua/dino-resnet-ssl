import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P

class DinoHead(nn.Module):
    def __init__(
                self, 
                in_dim,
                K=65536,
                bottleneck_dim=256,
                nlayers=3, 
                hidden_dim=2048,
            ):
        super().__init__()

        self.input_layer = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.n_mlp = DinoHead._make_mlp(hidden_dim=hidden_dim, nlayers=nlayers)
        self.out_layer = nn.Linear(in_features=hidden_dim, out_features=bottleneck_dim)

        self.last_layer = P.weight_norm(
            nn.Linear(in_features=bottleneck_dim, out_features=K, bias=False)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, x):

        x = self.input_layer(x)

        x = self.n_mlp(x)

        x = F.normalize(x, p=2, dim=-1)

        x = self.out_layer(x)
        
        x = self.last_layer(x)

        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_mlp(hidden_dim, nlayers):
        layers = nn.ModuleList()
        for _ in range(nlayers):
            layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            ])

        return nn.Sequential(*layers)
    

if __name__ == "__main__":

    x = torch.randn((2, 3, 224, 224))
    x = x.flatten(1, -1)
    projection_head = DinoHead(in_dim=x.shape[-1])
    x = projection_head(x)
    print(f"x: {x.shape}")