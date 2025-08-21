import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoHead(nn.Module):
    def __init__(
                self, 
                in_dim,
                out_dim, 
                num_layers,
                hidden_dim,
                
            ):
        super().__init__()

        

