import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DINOloss(nn.Module):

    def __init__(self, 
                 nepochs,
                 local_views,
                 global_views,
                 t_temp=(0.04, 0.07),
                 s_temp=0.1, 
                 center_momentum=0.9,
                 warmup_teacher_epochs=30,
                 K=65536,
                 ):
        super().__init__()

        self.t_temp = np.concatenate([
            np.linspace(start=t_temp[0], stop=t_temp[1], num=warmup_teacher_epochs),
            np.ones(nepochs - warmup_teacher_epochs) * t_temp[1]
        ])

        self.s_temp = s_temp
        self.center_momentum = center_momentum
        self.num_views = [local_views, global_views]


        self.register_buffer("center", torch.zeros(1, K))

    def forward(self, teacher_outs, student_outs, current_epoch): # [bs * num_views, out_dim]
    
        student_outs = student_outs / self.s_temp
        teacher_outs = F.softmax((teacher_outs - self.center) / self.t_temp[current_epoch], dim=-1)

        student_outs = torch.chunk(student_outs, chunks=(self.num_views[0] + self.num_views[1])) 
        teacher_outs = torch.chunk(teacher_outs.detach(), chunks=self.num_views[1])

        
        total_loss = 0
        nterms = 0
        for t_idx, t_outs in enumerate(teacher_outs):
            for s_idx, s_outs in enumerate(student_outs):
                if t_idx == s_idx:
                    continue
                
                loss = torch.sum(-t_outs * F.log_softmax(s_outs, dim=-1), dim=-1) # 
                total_loss += loss.mean()
                nterms += 1

        total_loss /= nterms

        self.update_center(teacher_outs)

        return total_loss
                    
    @torch.no_grad()
    def update_center(self, teacher_outs):

        center = torch.mean(torch.cat(teacher_outs, dim=0), dim=0, keepdim=True)

        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * center

                
if __name__ == "__main__":

    teacher_outs = torch.randn((4, 65536))
    student_outs = torch.randn((10, 65536))

    criterion = DINOloss(nepochs=100, local_views=8, global_views=2)
    loss = criterion(teacher_outs, student_outs, 1)
    print(f"loss: {loss.item()}")