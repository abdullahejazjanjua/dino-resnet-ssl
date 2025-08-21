import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOloss(nn.Module):

    def __init__(self, 
                 t_temp,
                 s_temp, 
                 center_momentum,
                 nepochs,
                 out_dim,
                 ):

        self.t_temp = t_temp
        self.s_temp = s_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, teacher_outs, student_outs): # [bs, num_views, out_dim]
        
        student_outs = student_outs / self.s_temp
        teacher_outs = (teacher_outs.detach() - self.center) / self.t_temp

        student_outs = torch.split(student_outs, dim=1) # [bs_1, out_dim_1, bs_2, out_dim_2]
        teacher_outs = torch.split(teacher_outs, dim=1) # [bs_1, out_dim_1, ... , bs_8, out_dim_8]

        
        total_loss = 0
        nterms = 0
        for t_outs in teacher_outs:
            for s_outs in student_outs:
                if t_outs == s_outs:
                    continue
                
                loss = torch.sum(-t_outs * F.log_softmax(s_outs, dim=-1), dim=-1) # 
                total_loss += loss.mean()
                nterms += 1

        total_loss /= nterms

        self.update_center(teacher_outs)

        return total_loss
                    
    @torch.no_grad()
    def update_center(self, teacher_outs):

        center = torch.mean(teacher_outs, dim=0, keepdim=True)

        return self.center * self.center_momentum + (1 - self.center_momentum) * center

                



