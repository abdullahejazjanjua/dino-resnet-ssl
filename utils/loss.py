import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOloss(nn.Module):
    def __init__(self, tpt, tps, m):

        self.teacher_tmp = tpt
        self.student_tmp = tps
        self.m = m

        self.register_buffer("center", )

    def __call__(self, teacher_outs, student_outs):
        teacher_outs = teacher_outs.detach()

        p_teacher_outs = F.softmax(
            (teacher_outs - self.centre) / self.teacher_tmp, dim=1
        )
        # p_student_outs = F.softmax(student_outs / self.student_tmp, dim=1) fucking nans
        p_student_outs = F.log_softmax(student_outs / self.student_tmp, dim=1)

        # return -(p_teacher_outs * torch.log(p_student_outs)).sum(dim=1).mean()
        return -(p_teacher_outs * p_student_outs).sum(dim=1).mean()


