import torch.optim.adamw
import tqdm
import torch
import torch.nn as nn
from .dino import DINOAug, DINOloss
from .data_loader import get_dataloader


class Solver:
    def __init__(
        self, student_model, teacher_model, epochs, m=0.09, lr=1e-4, print_freq=10
    ):

        self.Criterion = DINOloss(m)
        self.augment = DINOAug()
        self.optimizer = torch.optim.adamw(student_model.params(), lr)
        self.model_s = student_model
        self.model_t = teacher_model
        self.print_freq = print_freq
        self.epochs = epochs

    def train(self):
        train_dataloader = get_dataloader(set="train")
        total = len(train_dataloader)
        for epoch in range(self.epochs):
            for img_idx, img in enumerate(train_dataloader):
                local_augs, global_augs = self.augment(img)

                s_local, s_global = self.model_s(local_augs), self.model_t(global_augs)
                with torch.no_grad:
                    t_local, t_global = self.model_t(local_augs), self.model_t(
                        global_augs
                    )

                l_1 = self.Criterion(s_local, t_global)
                l_2 = self.Criterion(t_local, s_global)

                loss = (l_1 + l_2) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if img_idx % self.print_freq == 0:
                    print(f"Epoch: [{epoch}] {img_idx}/{total} loss: {loss}")

    def update_centre(teacher_outs_1, teacher_outs_2, centre, m):

        return m * centre + (1 - m) * torch.concat(
            [teacher_outs_1, teacher_outs_2]
        ).mean(dim=0)

    def evaluate():
        pass
