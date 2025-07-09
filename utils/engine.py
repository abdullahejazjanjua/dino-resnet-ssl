import os
import numpy as np
import torch.optim as optim
import tqdm
import torch
import torch.nn as nn
from .misc import EMA
from .dino import DINOAug, DINOloss
from .data_loader import get_dataloader


class Solver:
    def __init__(
        self,
        student_model,
        teacher_model,
        epochs,
        save_dir
        tps=0.1,
        tpt=0.06,
        m=0.09,
        lr=1e-4,
        print_freq=10,
    ):

        self.model_s = student_model
        self.model_t = teacher_model
        self.print_freq = print_freq
        self.epochs = epochs
        self.save_dir = save_dir
        self.model_s.register_buffer("centre", torch.ones([1, 1024]))

        self.Criterion = DINOloss(m, tps, tpt, self.model_s.centre)
        self.augment = DINOAug()
        self.ema = EMA()
        self.optimizer = optim.SGD(self.model_s.parameters(),  lr=0.1, momentum=0.9, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.1, patience=5)

    def train(self):
        train_dataloader = get_dataloader(set="train")
        # total = len(train_dataloader)
        for epoch in range(self.epochs):
            losses = []
            running_loss = 0
            for img_idx, img in enumerate([1]):
                img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                local_augs, global_augs = self.augment(img)
                local_augs, global_augs = local_augs.unsqueeze(0), global_augs.unsqueeze(0)
                s_local, s_global = self.model_s(local_augs), self.model_s(global_augs)

                with torch.no_grad():
                    t_local, t_global = self.model_t(local_augs), self.model_t(global_augs)

                l_1 = self.Criterion(s_local, t_global)
                l_2 = self.Criterion(t_local, s_global)

                loss = (l_1 + l_2) / 2
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.ema(self.model_s, self.model_t)


                running_loss += loss.item()
                avg_run_loss = running_loss / (img_idx + 1)

                if img_idx % self.print_freq == 0:
                    print(f"Epoch: [{epoch}] {img_idx}/{total} avg_loss: {avg_run_loss:.2f}, running_loss: {running_loss:.2f}")

            checkpoint = os.path.join(self.save_dir, f"checkpoint_{epoch}")
            torch.save({
            'epoch': epoch,
            'model_s_state_dict': self.model_s.state_dict(),
            'model_t_state_dict': self.model_t.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, checkpoint)
            avg_loss = sum(losses) / len(losses)
            self.scheduler.step(avg_loss)



    def update_centre(self, teacher_outs_1, teacher_outs_2):

        return self.m * self.centre + (1 - self.m) * torch.concat(
            [teacher_outs_1, teacher_outs_2]
        ).mean(dim=0)

    def evaluate():
        pass
