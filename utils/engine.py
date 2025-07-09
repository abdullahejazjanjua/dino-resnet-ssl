import os
import numpy as np
import torch.optim as optim
import tqdm
import torch
import torch.nn as nn
from .misc import EMA
from .dino import DINOAug, DINOloss
import torchvision
from .dataloader import ImageNet
from torch.utils.data import DataLoader


augment = DINOAug()

class Solver:
    def __init__(
        self,
        student_model,
        teacher_model,
        args,
        tps=0.1,
        tpt=0.06,
        m=0.09,
        lr=0.1,
        print_freq=10,
    ):

        self.model_s = student_model
        self.model_t = teacher_model
        self.print_freq = print_freq
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.batch_size = args.batch_size
        self.dataset_path = args.dataset_path
        self.m = m
        self.device = args.device
        # self.model_s.register_buffer("centre", torch.ones([1, 1024])).to(self.device)
        self.verbose = args.verbose

        self.Criterion = DINOloss(m, tps, tpt)
        self.optimizer = optim.SGD(self.model_s.parameters(),  lr=lr, momentum=0.9, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.1, patience=5)

    def train(self):
        train_dataset = ImageNet(root=self.dataset_path)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        total = len(train_dataloader)
        current_step = 0
        total_steps = total * self.epochs
        for epoch in range(self.epochs):
            losses = []
            running_loss = 0
            for img_idx, crops in enumerate(train_dataloader):
               
                local_augs = crops["local_crops"].to(self.device)
                global_augs = crops["global_crops"].to(self.device)
                # local_augs, global_augs = self.augment(img)
                # local_augs, global_augs = local_augs.unsqueeze(0), global_augs.unsqueeze(0)
                s_local, s_global = self.model_s(local_augs), self.model_s(global_augs)

                with torch.no_grad():
                    t_local, t_global = self.model_t(local_augs), self.model_t(global_augs)
                # for param in self.model_t.parameters():
                #     print(param.data)
                l_1 = self.Criterion(s_local, t_global, self.model_s.centre)
                l_2 = self.Criterion(t_local, s_global, self.model_s.centre)

                loss = (l_1 + l_2) / 2
                losses.append(loss.item())
                # print(f"loss: {loss.item()}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ema = EMA(total_steps=total_steps, current_step=current_step)
                ema(self.model_s, self.model_t)

                self.model_s.centre = self.update_centre(t_local, t_global)

                running_loss += loss.item()
                avg_run_loss = running_loss / (img_idx + 1)

                if img_idx % self.print_freq == 0:
                    print(f"Epoch: [{epoch}] {img_idx}/{total} avg_loss: {avg_run_loss:.2f}, running_loss: {running_loss:.2f}")
                if self.verbose:
                    print(f"    Iterations [{img_idx} / {total}] loss: {loss.item()}")
                current_step += 1

            checkpoint = os.path.join(self.save_dir, f"checkpoint_{epoch}")
            avg_loss = sum(losses) / len(losses)
            self.scheduler.step(avg_loss)

            torch.save({
            'epoch': epoch,
            'model_s_state_dict': self.model_s.state_dict(),
            'model_t_state_dict': self.model_t.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
            }, checkpoint)



    def update_centre(self, teacher_outs_1, teacher_outs_2):

        return self.m * self.model_s.centre + (1 - self.m) * torch.concat(
            [teacher_outs_1, teacher_outs_2]
        ).mean(dim=0)
    
    def evaluate():
        pass


# def batch_augment(batch):
#     local_batch_crops = []
#     global_batch_crops = []
#     remaining_dim = batch.shape()
#     for item in batch:
#         g, l = augment(item.permute(1, 2, 0).cpu().numpy())
#         local_batch_crops.append(l)
#         global_batch_crops.append(g)


#     return {
#         "local_crops": torch.stack(local_batch_crops),
#         "global_crops": torch.stack(global_batch_crops)
#     }
