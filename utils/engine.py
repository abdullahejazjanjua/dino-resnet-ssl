import os
import numpy as np
import torch.optim as optim
import tqdm
import logging
import torch
import torch.nn as nn
from .misc import EMA
from .dino import DINOAug, DINOloss
import torchvision
from .dataloader import ImageNet
from torch.utils.data import DataLoader


augment = DINOAug()
logger = logging.getLogger("dino")



class Solver:
    def __init__(
        self,
        student_model,
        teacher_model,
        args,
        tps=0.1,
        tpt=0.06,
        m=0.09,
        lr=1e-4,
        print_freq=100,
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
        # self.resume = args.resume
        self.loss_threshold = args.loss_threshold
        self.checkpoint_path = args.checkpoint_path
        self.Criterion = DINOloss(m, tps, tpt)
        self.optimizer =  torch.optim.AdamW(self.model_s.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.model_s.parameters(),  lr=lr, momentum=0.9, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.1, patience=5)

    def train(self):
        train_dataset = ImageNet(root=self.dataset_path)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        total = len(train_dataloader)
        current_step = 0
        total_steps = total * self.epochs
        start_epoch = 0

        if os.path.exists(self.save_dir):
            checkpoints = []
            for file in os.listdir(self.save_dir):
                if "checkpoint_" in file:
                    checkpoints.append(file)
            if len(checkpoints) > 0:
                checkpoints.sort()
                checkpoint_epoch = checkpoints[-1]
                checkpoint_path = os.path.join(self.save_dir, checkpoint_epoch)
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, weights_only=True)
                    checkpoint = torch.load(self.checkpoint_path, weights_only=True)
                    self.model_s.load_state_dict(checkpoint['model_s_state_dict'])
                    self.model_t.load_state_dict(checkpoint['model_t_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint["epoch"]
                    avg_loss = checkpoint['loss']
                    print(f"Checkpoint found {checkpoint_epoch}.")
                    print(f"Resuming training from epoch {start_epoch} !")
            else:
                print(f"Checkpoint not found.")
                print(f"Starting training!")
        last_epoch_loss = 0
        for epoch in range(start_epoch, self.epochs):
            losses = []
            running_loss = 0
            logger.info(f"Epoch: [{epoch}]")
            print(f"Epoch: [{epoch}]")
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

                if img_idx % self.print_freq == 0 and img_idx > 0 and not self.verbose:
                    print(f"    {img_idx}/{total} avg_loss: {avg_run_loss:.2f}")
                    logger.info(f"  {img_idx}/{total} avg_loss: {avg_run_loss:.2f}")
                if self.verbose:
                    print(f"        Iterations [{img_idx} / {total}] loss: {loss.item():.20f} avg_loss: {avg_run_loss:.2f}")
                current_step += 1

            checkpoint_save_dir = os.path.join(self.save_dir, f"checkpoint_{epoch}")
            avg_loss = sum(losses) / len(losses)
            self.scheduler.step(avg_loss)

            torch.save({
            'epoch': epoch,
            'model_s_state_dict': self.model_s.state_dict(),
            'model_t_state_dict': self.model_t.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
            }, checkpoint_save_dir)

            if epoch > start_epoch: 
                 loss_difference = abs(last_epoch_loss - avg_loss)

                 if loss_difference < self.loss_threshold:
                     print(f"Loss difference between epochs ({loss_difference:.4f}) is below the threshold ({self.loss_threshold}). Stopping training.")
                     break

            last_epoch_loss = avg_loss

    def update_centre(self, teacher_outs_1, teacher_outs_2):

        return self.m * self.model_s.centre + (1 - self.m) * torch.concat(
            [teacher_outs_1, teacher_outs_2]
        ).mean(dim=0)

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
