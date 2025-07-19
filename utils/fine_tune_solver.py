import os
import torch.optim as optim
import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class Solver_finetune:
    def __init__(
        self,
        model,
        args,
        print_freq=100,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        factor=0.1,
        patience=5
    ):

        self.model = model
        self.print_freq = print_freq
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.batch_size = args.batch_size
        self.device = args.device
        self.verbose = args.verbose
        self.loss_threshold = args.loss_threshold
        # self.resume = args.resume
        self.checkpoint_path = args.checkpoint_path
        self.model_path = args.model_path

        self.Criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),  lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = factor, patience=patience)


        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def train(self):
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                shuffle=True)
        
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=False)


        total = len(train_dataloader)
        total_val = len(val_dataloader)
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
                    self.model.load_state_dict(checkpoint['model_t_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint["epoch"]
                    avg_loss = checkpoint['loss']
                    print(f"Checkpoint found {checkpoint_epoch}.")
                    print(f"Resuming training from epoch {start_epoch} !")
                else:
                    print(f"Checkpoint not found.")
                    print(f"Starting training!")

        # self.model.train()
        last_epoch_loss = 0
        for epoch in range(start_epoch, self.epochs):
            losses = []
            running_loss = 0
            print(f"Epoch: [{epoch}]")
            val_losses = []
            self.model.train()
            for img_idx, (img, grd_truth) in enumerate(train_dataloader):
                img = img.to(self.device)
                grd_truth = grd_truth.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(img)
                loss = self.Criterion(outputs, grd_truth)
                losses.append(loss)


                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                avg_run_loss = running_loss / (img_idx + 1)
                
                if img_idx % self.print_freq == 0 and img_idx > 0 and not self.verbose: 
                    print(f"    {img_idx}/{total} avg_loss: {avg_run_loss:.2f}")
                if self.verbose:
                    print(f"        Iterations [{img_idx} / {total}] loss: {loss.item():.20f} avg_loss: {avg_run_loss:.2f}")

            avg_loss = sum(losses) / len(losses)
            self.scheduler.step(avg_loss)
        
            checkpoint_save_dir = os.path.join(self.save_dir, f"checkpoint_{epoch}")


            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                }, checkpoint_save_dir)
            
            if epoch > start_epoch: 
                 loss_difference = abs(last_epoch_loss - avg_loss)

                 if loss_difference < self.loss_threshold:
                     print(f"Loss difference between epochs ({loss_difference:.4f}) is below the threshold ({self.loss_threshold}). Stopping training.")
                     break

            last_epoch_loss = avg_loss
            
            self.model.eval()
            correct = 0
            total_gt = 0
            with torch.no_grad():
                print(f"--------Validation--------")
                for val_idx, (img, grd_truth) in enumerate(val_dataloader):
                    img = img.to(self.device)
                    grd_truth = grd_truth.to(self.device)

                    out = self.model(img)
                    val_loss = self.Criterion(out, grd_truth)
                    val_losses.append(val_loss)

                    output = F.softmax(out, dim=1)

                    preds = torch.argmax(output, dim=1)
                    total_gt += grd_truth.size(0)
                    correct += (preds == grd_truth).sum().item()


                    if val_idx % self.print_freq == 0 and val_idx > 0:
                        print(f"        Iterations [{val_idx} / {total_val}] loss: {val_loss.item():.20f}")
                    
            print('Accuracy on val images: ', 100*(correct/total_gt), '%')

        print("-----------Done training-----------")
