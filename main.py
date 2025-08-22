import os
import copy
import torch
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.dino import DINO
from model.engine import train_one_epoch

from utils.loss import DINOloss
from utils.misc import EMA, cosine_decay

def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")

    # Crops specifications
    parser.add_argument("--local_crop_scale", default=(0.05, 0.4), type=tuple, help="Size of local crops")
    parser.add_argument("--global_crop_scale", default=(0.4, 1.0), type=tuple, help="Size of global crops")
    parser.add_argument("--num_local_crop", default=8, type=int, help="Number of local crops to use")
    parser.add_argument("--num_global_crop", default=2, type=int, help="Number of global crops to use")


    # Model and Data specifications    
    parser.add_argument("--dataset_path", default="tiny-imagenet-200", type=str)
    parser.add_argument("--model_name", default="resnet50", type=str)

    # Training specifications
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    
    parser.add_argument("--start_lr", default=1e-6, type=float)
    parser.add_argument("--end_lr", default=0.0005, type=float)

    parser.add_argument("--weight_decay_start", default=0.004, type=float)
    parser.add_argument("--weight_decay_end", default=0.4, type=float)

    parser.add_argument("--start_ema_value", default=0.996, type=float)
    parser.add_argument("--end_ema_value", default=1.0, type=float)
    
    
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--grad_clip", default=3.0, type=float)
    parser.add_argument("--loss_threshold", default=0.001, type=float)



    # Additional parametres
    parser.add_argument("--save_dir", default="logs/", type=str, help="Path to save checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Provides more detailed information. Recommended for debugging only!")

    return parser


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    student_model = DINO(args.model_name).to(args.device)
    teacher_model = copy.deepcopy(student_model).to(args.device)

    print("Freezing teacher model")
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
        print(f"{name} is frozen")

    batch_size = 1
    num_samples = 100

    x_0 = torch.randn((num_samples, 8, 3, 224, 224))  
    x_1 = torch.randn((num_samples, 2, 3, 224, 224))  
    dataloader = DataLoader(list(zip(x_0, x_1)), batch_size=batch_size, shuffle=True)

    iter_per_epochs = len(dataloader)

    criterion = DINOloss(
                    nepochs=args.epochs, 
                    local_views=args.num_local_crop, 
                    global_views=args.num_global_crop,
                    t_temp=(0.04, 0.07),
                    s_temp=0.1,
                    center_momentum=0.9,
                    warmup_teacher_epochs=30,
                    K=65536
                    )
    
    optimizer = AdamW(params=student_model.parameters())
    lr_schedule = cosine_decay(
                        start_value=(0.0005 * args.batch_size / 256), 
                        end_value=args.end_lr,
                        epochs=args.epochs,
                        iter_per_epochs=iter_per_epochs,
                        warmup_epochs=10
                    )
    weight_schedule = cosine_decay(
                        start_value=args.weight_decay_start,
                        end_value=args.weight_decay_end,
                        epochs=args.epochs,
                        iter_per_epochs=iter_per_epochs,
                    )
    
    ema = EMA(
        nepochs=args.epochs,
        iter_per_epochs=iter_per_epochs,
        start_value=args.start_ema_value,
        end_value=args.end_ema_value
    )
    
    print("\nStarting training")

    global_iter = 0
    for epoch in range(args.epochs):

        train_one_epoch(
                    student_model=student_model, 
                    teacher_model=teacher_model, 
                    dataloader=dataloader,
                    criterion=criterion,
                    epoch=epoch,
                    optimizer=optimizer,
                    global_iter=global_iter,
                    weight_schedule = weight_schedule,
                    lr_schedule = lr_schedule,
                    ema=ema,
                    args=args
                    )
        

    
    

if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
