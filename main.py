import os
import copy
import torch
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD

from utils.dataloader import ImageNet

from model.dino import DINO
from model.engine import train_one_epoch

from utils.loss import DINOloss
from utils.misc import EMA, cosine_decay, get_latest_checkpoint


def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")

    # Crops specifications
    parser.add_argument(
        "--local_crop_scale",
        default=(0.05, 0.4),
        type=tuple,
        help="Size of local crops",
    )
    parser.add_argument(
        "--global_crop_scale",
        default=(0.4, 1.0),
        type=tuple,
        help="Size of global crops",
    )
    parser.add_argument(
        "--num_local_crop", default=8, type=int, help="Number of local crops to use"
    )
    parser.add_argument(
        "--num_global_crop", default=2, type=int, help="Number of global crops to use"
    )

    # Model and Data specifications
    parser.add_argument("--dataset_path", default="tiny-imagenet-200", type=str)
    parser.add_argument("--model_name", default="resnet50", type=str)

    # Training specifications
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)

    parser.add_argument("--start_lr", default=0.0005, type=float)
    parser.add_argument("--end_lr", default=1e-6, type=float)

    parser.add_argument("--weight_decay_start", default=0.004, type=float)
    parser.add_argument("--weight_decay_end", default=0.4, type=float)

    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adamw"])

    # Teacher model parameters
    parser.add_argument("--teacher_temp", default=(0.04, 0.07), type=tuple)
    parser.add_argument("--warmup_teacher_epochs", default=30, type=int)
    parser.add_argument("--center_momentum", default=0.9, type=float)
    parser.add_argument("--start_ema_value", default=0.996, type=float)
    parser.add_argument("--end_ema_value", default=1.0, type=float)

    
    # Student model parameters
    parser.add_argument("--student_temp", default=0.1, type=float)

    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--grad_clip", default=3.0, type=float)
    parser.add_argument("--kaggle", action="store_true", help="Doesn't print anything to allow training in background.")

    # Additional parametres
    parser.add_argument(
        "--save_dir", default="logs/", type=str, help="Path to save checkpoints"
    )
    parser.add_argument(
        "--save_period", default=1, type=int, help="Intervals to save model state dict"
    )
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--num_workers", default=2, type=int)

    return parser


def main(args):
    assert args.num_global_crop == 2, f"Only 2 num_global_crop is supported"
    
    os.makedirs(args.save_dir, exist_ok=True)

    student_model = DINO(args.model_name).to(args.device)
    teacher_model = copy.deepcopy(student_model).to(args.device)
    
    if args.optimizer == "adamw":
        optimizer = AdamW(params=student_model.parameters())
    elif args.optimizer == "sgd":
        optimizer = SGD(student_model.parameters(), lr=0, momentum=0.9)
    else:
        raise NotImplementedError
    
    dataset = ImageNet(root=args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    iter_per_epochs = len(dataloader)
    global_iter = 0
    start_epoch = 0

    checkpoint_name = get_latest_checkpoint(args.save_dir)
    if checkpoint_name != 0:
        model_state = torch.load(os.path.join(args.save_dir, checkpoint_name))

        student_model.load_state_dict(model_state["student_model"])
        teacher_model.load_state_dict(model_state["teacher_model"])
        optimizer.load_state_dict(model_state["optimizer"])

        start_epoch = model_state["epoch"]
        global_iter = model_state["global_iter"]
        args = model_state["args"]

    else:
        print(f"Checkpoint not found!\n")
    
    if not args.kaggle:
        print(f"Total Images: {len(dataset)}")
        print(f"\nUsing Arguments")
        print(args)
        print("\nFreezing teacher model")
        for name, param in teacher_model.named_parameters():
            param.requires_grad = False
            print(f"{name} is frozen")

    criterion = DINOloss(
        nepochs=args.epochs,
        local_views=args.num_local_crop,
        global_views=args.num_global_crop,
        t_temp=args.teacher_temp,
        s_temp=args.student_temp,
        center_momentum=args.center_momentum,
        warmup_teacher_epochs=args.warmup_teacher_epochs,
        K=65536,
    ).to(args.device)

    lr_schedule = cosine_decay(
        start_value=(args.start_lr * args.batch_size / 256),
        end_value=args.end_lr,
        epochs=args.epochs,
        iter_per_epochs=iter_per_epochs,
        warmup_epochs=10,
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
        end_value=args.end_ema_value,
    )

    print("\nStarting training")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch}]: ")
        global_iter = train_one_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            dataloader=dataloader,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer,
            global_iter=global_iter,
            weight_schedule=weight_schedule,
            lr_schedule=lr_schedule,
            ema=ema,
            args=args,
        )


        if epoch % args.save_period == 0:
            model_state_dict = {
                "student_model": student_model.state_dict(),
                "teacher_model": teacher_model.state_dict(),
                "epoch": epoch + 1,
                "global_iter": global_iter,
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{epoch}.pth")
            torch.save(model_state_dict, checkpoint_path)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
