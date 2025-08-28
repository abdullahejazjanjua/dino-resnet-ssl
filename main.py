import os
import time
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import AdamW, SGD

from model.engine import train_one_epoch, evaluate
from model.model import DINOResnet
from utils.misc import initialize_model, load_model_from_ckpt, get_latest_checkpoint


def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")
    # Default values are the ones mentioned in the paper.

    # Model and Data specifications
    parser.add_argument("--dataset_path", default="tiny-imagenet-200", type=str)
    parser.add_argument("--model_name", default="resnet50", type=str)
    

    # Training specifications
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--grad_steps", default=8, type=int, help="For simulating higher batch sizes")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="Base learning rate at start of cosine decay")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Base weight decay at start of cosine decay")
    parser.add_argument("--optimizer", default="adamw", type=str, choices=["sgd", "adamw"])
    parser.add_argument("--grad_clip", default=3.0, type=float)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--model_path", type=str, default="logs/checkpoint_29.pth", help="path to the last checkpoint from DINO training")


    # Additional parametres
    parser.add_argument("--save_dir", default="logs/", type=str, help="Path to save checkpoints")
    parser.add_argument("--save_period", default=1, type=int, help="Intervals to save model state dict")
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--kaggle", action="store_true", help="Doesn't print anything to allow training in background.")


    return parser


def main(args):
    assert args.model_path is not None, f"model_path must be specified"

    os.makedirs(args.save_dir, exist_ok=True)

    model = DINOResnet(args.model_path)
    
    if args.optimizer == "adamw":
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError
    
    transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=(args.batch_size * args.grad_steps), shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    start_epoch = 0

    checkpoint_name = get_latest_checkpoint(args.save_dir)

    device = args.device
    if args.resume is not None:
        start_epoch, args, = load_model_from_ckpt(
                    checkpoint_path=args.resume, 
                    model=model, 
                    optimizer=optimizer,
                )
    elif checkpoint_name == - 1:
        start_epoch, args, = load_model_from_ckpt(
                    checkpoint_path=(os.path.join(args.save_dir, checkpoint_name)), 
                    model=model, 
                    optimizer=optimizer,
                )
    else:
        print(f"Checkpoint not found!\n")
    
    args.device = device # Ensures --device is used and not --device from checkpoint
    print(f"Using device: {args.device}")
    print(f"Total train images: {len(train_dataset)}")
    print(f"Total test images: {len(test_dataset)}")
    print(f"\nUsing Arguments")
    print(args)

    criterion = nn.CrossEntropyLoss()

    model = model.to(args.device)
    
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
            

    print("\nStarting training")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch}]: ")
        start = time.time()
        loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            args=args,
        )
        end = time.time()
        print("Average stats:")
        print(f"    loss: {loss}, time: {(end-start):.4f}s")
        print(f"Evaluating performance:")
        evaluate(
            model=model, 
            dataloader=test_dataloader,
            args=args
            )

        if epoch % args.save_period == 0:
            model_state_dict = {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{epoch}.pth")
            torch.save(model_state_dict, checkpoint_path)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)

