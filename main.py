import torch
import copy
import argparse
import logging
from utils.engine import train_one_epoch, evaluate
from torchvision.models import resnet50, resnet101, resnet152

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")

    parser.add_argument("--local_crop_scale", default=(0.05, 0.4), type=tuple)
    parser.add_argument("--global_crop_scale", default=(0.4, 1.0), type=tuple)
    parser.add_argument("--num_local_crop", default=8, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_size", default=50, type=int)

    parser.add_argument("--epochs", default=30, type=int)
    
    parser.add_argument("--save_dir", default="logs/", type=str)
    parser.add_argument("--verbose_training", action="store_true")

    return parser


def main(args):
    
    student_model = None
    if args.model_size == 50:
        student_model = resnet50(weights=None)
    elif args.model_size == 101:
        student_model = resnet101(weights=None)
    elif args.model_size == 152:
        student_model = resnet152(weights=None)
    
    if student_model is None:
        logging.critical(f"Expected size to be 50, 101 and 150 but {args.model_size} provided")
    teacher_model = copy.deepcopy(student_model)
    for p in teacher_model.parameters():
        p.requires_grad = False

    logging.info("Stop gradient added to teacher model")
    student_model.train()
    
    for epoch in range(args.epochs):

        output = train_one_epoch(student_model, teacher_model)
        

    


args = args_parser().parse_args()
main(args)
