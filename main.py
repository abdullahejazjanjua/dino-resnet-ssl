import os
import copy
import argparse
import logging
from utils.dino import DINO

def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")

    # Crops specifications
    parser.add_argument("--local_crop_scale", default=(0.05, 0.4), type=tuple, help="Size of local crops")
    parser.add_argument("--global_crop_scale", default=(0.4, 1.0), type=tuple, help="Size of global crops")
    parser.add_argument("--num_local_crop", default=8, type=int, help="Number of local crops to use")

    # Model and Data specifications    
    parser.add_argument("--dataset_path", default="tiny-imagenet-200", type=str)
    parser.add_argument("--model_size", default=50, type=int)

    # Training specifications
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--loss_threshold", default=0.001, type=int)

    # Additional parametres
    parser.add_argument("--save_dir", default="logs/", type=str, help="Path to save checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Provides more detailed information. Recommended for debugging only!")

    return parser


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    student_model = DINO()
    teacher_model = copy.deepcopy(student_model)
    
    

if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
