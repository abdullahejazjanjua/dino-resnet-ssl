import os
import argparse
import torch
import torch.nn as nn
from utils.fine_tune_solver import Solver_finetune
from torchvision.models import resnet50, resnet101, resnet152


def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--model_size", default=50, type=int)

    parser.add_argument("--epochs", default=30, type=int)

    parser.add_argument("--save_dir", default="logs/", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)


    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--checkpoint_path", type=str)

    return parser

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    resnet_base = resnet50()

    modules = list(resnet_base.children())[:-2]

    custom_layers = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(2048),
        nn.LayerNorm(2048),
        nn.Linear(2048, 1024)
    )

    model_part_to_load = nn.Sequential(*modules, custom_layers)

    checkpoint = torch.load(args.model_path, map_location=args.device)

    model_state_dict = checkpoint['model_t_state_dict']

    model_part_to_load.load_state_dict(model_state_dict, strict=False)

    model_part_to_load.to(args.device)

    for p in model_part_to_load.parameters():
            p.requires_grad = False

    classification_layer = nn.Linear(1024, args.num_classes)

    model = nn.Sequential(model_part_to_load, classification_layer)

    model.to(args.device)

    for p in classification_layer.parameters():
        p.requires_grad = True

    solver = Solver_finetune(model, args)

    ##############################################
    #            training                        #
    #############################################

    solver.train()

    ############################################
    #            end training                  #
    ############################################


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)