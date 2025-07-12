import os
import copy
import argparse
import logging
from utils.misc import MetricLogger
from utils.engine import *
from torchvision.models import resnet50, resnet101, resnet152


def args_parser():
    parser = argparse.ArgumentParser(description="DINO parametres")

    parser.add_argument("--local_crop_scale", default=(0.05, 0.4), type=tuple)
    parser.add_argument("--global_crop_scale", default=(0.4, 1.0), type=tuple)
    parser.add_argument("--num_local_crop", default=8, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dataset_path", default="tiny-imagenet-200", type=str)
    parser.add_argument("--model_size", default=50, type=int)

    parser.add_argument("--device", default="mps", type=str)

    parser.add_argument("--epochs", default=30, type=int)

    parser.add_argument("--save_dir", default="logs/", type=str)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--loss_threshold", default=0.001, type=int)
    parser.add_argument("--checkpoint_path", type=str)

    return parser


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=args.save_dir + "/log.txt",
        filemode="a",
        format="%(message)s",
    )
    logger = logging.getLogger("dino")
    model = None

    if args.model_size == 50:
        model = resnet50(weights=None)
    elif args.model_size == 101:
        model = resnet101(weights=None)
    elif args.model_size == 152:
        model = resnet152(weights=None)

    if model is None:
        logging.critical(
            f"Expected size to be 50, 101 and 150 but {args.model_size} provided"
        )
        raise Exception
    resnet_layers = list(model.children())[:-2]
    additional_layers = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(2048),
        nn.LayerNorm(2048),
        nn.Linear(2048, 1024)
    )
    # last_layers.append(nn.Flatten())
    # last_layers.append(nn.LazyLinear(2048))
    # last_layers.append(nn.LayerNorm(2048))
    # last_layers.append(nn.Linear(2048, 1024))

    student_model = nn.Sequential(*resnet_layers, additional_layers)
    student_model.register_buffer("centre", torch.ones([1, 1024]))

    teacher_model = copy.deepcopy(student_model).to(args.device)

    for p in teacher_model.parameters():
        p.requires_grad = False
    student_model.to(args.device).train()

    solver = Solver(student_model, teacher_model, args)

    ##############################################
    #            training                        #
    #############################################

    solver.train()

    ############################################
    #            end training                  #
    ############################################

    # #######################################################
    # #                    Evaluate                         #
    # #######################################################

    # solver.evaluate(student_model, teacher_model, set="val")

    # ########################################################
    # #                  End Evaluate                        #
    # ########################################################


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
