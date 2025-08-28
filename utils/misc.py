import os
import torch

def initialize_model(model_path, model):
    
    model_state = torch.load(model_path, map_location="cpu", weights_only=False)
    model_state = model_state["teacher_model"]

    new_model_state = {}
    for k, v in model_state.items():
        if "DinoHead" in k:
            continue
        new_model_state[k] = v
    
    model_keys = model.state_dict().keys()
    assert set(model_keys) == set(new_model_state), f"Ensure that model is the same on as in model"

    model.load_state_dict(new_model_state)

    return model

def load_model_from_ckpt(checkpoint_path, model, optimizer):
    
    model_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(model_state["model"])
    optimizer.load_state_dict(model_state["optimizer"])

    start_epoch = model_state["epoch"]
    args = model_state["args"]

    return start_epoch, args

def get_latest_checkpoint(path):
    all_checkpoints = []

    for file in os.listdir(path):
        if file.endswith(".pth"):
            all_checkpoints.append(file)

    if len(all_checkpoints) == 0:
        return 0

    all_checkpoints = sorted(all_checkpoints)
    return all_checkpoints[-1]