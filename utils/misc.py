import torch

def load_model_from_ckpt(checkpoint_path, model, optimizer):
    
    model_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(model_state["teacher_model"])

    return model