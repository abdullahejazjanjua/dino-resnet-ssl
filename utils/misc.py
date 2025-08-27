import os
import torch
import numpy as np

class EMA(object):
    def __init__(self, nepochs, iter_per_epochs, start_value=0.996, end_value=1.0):

        self.decay = cosine_decay(
            start_value=start_value,
            end_value=end_value,
            epochs=nepochs,
            iter_per_epochs=iter_per_epochs,
        )

    def __call__(self, student_model, teacher_model, step):

        for std_param, t_param in zip(
            student_model.parameters(), teacher_model.parameters()
        ):
            t_param.data = (
                self.decay[step] * t_param.data
                + (1 - self.decay[step]) * std_param.data
            )

def cosine_decay(
    start_value,
    end_value,
    epochs,
    iter_per_epochs,
    start_warmup_epoch=0,
    warmup_epochs=0,
):
    """
    value(t) = end_value + (start_value - end_value) * (1 + cos(π * t / T)) / 2
    """
    warmup_iters = warmup_epochs * iter_per_epochs
    if warmup_epochs > 0:
        linear_decay = np.linspace(
            start=start_warmup_epoch, stop=start_value, num=warmup_iters
        )

    total_iters = epochs * iter_per_epochs
    decay_iters = np.arange(stop=(total_iters - warmup_iters))

    decay_factor = (1 + np.cos(np.pi * decay_iters / total_iters)) / 2

    cosine_decay = end_value + (start_value - end_value) * decay_factor

    if warmup_epochs > 0:
        return np.concatenate([linear_decay, cosine_decay])
    return cosine_decay


def get_latest_checkpoint(path):
    all_checkpoints = []

    for file in os.listdir(path):
        if file.endswith(".pth"):
            all_checkpoints.append(file)

    if len(all_checkpoints) == 0:
        return 0

    all_checkpoints = sorted(all_checkpoints)
    return all_checkpoints[-1]


def load_model_from_ckpt(checkpoint_path, student_model, teacher_model, optimizer):
    model_state = torch.load(checkpoint_path)

    student_model.load_state_dict(model_state["student_model"])
    teacher_model.load_state_dict(model_state["teacher_model"])
    optimizer.load_state_dict(model_state["optimizer"])

    start_epoch = model_state["epoch"]
    global_iter = model_state["global_iter"]
    args = model_state["args"]

    return start_epoch, global_iter, args



if __name__ == "__main__":

    value = cosine_decay(1e-6, (0.0005 * 32 / 256), 30, 3, warmup_epochs=10)
    print(f"{value}")
