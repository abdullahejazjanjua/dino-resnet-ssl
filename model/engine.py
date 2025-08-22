import torch
import torch.nn as nn
import logging


def train_one_epoch(
    student_model,
    teacher_model,
    dataloader,
    criterion,
    epoch,
    optimizer,
    global_iter,
    weight_schedule,
    lr_schedule,
    ema,
    args,
):
    avg_loss = []
    for img_idx, (imgs) in enumerate(dataloader):

        for idx, layer_param in enumerate(optimizer.param_groups):
            layer_param["lr"] = lr_schedule[global_iter]
            if idx == 0:
                layer_param["weight_decay"] = weight_schedule[global_iter]
            layer_param["weight_decay"] = 0.0

        img_global = imgs["global_crops"].flatten(0, 1)
        img_local = imgs["local_crops"].flatten(0, 1)

        student_outs_local = student_model(img_local)
        student_outs_global = student_model(img_global)

        student_outs = torch.cat([student_outs_global, student_outs_local], dim=0)

        teacher_outs = teacher_model(img_global)

        loss = criterion(
            teacher_outs=teacher_outs, student_outs=student_outs, current_epoch=epoch
        )

        avg_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            ema(
                student_model=student_model,
                teacher_model=teacher_model,
                step=global_iter,
            )

        global_iter += 1
        if args.verbose:
            print(f"    [{img_idx}]: {loss.item()}")

        elif img_idx > 0 and img_idx % 10 == 0 and not args.verbose:
            print(f"    loss: {sum(avg_loss) / len(avg_loss)}")

    return global_iter
