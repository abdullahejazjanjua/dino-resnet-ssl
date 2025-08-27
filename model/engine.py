import torch
import torch.nn as nn

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
    total_len_dataset = len(dataloader)
    total_loss = 0
    for img_idx, imgs in enumerate(dataloader):

        sub_batch_size = args.batch_size // args.grad_steps

        for idx, layer_param in enumerate(optimizer.param_groups):
            layer_param["lr"] = lr_schedule[global_iter]
            if idx == 0:
                layer_param["weight_decay"] = weight_schedule[global_iter]
            else:
                layer_param["weight_decay"] = 0.0

        optimizer.zero_grad()
        losses = 0
        for i in range(args.grad_steps):
            start_idx = sub_batch_size * i
            end_idx = start_idx + sub_batch_size

            img_global = imgs["global_crops"][start_idx:end_idx, ...]
            img_local = imgs["local_crops"][start_idx:end_idx, ...]

            img_global = img_global.flatten(0, 1).to(args.device)
            img_local = img_local.flatten(0, 1).to(args.device)

            student_outs_local = student_model(img_local)
            student_outs_global = student_model(img_global)
            
            teacher_outs = teacher_model(img_global)

            student_outs = torch.cat([student_outs_global, student_outs_local], dim=0)

            loss = criterion(
                teacher_outs=teacher_outs, student_outs=student_outs, current_epoch=epoch
            )
            loss = loss / args.grad_steps
            total_loss += loss.item()
            loss.backward()
        
        nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            ema(
                student_model=student_model,
                teacher_model=teacher_model,
                step=global_iter,
            )

        if args.kaggle:
            if img_idx == total_len_dataset - 1 or img_idx == 0:
                print(
                    f"   [{img_idx}/{total_len_dataset}] loss: {total_loss / (img_idx + 1)} lr: {lr_schedule[global_iter]} weight_decay: {weight_schedule[global_iter]}"
                )
        elif img_idx % args.print_freq == 0 or img_idx == total_len_dataset - 1:

            print(f"   [{img_idx}/{total_len_dataset}] loss: {total_loss / (img_idx + 1)} lr: {lr_schedule[global_iter]} weight_decay: {weight_schedule[global_iter]}")

        global_iter += 1
    
    return global_iter, (total_loss / total_len_dataset)
