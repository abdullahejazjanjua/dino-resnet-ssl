import torch
import torch.nn as nn

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    args,
):
    total_len_dataset = len(dataloader)
    total_loss = 0

    for img_idx, (imgs, target) in enumerate(dataloader):

        sub_batch_size = args.batch_size // args.grad_steps

        optimizer.zero_grad()
        for i in range(args.grad_steps):
            start_idx = sub_batch_size * i
            end_idx = start_idx + sub_batch_size

            img = imgs[start_idx:end_idx, ...]

            outs = model(img)
            
            loss = criterion(outs, target)

            loss = loss / args.grad_steps
            total_loss += loss.item()
            loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()


        if args.kaggle:
            if img_idx == total_len_dataset - 1 or img_idx == 0:
                print(f"   [{img_idx + 1}/{total_len_dataset}] loss: {total_loss / (img_idx + 1)}")
        elif img_idx % args.print_freq == 0 or img_idx == total_len_dataset - 1:
            print(f"   [{img_idx + 1}/{total_len_dataset}] loss: {total_loss / (img_idx + 1)}")

    
    return (total_loss / total_len_dataset)
