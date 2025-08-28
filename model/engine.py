import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    args,
):
    total_len_dataset = len(dataloader)
    total_loss = 0

    for img_idx, (imgs, targets) in enumerate(dataloader):

        sub_batch_size = args.batch_size // args.grad_steps

        optimizer.zero_grad()
        for i in range(args.grad_steps):
            start_idx = sub_batch_size * i
            end_idx = start_idx + sub_batch_size

            new_imgs = imgs[start_idx:end_idx, ...].to(args.device)
            new_targets = targets[start_idx:end_idx]
            outs = model(new_imgs)
            
            loss = criterion(outs, new_targets)

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


def evaluate(model, dataloader, args):

    len_dataset = len(dataloader)
    correct = 0
    total = 0
    for img_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(args.device)
        
        outs = model(imgs)
        out_probs = F.softmax(outs, dim=-1)
        preds = torch.argmax(out_probs, dim=-1)

        correct += (preds == targets.to(args.device)).sum().item()
        total += targets.shape[0]

        if img_idx % args.print_freq == 0 or img_idx == len_dataset - 1:
            print(f"{img_idx + 1}/{len_dataset} Accuracy: {((correct / total) * 100):.2f}")

