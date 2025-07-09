import os
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision import transforms
from .dino import DINOAug
import torch.nn.functional as F


class ImageNet(Dataset):
    def __init__(self, root, set="train"):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.datapath = os.path.join(root, set)
        self.augment = DINOAug()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        # (current_path, directories in current_path, files in current_path)
        for current_path, current_dirs, current_files in os.walk(self.datapath):
            if not "images" in current_path:
                continue
            # print(current_path)
            # print(current_files)
            for file in current_files:
                if file.endswith(".JPEG") or file.endswith(".jpg"):
                    file_path = os.path.join(current_path, file)
                    self.image_paths.append(file_path)
            

    def __getitem__(self, index):
        img = decode_image(self.image_paths[index])
        # print(img)
        g, l = self.augment(img.permute(1, 2, 0).cpu().numpy())

        return {
            "local_crops": g,
            "global_crops": F.pad(l, (64, 64, 64, 64))
        }
        return self.transform(img)
    
    def __len__(self):
        return len(self.image_paths)

    
if __name__ == "__main__":
    d = ImageNet(root="tiny-imagenet-200")

