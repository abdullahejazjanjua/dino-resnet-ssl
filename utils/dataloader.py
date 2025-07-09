import os
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision import transforms
from .dino import DINOAug
import torch.nn.functional as F
import albumentations as A
class ImageNet(Dataset):
    def __init__(self, root, set="train"):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.datapath = os.path.join(root, set)
        self.augment = DINOAug()
        self.to_rbg = A.Compose([
            A.ToRGB(p=1.0),
            A.ToTensorV2()
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
        if img.shape[0] != 3:
            # print(f"Found {img.shape} at {self.image_paths[index]}")
            # index += 1
            # return self.__getitem__(index)
            img = self.to_rbg(image=img.permute(1, 2, 0).cpu().numpy())["image"]
            
        g, l = self.augment(img.permute(1, 2, 0).cpu().numpy())
       
        return {
            "local_crops": F.pad(l, (64, 64, 64, 64)),
            "global_crops": g
        }
        # return self.transform(img)
    
    def __len__(self):
        return len(self.image_paths)

    
if __name__ == "__main__":
    d = ImageNet(root="tiny-imagenet-200")

