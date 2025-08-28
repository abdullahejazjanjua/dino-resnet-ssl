import os
from torch.utils.data import Dataset
from torchvision.io import decode_image
from .augmentations import DinoAugmentations
import torch.nn.functional as F
import albumentations as A


class ImageNet(Dataset):
    def __init__(self, root, set="train"):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.datapath = os.path.join(root, set)
        self.augment = DinoAugmentations()
        self.to_rbg = A.Compose([A.ToRGB(p=1.0), A.ToTensorV2()])

        for current_path, _, current_files in os.walk(self.datapath):

            if not "images" in current_path:
                continue

            for file in current_files:
                if file.endswith(".JPEG") or file.endswith(".jpg"):
                    file_path = os.path.join(current_path, file)
                    self.image_paths.append(file_path)

    def __getitem__(self, index):

        img = decode_image(self.image_paths[index])
        if img.shape[0] != 3:
            img = self.to_rbg(image=img.permute(1, 2, 0).cpu().numpy())["image"]

        l, g = self.augment(img.permute(1, 2, 0).cpu().numpy())

        return {"local_crops": l, "global_crops": g}

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    d = ImageNet(root="tiny-imagenet-200")
