import cv2
import torch
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform

class rgbGray(ImageOnlyTransform):
    def __init__(self, p = 0.5):
        super().__init__(p=p)
        self.to_gray = A.ToGray(p=0.2)
        self.to_rbg = A.ToRGB(p=1.0)
    def apply(self, img, **params):
        im = self.to_gray(image=img)["image"]
        if im.shape[2] != 3:
            return self.to_rbg(image=im)["image"]
        return im
        

class DinoAugmentations(object):
    def __init__(
        self,
        num_local_crops=8,
        global_scale_crop=(0.4, 1.0),
        local_scale_crop=(0.05, 0.4),
    ):

        flip_and_color_jitter = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                rgbGray(p=0.2)
            ]
        )

        normalize = A.Compose(
            [A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), A.ToTensorV2()]
        )

        self.global_crop_01 = A.Compose(
            [
                A.RandomResizedCrop((224, 224), global_scale_crop),
                flip_and_color_jitter,
                A.GaussianBlur(p=1.0),
                normalize,
            ]
        )

        self.global_crop_02 = A.Compose(
            [
                A.RandomResizedCrop(
                    (224, 224), global_scale_crop, interpolation=cv2.INTER_CUBIC
                ),
                flip_and_color_jitter,
                A.GaussianBlur(p=0.1),
                A.Solarize(p=0.2),
                normalize,
            ]
        )

        self.local_crop = A.Compose(
            [
                A.RandomResizedCrop(
                    (96, 96), local_scale_crop, interpolation=cv2.INTER_CUBIC
                ),
                flip_and_color_jitter,
                A.GaussianBlur(p=0.5),
                normalize,
            ]
        )

        self.num_local_crops = num_local_crops

    def __call__(self, imgs):
    
        aug_imgs_local = []
        aug_imgs_global = []

        aug_imgs_global.append(self.global_crop_01(image=imgs)["image"])
        aug_imgs_global.append(self.global_crop_02(image=imgs)["image"])

        for _ in range(self.num_local_crops):
            aug_imgs_local.append(self.local_crop(image=imgs)["image"])

        return torch.stack(aug_imgs_local), torch.stack(aug_imgs_global)

if __name__ == "__main__":
    x = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    AUG = DinoAugmentations()

    aug_x_local, aug_x_global = AUG(x)
    print(f"Global crop: {aug_x_global.shape}")
    print(f"Local crop: {aug_x_local.shape}")

    
