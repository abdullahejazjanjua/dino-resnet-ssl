import cv2
import torch
import numpy as np
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
        

class DINOAug(object):
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

    def __call__(self, imgs, multi_crop=False):
    
        if multi_crop:
            aug_imgs_local = []
            aug_imgs_global = []

            aug_imgs_global.append(self.global_crop_01(image=imgs)["image"])
            aug_imgs_global.append(self.global_crop_02(image=imgs)["image"])

            for _ in range(self.num_local_crops):
                aug_imgs_local.append(self.local_crop(image=imgs)["image"])

            return torch.stack(aug_imgs_local), torch.stack(aug_imgs_global)
        else:
            if self.global_crop_01(image=imgs)["image"].shape[0] != 3 or self.local_crop(image=imgs)["image"].shape[0] != 3:
                print()
            return (
                self.global_crop_01(image=imgs)["image"],
                self.local_crop(image=imgs)["image"],
            )


class DINOloss:
    def __init__(self, tpt, tps, m):

        self.teacher_tmp = tpt
        self.student_tmp = tps
        self.m = m

    def __call__(self, teacher_outs, student_outs, centre):
        teacher_outs = teacher_outs.detach()

        p_teacher_outs = F.softmax(
            (teacher_outs - centre) / self.teacher_tmp, dim=1
        )
        # p_student_outs = F.softmax(student_outs / self.student_tmp, dim=1) fucking nans
        p_student_outs = F.log_softmax(student_outs / self.student_tmp, dim=1)

        # return -(p_teacher_outs * torch.log(p_student_outs)).sum(dim=1).mean()
        return -(p_teacher_outs * p_student_outs).sum(dim=1).mean()



if __name__ == "__main__":
    x = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    AUG = DINOAug()

    aug_x = AUG(x)

    print(len(aug_x))
    for x_ in aug_x:
        print(x_["image"].shape)
        print()
