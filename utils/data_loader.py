import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ImageNet(Dataset):
    def __init__(self, annotation_path):
        super().__init__()

        self.annotations = pd.read_csv(annotation_path)


def get_dataloader(set="train"):
    pass