import pandas as pd
import general_utils
import torch

from torch.utils.data import Dataset
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class TissueTileDataset(Dataset):
    def __init__(self, df, tile_dir, transforms=None):
        self.df = df
        self.tile_dir = tile_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        tile = Image.open(row['tile_file_name'])

        if self.transforms:
            tile = self.transforms(tile)

        label = float(row['tile_class'])

        return '/'.join(row['tile_file_name'].split('/')[-2:]), tile.float(), label
