from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import general_utils


class HistoCoxDataset(Dataset):
    def __init__(self, df, tile_dir, transforms=None):
        self.df = df
        self.tile_dir = tile_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        address = row.tile_address
        img_hid = row.img_hid

        file_name = os.path.join(self.tile_dir,
                                 img_hid,
                                 str(address[0]).replace('(', '').replace(')', '').replace(', ', '_') + '.png')
        tile = Image.open(file_name)

        if self.transforms:
            tile = self.transforms(tile)

        try:
            survival = row['duration']
        except KeyError:
            survival = row['survival']

        observed = row['observed']

        return '/'.join(file_name.split('/')[-2:]), tile, survival, observed


class HistoMarkerDataset(Dataset):
    def __init__(self, df, tile_dir, transforms=None, score_col='is_hrd'):
        self.df = df
        self.tile_dir = tile_dir
        self.transforms = transforms
        self.score_col = score_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        img_hid = row.img_hid

        file_name = os.path.join(self.tile_dir,
                                 str(img_hid),
                                 row.tile_file_name)
        tile = Image.open(file_name)

        if self.transforms:
            tile = self.transforms(tile)

        score = float(row[self.score_col])

        return '/'.join(file_name.split('/')[-2:]), tile, score


class TissueTileDataset(Dataset):
    def __init__(self, df, tile_dir, transforms=None):
        self.df = df
        self.tile_dir = tile_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        file_name = general_utils.get_tile_file_name(self.tile_dir, row.img_hid, row.tile_address)
        tile = Image.open(file_name)

        if self.transforms:
            tile = self.transforms(tile)

        label = float(row['tile_class'])

        return '/'.join(file_name.split('/')[-2:]), tile.float(), label


class EmbeddingDataset(Dataset):
    def __init__(self, df, embedding_dir, pad_to_longest=0, return_slide_ids=False, inference_strategy='regression'):
        self.df = df
        self.embedding_dir = embedding_dir
        self.pad_to_longest = pad_to_longest
        self.return_slide_ids = return_slide_ids
        self.inference_strategy = inference_strategy
        assert self.inference_strategy in ['regression', 'classification']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        single_slide_df = pd.read_csv(os.path.join(self.embedding_dir, '{}.csv'.format(
            int(row.image_id))))

        # if len(single_slide_df) > 2:
        #     print('randomly sampling 2 tiles from {}'.format(int(row.image_id)))
        #     single_slide_df = single_slide_df.sample(2)

        item = single_slide_df.filter(regex='feat_', axis=1)
        assert len(item.columns) > 0
        item = torch.Tensor(item.values)

        if self.pad_to_longest:
            discrepancy = self.pad_to_longest - item.shape[0]
            item = F.pad(item, (0, 0, 0, discrepancy), mode="constant", value=0)

        if self.inference_strategy == 'classification':
            score = row.category
        else:
            score = row.Oncotype_score

        if self.return_slide_ids:
            return row.image_id, item, score
        else:
            return item, score
