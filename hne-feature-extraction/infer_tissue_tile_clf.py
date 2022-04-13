import pandas as pd
import numpy as np
from models import load_tissue_tile_net
from train_tissue_tile_clf import prep_df, make_preds
import config
import torch.nn as nn
from dataset import TissueTileDataset
import torch
import csv
import os
import general_utils
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from general_utils import setup


def make_preds_by_slide(model, df, device, file_dir, n_classes):
    header = ['tile_file_name', 'label']
    header.extend(['score_{}'.format(k) for k in range(n_classes)])

    for image_id, sub_df in df.groupby('image_id'):
        dataset = TissueTileDataset(df=sub_df,
                                        tile_dir=config.args.tile_dir,
                                        transforms=general_utils.get_val_transforms())
        loader = DataLoader(dataset,
                                batch_size=config.args.batch_size,
                                num_workers=4)

        file_name = os.path.join(file_dir, str(image_id) + '.csv')

        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header)

            with torch.no_grad():
                for ids, tiles, labels in loader:
                    preds = model(tiles.to(device))
                    preds = preds.detach().cpu().tolist()
                    for idx, label, pred_list in zip(ids, labels.tolist(), preds):
                        row = [idx, label]
                        row.extend(pred_list)
                        writer.writerow(row)


def get_fold_slides(df, world_size, rank):
    all_slides = df.image_id.unique()
    chunks = np.split(all_slides, world_size)
    return chunks[rank]


def distribute(rank, world_size, df_, n_classes, val_dir):
    setup(rank, world_size)
    device_ids = [config.args.gpu[rank]]
    device = torch.device('cuda:{}'.format(device_ids[0]))
    print('distributed to device {}'.format(str(device)))

    df = df_[df_.image_id.isin(get_fold_slides(df_, world_size, rank))]

    model = load_tissue_tile_net(config.args.checkpoint_path, activation=nn.Softmax(), n_classes=n_classes)
    model.to(device)
    model.eval()

    make_preds_by_slide(model,
                        df,
                        device,
                        val_dir,
                        n_classes)


def serialize(df, n_classes, val_dir):
    model = load_tissue_tile_net(config.args.checkpoint_path, activation=nn.Softmax(), n_classes=n_classes)

    device = torch.device('cuda:{}'.format(config.args.gpu[0]))
    model.to(device)
    model.eval()

    make_preds_by_slide(model,
                        df,
                        device,
                        val_dir,
                        n_classes)


if __name__ == '__main__':
    assert config.args.checkpoint_path

    checkpoint_name = config.args.checkpoint_path.split('/')[-1].replace('.torch', '')
    inference_dir = 'inference/{}'.format(checkpoint_name)
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    world_size_ = len(config.args.gpu)
    df, n_classes, _, _ = prep_df(config.args.preprocessed_cohort_csv_path, map_classes=False)

    if world_size_ == 1:
        serialize(df, n_classes, inference_dir)
    else:
        mp.spawn(distribute,
                 args=(world_size_, df, n_classes, inference_dir),
                 nprocs=world_size_,
                 join=True)
