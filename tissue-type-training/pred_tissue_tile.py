import torch
import os
import re

from torch.utils.data import DataLoader

import config
import general_utils
from dataset import TissueTileDataset
from train_tissue_tile_clf import make_preds, prep_df
from models import TissueTileNet, get_model

if __name__ == '__main__':
    assert config.args.checkpoint_path
    assert len(config.args.gpu) == 1

    device_str = 'cuda:{}'.format(config.args.gpu[0])
    device = torch.device(device_str)

    num_workers = 8
    transforms = general_utils.get_val_transforms()
    seed = 1123011750

    df_, n_classes, map_key, map_reverse_key = prep_df(config.args.preprocessed_cohort_csv_path,
                                                       tile_dir=config.args.tile_dir)
    fold = int(config.args.checkpoint_path.split('fold')[1][0])
    df = general_utils.k_fold_ptwise_crossval(df_, config.args.crossval, seed)[fold]

    model = TissueTileNet(model=get_model(config),
                          n_classes=n_classes,
                          activation=torch.nn.Softmax(dim=1))
    model.load_state_dict(general_utils.load_ddp_state_dict_to_device(config.args.checkpoint_path))
    model.to(device)


    print('making val preds for fold {}'.format(fold))
    val_dataset = TissueTileDataset(df=df[df.split == 'val'],
                                        tile_dir=config.args.tile_dir,
                                        transforms=transforms)
    assert len(val_dataset) > 0
    val_loader = DataLoader(val_dataset,
                                batch_size=config.args.batch_size,
                                num_workers=num_workers)
    make_preds(model,
                   val_loader,
                   device,
                   config.args.val_pred_file,
                   n_classes)
