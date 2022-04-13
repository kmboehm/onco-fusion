from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import compute_class_weight

import numpy as np
import csv
import pickle
import config
import general_utils
import torch
import re
import pandas as pd

import models
from dataset import TissueTileDataset
from general_utils import get_starting_timestamp, get_checkpoint_path, get_train_transforms, get_val_transforms, \
    log_results_string
from models import TissueTileNet


def train_epoch(model, train_loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    y_pred = []
    y_true = []
    n = len(train_loader.dataset)
    for idx, tiles, labels in train_loader:
        # forward
        optimizer.zero_grad()
        output = model(tiles.to(device))

        # calculate loss
        loss = criterion(input=output, target=labels.long().to(device))
        loss.backward()

        # backward
        optimizer.step()
        total_loss += loss.detach().cpu().item()

        # keep track of true and predicted classes
        y_pred.extend(output.argmax(1).detach().cpu().reshape(-1).numpy())
        y_true.extend(labels.reshape(-1).numpy())

    total_loss /= n
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    confusion = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return total_loss, acc, f1, confusion


def validate_epoch(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    n = len(val_loader.dataset)
    with torch.no_grad():
        for idx, tiles, labels in val_loader:
            # forward
            output = model(tiles.to(device))

            # calculate loss
            loss = criterion(input=output, target=labels.long().to(device))
            total_loss += loss.detach().cpu().item()

            # keep track of true and predicted classes
            y_pred.extend(output.argmax(1).detach().cpu().reshape(-1).numpy())
            y_true.extend(labels.reshape(-1).numpy())

    total_loss /= n
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    confusion = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return total_loss, acc, f1, confusion


def make_preds(model, loader, device, file_name, n_classes):
    header = ['tile_file_name', 'label']
    header.extend(['score_{}'.format(k) for k in range(n_classes)])
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(header)

        model.eval()
        with torch.no_grad():
            for ids, tiles, labels in loader:
                preds = model(tiles.to(device))
                preds = preds.detach().cpu().tolist()
                for idx, label, pred_list in zip(ids, labels.tolist(), preds):
                    row = [idx, label]
                    row.extend(pred_list)
                    writer.writerow(row)


def serialize(device_id, df, starting_timestamp, fold):
    starting_epoch = 0
    device = torch.device('cuda:{}'.format(device_id))
    train_df = df[df.split == 'train'].copy()
    train_df.loc[:, 'long_tile_file_name'] = train_df.img_hid.str.replace('.svs',
                                                                              '') + '/' + train_df.tile_file_name
    assert 'val' not in train_df.split
    val_df = df[df.split == 'val'].copy()
    assert 'train' not in val_df.split

    print('train ({}):'.format(len(train_df)))
    print(train_df.tile_class.value_counts())
    print('val ({}):'.format(len(val_df)))
    print(val_df.tile_class.value_counts())
    do_validation = len(val_df) > 0

    train_dataset = TissueTileDataset(df=train_df,
                                      tile_dir=config.args.tile_dir,
                                      transforms=get_train_transforms(normalize=True))
    train_loader = DataLoader(train_dataset,
                                  batch_size=config.args.batch_size,
                                  num_workers=8,
                                  pin_memory=False,
                                  shuffle=True)

    if do_validation:
        val_dataset = TissueTileDataset(df=val_df,
                                            tile_dir=config.args.tile_dir,
                                            transforms=get_val_transforms(normalize=True))
        val_loader = DataLoader(val_dataset,
                                    batch_size=config.args.batch_size,
                                    num_workers=8)

    model = TissueTileNet(model=models.get_model(config), n_classes=n_classes)
    model.to(device)

    optimizer = Adam(model.parameters(),
                         lr=config.args.learning_rate,
                         weight_decay=config.args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(compute_class_weight(
        class_weight='balanced',
        classes=np.sort(train_df.tile_class.unique()),
        y=train_df.tile_class)).to(device).float())

    for epoch in range(config.args.num_epochs):
        train_loss, train_acc, train_f1, train_confusion = train_epoch(model,
                                                                       train_loader,
                                                                       optimizer,
                                                                       device,
                                                                       criterion)
        if do_validation:
            val_loss, val_acc, val_f1, val_confusion = validate_epoch(model,
                                                                      val_loader,
                                                                      device,
                                                                      criterion)
        else:
            val_loss, val_acc, val_f1, val_confusion = -1, -1, -1, -1

        log_results_string(epoch, starting_epoch, train_loss, val_loss, starting_timestamp, fold)
        results_str = 'training acc {:7.6f} | validation acc {:7.6f}'.format(train_acc, val_acc)
        print(results_str)
        results_str = 'training f1 {:7.6f} | validation f1 {:7.6f}'.format(train_f1, val_f1)
        print(results_str)
        print('training:\n' + str(train_confusion))
        print('validation:\n' + str(val_confusion))
        print('---')
        torch.save(model.state_dict(), get_checkpoint_path(starting_timestamp,
                                                                   epoch, starting_epoch,
                                                                   fold=fold))


def prep_df(csv_path, map_classes=True):
    # load dataframe
    df_ = pd.read_csv(csv_path, low_memory=False)
    df_ = df_[df_.x > 0]
    df_.tile_address = df_.tile_address.map(eval)
    df_ = df_.explode('tile_address').reset_index()
    df_['tile_class'] = df_.tile_address.apply(lambda x: x[1])
    df_['tile_address'] = df_.tile_address.apply(lambda x: x[0])

    slideviewer_class_map = {1: 'Stroma',
                             2: 'Stroma',
                             3: 'Tumor',
                             4: 'Tumor',
                             5: 'Fat',
                             6: 'Vessel',
                             7: 'Vessel',
                             10: 'Necrosis',
                             # 11: 'Glass',
                             14: 'Pen'}
    map_key = {'Stroma': 0,
               'Tumor': 1,
               'Fat': 2,
               'Necrosis': 3}
               # 'Vessel': 3,
               # 'Necrosis': 4}
               # 'Glass': 5,
               # 'Pen': 5}

    n_classes = len(map_key)
    map_reverse_key = dict([(v, k) for k, v in map_key.items()])

    print(df_.tile_class.value_counts())
    if map_classes:
        df_['tile_class'] = df_['tile_class'].map(slideviewer_class_map).map(map_key)
        df_ = df_[df_['tile_class'].isin(map_key.values())]
    print(df_.tile_class.value_counts())
    # exit()

    if 'img_hid' not in df_.columns:
        df_['img_hid'] = df_['image_id'].astype(str) + '.svs'

    df_['tile_file_name'] = df_.apply(
        lambda x: general_utils.get_tile_file_name('---', x.img_hid, x.tile_address),
        axis=1).str.split('/').map(lambda x: x[-1])
    return df_, n_classes, map_key, map_reverse_key


if __name__ == '__main__':
    starting_timestamp_ = get_starting_timestamp(config)
    print(starting_timestamp_)

    seed = int(re.sub('[^0-9]', '', starting_timestamp_[5:]))

    df_, n_classes, map_key, map_reverse_key = prep_df(config.args.preprocessed_cohort_csv_path)

    with open('checkpoints/{}_config.pickle'.format(starting_timestamp_), 'wb') as f:
        pickle.dump(config.args, f)

    if config.args.crossval > 0:
        for fold, df in enumerate(general_utils.k_fold_ptwise_crossval(df_, config.args.crossval, seed)):
            print('\nFOLD {}'.format(fold))
            serialize(config.args.gpu[0], df, starting_timestamp_, fold)
    elif config.args.crossval == -2:
        print('warning: no validation set')
        df_.loc[:, 'split'] = 'train'
        serialize(config.args.gpu[0], df_, starting_timestamp_, -2)
