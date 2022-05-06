import colorsys
import re

from openslide import OpenSlide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torch import distributed as dist

import numpy as np
import pandas as pd
import torch
import os
from random import choice
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import rectangle_perimeter, rectangle
from skimage import color
from copy import deepcopy
from datetime import datetime
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


def get_magnification(slide):
    return int(slide.properties['aperio.AppMag'])


def get_downscaled_thumbnail(slide, scale_factor=32):
    new_width = slide.dimensions[0] // scale_factor
    new_height = slide.dimensions[1] // scale_factor
    img = slide.get_thumbnail((new_width, new_height))
    return np.array(img)


def get_full_resolution_generator(slide, tile_size, overlap=0, level_offset=0):
    assert isinstance(slide, OpenSlide) or isinstance(slide, ImageSlide)
    generator = DeepZoomGenerator(slide, overlap=overlap, tile_size=tile_size, limit_bounds=False)
    generator_level = generator.level_count - 1 - level_offset
    if level_offset == 0:
        assert generator.level_dimensions[generator_level] == slide.dimensions
    return generator, generator_level


def adjust_scale_for_slide_mag(slide_mag, desired_mag, scale):
    if slide_mag != desired_mag:
        if slide_mag < desired_mag:
            raise AssertionError('expected mag >={} but got {}'.format(desired_mag, slide_mag))
        elif (slide_mag / desired_mag) == 2:
            scale *= 2
        elif (slide_mag / desired_mag) == 4:
            scale *= 4
        else:
            raise AssertionError('expected mag {} or {} but got {}'.format(desired_mag, 2 * desired_mag, slide_mag))
    return scale


def visualize_tiling(_thumbnail, tile_size, tile_addresses, overlap=0):
    """
    Draw black boxes around tiles passing threshold
    :param _thumbnail: np.ndarray
    :param tile_size: int
    :param tile_addresses:
    :return: new thumbnail image with black boxes around tiles passing threshold
    """
    assert isinstance(_thumbnail, np.ndarray) and isinstance(tile_size, int)
    thumbnail = deepcopy(_thumbnail)
    generator, generator_level = get_full_resolution_generator(array_to_slide(thumbnail),
                                                               tile_size=tile_size,
                                                               overlap=overlap)

    for address in tile_addresses:
        if isinstance(address, list):
            address = address[0]
        extent = generator.get_tile_dimensions(generator_level, address)
        start = (address[1] * tile_size, address[0] * tile_size)  # flip because OpenSlide uses
                                                                  # (column, row), but skimage
                                                                  # uses (row, column)
        rr, cc = rectangle_perimeter(start=start, extent=extent, shape=thumbnail.shape)
        thumbnail[rr, cc] = 1

    return thumbnail


def colorize(image, hue, saturation=1.0):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    rgb = (color.hsv2rgb(hsv) * 255).astype(int)
    return rgb


def visualize_tile_scoring(_thumbnail, tile_size, tile_addresses, tile_scores, overlap=0, range_=[-3, 3]):
    """
    Draw black boxes around tiles passing threshold
    :param _thumbnail: np.ndarray
    :param tile_size: int
    :param tile_addresses:
    :return: new thumbnail image with black boxes around tiles passing threshold
    """
    denom = 2 * (range_[1] - range_[0])
    assert isinstance(_thumbnail, np.ndarray) and isinstance(tile_size, int)
    thumbnail = deepcopy(_thumbnail)
    generator, generator_level = get_full_resolution_generator(array_to_slide(thumbnail),
                                                               tile_size=tile_size,
                                                                   overlap=overlap)

    for address, score in zip(tile_addresses, tile_scores):
        extent = generator.get_tile_dimensions(generator_level, address)
        start = (address[1] * tile_size, address[0] * tile_size)  # flip because OpenSlide uses
                                                                  # (column, row), but skimage
                                                                  # uses (row, column)

        rr, cc = rectangle(start=start, extent=extent, shape=thumbnail.shape)
        thumbnail[rr, cc] = colorize(thumbnail[rr, cc], hue=0.5-(score-range_[0])/denom, saturation=0.5)

    return thumbnail


def array_to_slide(arr):
    assert isinstance(arr, np.ndarray)
    slide = ImageSlide(Image.fromarray(arr))
    return slide


def get_current_time():
    return str(datetime.now()).replace(' ', '_').split('.')[0].replace(':', '.')


def load_preprocessed_df(file_name, min_n_tiles, cols=None, seed=None, explode=True):
    if cols:
        df_ = pd.read_csv(file_name, low_memory=False, usecols=cols)
    else:
        df_ = pd.read_csv(file_name, low_memory=False)

    df_ = df_[df_.n_foreground_tiles >= min_n_tiles]

    df_.tile_address = df_.tile_address.map(eval)
    if explode:
        df_ = df_.explode('tile_address')
        df_['tile_file_name'] = df_.tile_address.apply(
            lambda x: str(x).split(')')[0].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(', ', '_') + '.png')
        #     lambda x: get_tile_file_name('---', x.img_hid, x.tile_address),
        #     axis=1).str.split('/').map(lambda x: x[-1])

    return df_


def k_fold_ptwise_crossval(df, k, seed):
    if 'observed' in df.columns:
        if 'Patient ID' in df.columns:
            temp_df = df.groupby('Patient ID').agg('mean').observed
        else:
            temp_df = df.groupby('image_path').agg('mean').observed
            
        patient_ids = np.array(temp_df.index.tolist())
        observed = np.array(temp_df.tolist())

        kf = StratifiedKFold(n_splits=k, random_state=seed % (2**32 - 1), shuffle=True).split(patient_ids, observed)
    else:
        if 'Patient ID' in df.columns:
            patient_ids = np.sort(df['Patient ID'].unique())
        else:
            patient_ids = np.sort(df['image_path'].unique())

        kf = KFold(n_splits=k, random_state=seed % (2**32 - 1), shuffle=True).split(patient_ids)

    df_list = []
    for train_indices, test_indices in kf:
        train_labels = patient_ids[train_indices]
        test_labels = patient_ids[test_indices]
        # train_labels = list(set(DF.index[train_indices].tolist()))
        # test_labels = list(set(DF.index.tolist()) - set(train_labels))
        if 'Patient ID' in df.columns:
            train_mask = df['Patient ID'].isin(train_labels)
            test_mask = df['Patient ID'].isin(test_labels)
        else:
            train_mask = df['image_path'].isin(train_labels)
            test_mask = df['image_path'].isin(test_labels)

        assert test_mask.sum() > 0
        assert train_mask.sum() > 0

        DF = df.copy(deep=True)
        DF.loc[train_mask, 'split'] = 'train'
        DF.loc[test_mask, 'split'] = 'val'
        df_list.append(DF)
    return df_list


def get_slide_dir(_dir, slide_file_name):
    slide_stem = slide_file_name[:-4]
    return os.path.join(_dir, slide_stem)


def get_tile_file_name(_dir, slide_file_name, address):
    address_suffix = str(address).replace('(', '').replace(')', '').replace(', ', '_')
    file_name = address_suffix + '.png'
    return os.path.join(get_slide_dir(_dir, slide_file_name), file_name)


def load_ddp_state_dict_to_device(path, device='cpu', ddp_to_serial=True):
    assert isinstance(device, str)
    ddp_state_dict = torch.load(path, map_location=device)
    if ddp_to_serial:
        state_dict = {}
        for key, value in ddp_state_dict.items():
            state_dict[key.replace('module.', '')] = value
        return state_dict
    else:
        return ddp_state_dict


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    try:
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except RuntimeError:
        os.environ['MASTER_PORT'] = '1234'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print('device {} initialized'.format(rank))


def cleanup():
    dist.destroy_process_group()


def get_starting_timestamp(config):
    if config.args.checkpoint_path:
        short_path = config.args.checkpoint_path.split('/')[-1]
        starting_timestamp_ = re.search('^.*(?=(\_epoch))', short_path).group(0)
        if '_fold' in starting_timestamp_:
            starting_timestamp_ = starting_timestamp_.split('_fold')[0]
    else:
        starting_timestamp_ = get_current_time()
    return starting_timestamp_


def load_model_state_dict(model, checkpoint_path, device='cpu'):
    assert isinstance(device, str)
    model.load_state_dict(load_ddp_state_dict_to_device(
            os.path.join('checkpoints', checkpoint_path), device=device))


def get_starting_epoch(config):
    if config.args.checkpoint_path:
        starting_epoch = int(re.search('epoch(\d+)', config.args.checkpoint_path).group(1))
    else:
        starting_epoch = 1
    return starting_epoch


def log_results_string(epoch, starting_epoch, train_loss, val_loss, starting_timestamp, fold, other_keys=dict()):
    epoch_str = get_epoch_str(epoch, starting_epoch)
    results_str = 'epoch {}: train loss {:.3e} | val loss {:.3e}'.format(
        epoch_str, train_loss, val_loss)
    if other_keys:
        other_keys = list(other_keys.items())
        other_keys.sort(key=lambda x: x[0])
        for key, val in other_keys:
            results_str += ' | {} {:.3e}'.format(key, val)
    print(results_str)
    with open('checkpoints/{}_fold{}_log.txt'.format(starting_timestamp, fold), 'a+') as file:
        file.write(results_str + '\n')


def get_train_transforms(smaller_dim=None, normalize=True):
    l = []
    if smaller_dim:
        l.append(transforms.RandomCrop(size=smaller_dim))

    l.extend([
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.5),
                                        transforms.ColorJitter(brightness=0.1,
                                                               contrast=0.1,
                                                               saturation=0.05,
                                                               hue=0.01),
                                        transforms.ToTensor()]
                                        )
    if normalize:
        l.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(l)


def get_val_transforms(smaller_dim=None, normalize=True):
    l = []
    if smaller_dim:
        l.append(transforms.RandomCrop(size=smaller_dim))
    l.append(transforms.ToTensor())
    if normalize:
        l.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(l)


def get_epoch_str(epoch, starting_epoch):
    return str(epoch + 1 + starting_epoch).zfill(3)


def get_checkpoint_path(starting_timestamp, epoch, starting_epoch, fold):
    epoch_str = get_epoch_str(epoch, starting_epoch)
    return 'checkpoints/{}_fold{}_epoch{}.torch'.format(starting_timestamp, fold, epoch_str)


def make_otsu(img, scale=1):
    """
    Make image with pixel-wise foreground/background labels.
    :param img: grayscale np.ndarray
    :return: np.ndarray where each pixel is 0 if background and 1 if foreground
    """
    assert isinstance(img, np.ndarray)
    _img = rgb2gray(img)
    threshold = threshold_otsu(_img)
    return (_img < (threshold * scale)).astype(float)


def label_image_tissue_type(thumbnail, map_key):
    """
    Labels tissue with hue overlay based on predicted classes.
    """
    vals = list(map_key.values())
    colors = []
    range_ = [np.min(vals), np.max(vals)]
    denom = 2 * (range_[1] - range_[0])
    for class_, score in map_key.items():
        colors.append(tuple([int(255 * x) for x in
                             colorsys.hsv_to_rgb(0.5 - (score - range_[0]) / denom, 0.5, 1.0)]))
    d = ImageDraw.Draw(thumbnail)

    text_locations = [(10, 10+40*x) for x in range(len(map_key))]
    for (class_, score), text_location in zip(map_key.items(), text_locations):
        d.text(text_location, class_, fill=colors[score])
    return thumbnail


def get_fold_slides(df, world_size, rank):
    all_slides = df.image_id.unique()
    chunks = np.array_split(all_slides, world_size)
    return chunks[rank]


def add_scale_bar(thumbnail, scale, slide_mag, len_in_um=1000):
    if slide_mag == 20:
        um_per_pix = 0.5
    elif slide_mag == 40:
        um_per_pix = 0.25
    else:
        raise RuntimeError("Unhandled slide mag {}x".format(slide_mag))

    um_per_pix *= scale

    len_in_pixels = len_in_um / float(um_per_pix)

    endpoints = [(10, 10+40*6), (10+len_in_pixels, 10+40*6)]
    d = ImageDraw.Draw(thumbnail)
    d.line(endpoints, fill='black', width=5)
    return thumbnail
