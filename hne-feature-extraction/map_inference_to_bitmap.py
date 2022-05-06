import pandas as pd
import numpy as np
import os
import yaml
from joblib import Parallel, delayed
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from skimage.draw import rectangle_perimeter, rectangle
from PIL import Image
import sys
sys.path.append('../tissue-type-training')
import general_utils
import config

def convert_to_bitmap(slide_path, bitmap_dir, inference_dir, scale, map_key):
    slide_id = slide_path.split('/')[-1][:-4]
    slide_bitmap_subdir = os.path.join(bitmap_dir, slide_id)
    if not os.path.exists(slide_bitmap_subdir):
        os.mkdir(slide_bitmap_subdir)
    map_reverse_key = dict([(v, k) for k, v in map_key.items()])

    # load thumbnail
    slide = OpenSlide(slide_path)

    slide_mag = general_utils.get_magnification(slide)
    scale = general_utils.adjust_scale_for_slide_mag(slide_mag=slide_mag,
                                                     desired_mag=config.args.magnification,
                                                     scale=scale)
    thumbnail = general_utils.get_downscaled_thumbnail(slide, scale)
    overlap = int(config.args.overlap // scale)

    # create bitmaps
    bitmaps = {}
    for key, val in map_key.items():
        bitmaps[key] = np.zeros(thumbnail.shape[:2], dtype=np.uint8)

    # load tile class inference csv, create tile_address and predicted_class column
    df = pd.read_csv(os.path.join(inference_dir, slide_id + '.csv'))
    df['predicted_class'] = df.drop(columns=['label', 'tile_file_name']).idxmax(axis='columns').str.replace(
        'score_', '').astype(int)
    df['address'] = df['tile_file_name'].apply(
        lambda x: [int(y) for y in x.replace('.png', '').split('/')[1].split('_')])
    # for each tile, populate the associated area in the bitmap with pred_class

    generator, generator_level = general_utils.get_full_resolution_generator(
        general_utils.array_to_slide(thumbnail),
        tile_size=config.desired_otsu_thumbnail_tile_size,
        overlap=overlap)

    for address, class_number in zip(df.address, df.predicted_class):
        extent = generator.get_tile_dimensions(generator_level, address)
        start = (address[1] * config.desired_otsu_thumbnail_tile_size,
                 address[0] * config.desired_otsu_thumbnail_tile_size)

        class_label = map_reverse_key[class_number]
        _thumbnail = bitmaps[class_label]
        rr, cc = rectangle(start=start, extent=extent, shape=_thumbnail.shape)
        _thumbnail[rr, cc] = 255

    # save bitmaps
    for class_label, bitmap in bitmaps.items():
        _thumbnail = Image.fromarray(bitmap)
        _thumbnail.save(os.path.join(slide_bitmap_subdir, class_label + '.png'))

    # generate and save overlay
    vals = list(map_key.values())
    range_ = [np.min(vals), np.max(vals)]
    thumbnail = general_utils.visualize_tile_scoring(thumbnail,
                                                     config.desired_otsu_thumbnail_tile_size,
                                                     df.address.tolist(),
                                                     df.predicted_class.tolist(),
                                                     overlap=overlap,
                                                     range_=range_)
    thumbnail = Image.fromarray(thumbnail)
    thumbnail = general_utils.label_image_tissue_type(thumbnail, map_key)
    thumbnail.save(os.path.join(slide_bitmap_subdir, '_overlay.png'))


if __name__ == '__main__':
    checkpoint_name = config.args.checkpoint_path.split('/')[-1].replace('.torch', '')
    inference_dir = 'inference/{}'.format(checkpoint_name)
    bitmap_dir = 'bitmaps/{}'.format(checkpoint_name)
    if not os.path.exists(bitmap_dir):
        os.mkdir(bitmap_dir)

    df = pd.read_csv(config.args.preprocessed_cohort_csv_path)
    with open('../global_config.yaml', 'r') as f:
        DIRECTORIES = yaml.safe_load(f)
        DATA_DIR = DIRECTORIES['data_dir']
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(DATA_DIR, x))

    scale_factor = config.args.tile_size / config.desired_otsu_thumbnail_tile_size

    map_key = {'Stroma': 0,
               'Tumor': 1,
               'Fat': 2,
               'Necrosis': 3}

    for slide_path in df['image_path']:
        print(slide_path)
        convert_to_bitmap(slide_path, bitmap_dir, inference_dir, scale_factor, map_key)

    # Parallel(n_jobs=32)(delayed(convert_to_bitmap)(slide, bitmap_dir, inference_dir, scale_factor, map_key) for slide in slide_list)
