import os
import json
import yaml
import pandas as pd
import numpy as np

from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from shapely.geometry import Polygon, Point
from PIL import Image
from joblib import Parallel, delayed
from skimage.color import rgb2lab

import config
import general_utils
import itertools

from general_utils import make_otsu


def percent_otsu_score(tile):
    """
    Get percent foreground score.
    :param tile: PIL.Image (greyscale)
    :return: float [0,1] of percent foregound in tile
    """
    assert isinstance(tile, Image.Image)
    arr = np.array(tile)
    return np.mean(arr)


def purple_score(tile_):
    """
    Get percent purple score.
    :param tile_: PIL.Image (RGB)
    :return: float [0, 1] of percent purple pixels in tile
    """
    assert isinstance(tile_, Image.Image)
    tile = np.array(tile_)
    r, g, b = tile[..., 0], tile[..., 1], tile[..., 2]
    score = np.sum((r > (g + 10)) & (b > (g + 10)))
    return score / tile.size


def score_tiles(otsu_img, rgb_img, tile_size):
    """
    Get scores for tiles based on percent foreground. When tile_size and img size are downscaled
    proportionally, these coordinates map directly into the slide with proportionately upscaled
    tile_size and img size.
    :param otsu_img: np.ndarray, possibly downsampled. binary thresholded
    :param rgb_img: np.ndarray, possibly downsampled. RGB. same size as otsu_img
    :param tile_size: side length
    :return: list of (int_x, int_y) tuples
    """
    assert isinstance(otsu_img, np.ndarray) and isinstance(tile_size, int)
    otsu_slide = general_utils.array_to_slide(otsu_img)
    otsu_generator, otsu_generator_level = general_utils.get_full_resolution_generator(otsu_slide,
                                                                                       tile_size=tile_size)
    rgb_slide = general_utils.array_to_slide(rgb_img)
    rgb_generator, rgb_generator_level = general_utils.get_full_resolution_generator(rgb_slide,
                                                                                     tile_size=tile_size)

    tile_x_count, tile_y_count = otsu_generator.level_tiles[otsu_generator_level]
    address_list = []
    for count, address in enumerate(itertools.product(range(tile_x_count), range(tile_y_count))):
        if count > 1000 and len(address_list) > 10 and PROTOTYPE:
            break
        dimensions = otsu_generator.get_tile_dimensions(otsu_generator_level, address)
        if not (dimensions[0] == tile_size) or not (dimensions[1] == tile_size):
            continue

        rgb_tile = rgb_generator.get_tile(rgb_generator_level, address)
        if not purple_score(rgb_tile) > config.args.purple_threshold:
            continue

        otsu_tile = otsu_generator.get_tile(otsu_generator_level, address)
        otsu_score = percent_otsu_score(otsu_tile)

        if otsu_score < config.args.otsu_threshold:
            continue

        address_list.append(address)

    return address_list


def score_tiles_manual(ann_file_name, otsu_img, thumbnail, tile_size, overlap):
    assert isinstance(thumbnail, np.ndarray) and isinstance(tile_size, int)

    try:
        with open(ann_file_name, 'r') as f:
            slide_annotations = json.load(f)['features']
    except FileNotFoundError:
        print('Warning: {} not found'.format(ann_file_name))
        return []

    slide = general_utils.array_to_slide(thumbnail)
    generator, generator_level = general_utils.get_full_resolution_generator(slide,
                                                                             tile_size=tile_size,
                                                                             overlap=overlap)
    otsu_slide = general_utils.array_to_slide(otsu_img)
    otsu_generator, otsu_generator_level = general_utils.get_full_resolution_generator(otsu_slide,
                                                                                       tile_size=tile_size,
                                                                                       overlap=overlap)
    tile_x_count, tile_y_count = generator.level_tiles[generator_level]
    print('{}, {}'.format(tile_x_count, tile_y_count))
    address_list = []
    for count, address in enumerate(itertools.product(range(tile_x_count), range(tile_y_count))):
        if count > 1000 and len(address_list) > 10 and PROTOTYPE:
            break
        dimensions = generator.get_tile_dimensions(generator_level, address)
        assert isinstance(tile_size, int)
        if dimensions[0] != (tile_size + 2*overlap) or dimensions[1] != (tile_size + 2*overlap):
            continue

        tile_location, _level_, new_tile_size = generator.get_tile_coordinates(generator_level, address)
        assert _level_ == 0
        tile_class = is_tile_in_annotations(tile_location, new_tile_size, slide_annotations)
        if not tile_class:
            continue
        else:
            if int(tile_class) in [3, 4, 6, 7]:
                otsu_tile = otsu_generator.get_tile(otsu_generator_level, address)
                otsu_score = percent_otsu_score(otsu_tile)

                if otsu_score < config.args.otsu_threshold:
                    continue
            address_list.append([address, tile_class])
    return address_list


def is_tile_in_annotations(tile_location, tile_size, slide_annotations):
    """
    Determine whether tile is in annotations, and if so, what class of annotation.
    :param tile_location:
    :param tile_size:
    :param slide_annotations:
    :return: 0 if not in annotation, else annotation index of first region that tile falls in
    """
    points = [Point(tile_location[0], tile_location[1]),
              Point(tile_location[0] + tile_size[0], tile_location[1]),
              Point(tile_location[0], tile_location[1] + tile_size[1]),
              Point(tile_location[0] + tile_size[0], tile_location[1] + tile_size[1])]
    for annotation_ in slide_annotations:
        point_count = 0
        class_ = annotation_['properties']['label_num']
        assert annotation_['geometry']['type'] == 'Polygon'

        coords = annotation_['geometry']['coordinates']
        if len(coords) >= 3:
            annotation = Polygon(coords)
            for point in points:
                if annotation.contains(point):
                    point_count += 1
            if point_count > 3:
                return class_
    return 0


def get_slide_tile_addresses(image_path, mag, scale, desired_tile_selection_size, index, tile_selection, overlap, annotation_path=None, visualize=False):
    assert isinstance(overlap, int)

    slide = OpenSlide(image_path)

    slide_mag = general_utils.get_magnification(slide)
    scale = general_utils.adjust_scale_for_slide_mag(slide_mag=slide_mag, desired_mag=mag, scale=scale)
    thumbnail = general_utils.get_downscaled_thumbnail(slide, scale)
    overlap = int(overlap // scale)
    otsu_thumbnail = make_otsu(thumbnail)
    if tile_selection == 'otsu':
        assert config.args.overlap == 0
        tile_addresses = score_tiles(otsu_thumbnail,
                                     thumbnail,
                                     tile_size=desired_tile_selection_size)
        tile_addresses = [[x, -1] for x in tile_addresses]  # for consistent formatting with manual
    elif tile_selection == 'manual':
        assert annotation_path is not None
        tile_addresses = score_tiles_manual(annotation_path,
                                            otsu_thumbnail,
                                            thumbnail,
                                            tile_size=desired_tile_selection_size,
                                            overlap=overlap)
    else:
        raise RuntimeError

    if visualize:
        thumbnail = general_utils.visualize_tiling(thumbnail,
                                                   desired_tile_selection_size,
                                                   tile_addresses,
                                                   overlap=overlap)
        thumbnail = Image.fromarray(thumbnail)
        thumbnail.save('tiling_visualizations/{}_tiling.png'.format(image_path.split('/')[-1]))
    return {index: tile_addresses}


if __name__ == '__main__':
    PROTOTYPE = False
    visualize = False
    serial = False

    scale_factor = config.args.tile_size / config.desired_otsu_thumbnail_tile_size

    df = pd.read_csv(config.args.cohort_csv_path)
    with open('../global_config.yaml', 'r') as f:
        DIRECTORIES = yaml.safe_load(f)
        DATA_DIR = DIRECTORIES['data_dir']
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(DATA_DIR, x))
    if 'segmentation_path' in df.columns:
        df['segmentation_path'] = df['segmentation_path'].apply(lambda x: os.path.join(DATA_DIR, x))

    if PROTOTYPE:
        df = df.head(4)

    coords = {}
    if serial:
        for index, row in df.iterrows():
            print(row['image_path'])
            tile_addresses = get_slide_tile_addresses(row['image_path'],
                                                      mag=config.args.magnification,
                                                      scale=scale_factor,
                                                      desired_tile_selection_size=
                                                      config.desired_otsu_thumbnail_tile_size,
                                                      index=index,
                                                      tile_selection=config.args.tile_selection_type,
                                                      overlap=config.args.overlap,
                                                      visualize=visualize,
                                                      annotation_path=row['segmentation_path'] if 'segmentation_path' in df.columns else None)
            coords.update(tile_addresses)
    else:
        _dicts = Parallel(n_jobs=64)(delayed(get_slide_tile_addresses)(row['image_path'],
                                                      mag=config.args.magnification,
                                                      scale=scale_factor,
                                                      desired_tile_selection_size=
                                                      config.desired_otsu_thumbnail_tile_size,
                                                      index=index,
                                                      tile_selection=config.args.tile_selection_type,
                                                      overlap=config.args.overlap,
                                                                       visualize=visualize,
                                                      annotation_path=row['segmentation_path'] if 'segmentation_path' in df.columns else None)
                                    for index, row in df.iterrows())
        for _dict in _dicts:
            coords.update(_dict)

    df['tile_address'] = pd.Series(coords)
    df['n_foreground_tiles'] = df['tile_address'].map(len)

    df.to_csv(config.args.preprocessed_cohort_csv_path, index=False)
