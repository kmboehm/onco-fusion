import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from PIL import Image
from skimage import measure
import config


def extract_fraction_necrosis(result_dict, class_bitmaps):
    necrotic_pixels = np.sum(class_bitmaps['Necrosis'])
    total_foreground_pixels = 0
    for class_, bitmap in class_bitmaps.items():
        total_foreground_pixels += np.sum(bitmap)
    feature = float(necrotic_pixels) / total_foreground_pixels
    result_dict['fraction_area_necrotic'] = feature


def extract_ratio_necrosis_to_tumor(result_dict, class_bitmaps):
    necrotic_pixels = np.sum(class_bitmaps['Necrosis'])
    tumor_pixels = np.sum(class_bitmaps['Tumor'])
    if tumor_pixels > 0:
        feature = float(necrotic_pixels) / tumor_pixels
        result_dict['ratio_necrosis_to_tumor'] = feature
    else:
        pass


def extract_ratio_necrosis_to_stroma(result_dict, class_bitmaps):
    necrotic_pixels = np.sum(class_bitmaps['Necrosis'])
    stroma_pixels = np.sum(class_bitmaps['Stroma'])
    if stroma_pixels > 0:
        feature = float(necrotic_pixels) / stroma_pixels
        result_dict['ratio_necrosis_to_stroma'] = feature
    else:
        pass


def extract_shannon_entropy(result_dict, class_bitmaps, prefix=None):
    n = 0
    p = []
    for bitmap in class_bitmaps.values():
        count = np.sum(bitmap != 0)
        n += count
        p.append(count)
    p = np.array(p)
    p = p/n

    shannon_entropy = 0
    for prob in p:
        if prob > 0:
            shannon_entropy -= prob * np.log2(prob)

    if prefix:
        key_name = '{}_shannon_entropy'.format(prefix)
    else:
        key_name = 'shannon_entropy'

    result_dict[key_name] = shannon_entropy


def extract_tumor_stroma_entropy(result_dict, class_bitmaps_):
    class_bitmaps = {'Tumor': class_bitmaps_['Tumor'],
                     'Stroma': class_bitmaps_['Stroma']}
    return extract_shannon_entropy(result_dict, class_bitmaps, prefix='Tumor_Stroma')


def get_classwise_regionprops(result_dict, class_bitmaps):
    for class_, bitmap in class_bitmaps.items():
        features = _get_single_class_regionprops(bitmap, class_)
        result_dict.update(features)

        # get regionprops for largest connected component
        largest_cc_map = _get_single_class_largest_cc_bitmap(bitmap)
        if largest_cc_map is not None:
            features = _get_single_class_regionprops(largest_cc_map, '_'.join([class_,
                                                                       'largest_component']))
            result_dict.update(features)


def _get_single_class_regionprops(class_bitmap, class_label):
    features = {}
    properties = measure.regionprops(class_bitmap)
    if properties:
        properties = properties[0]
    else:
        return {'_'.join([class_label, 'area']): 0}

    features['_'.join([class_label, 'area'])] = properties.area
    features['_'.join([class_label, 'convex_area'])] = properties.convex_area
    features['_'.join([class_label, 'eccentricity'])] = properties.eccentricity
    features['_'.join([class_label, 'equivalent_diameter'])] = properties.equivalent_diameter
    features['_'.join([class_label, 'euler_number'])] = properties.euler_number
    features['_'.join([class_label, 'extent'])] = properties.extent
    # features['_'.join([class_label, 'feret_diameter_max'])] = properties.feret_diameter_max
    # features['_'.join([class_label, 'filled_area'])] = properties.filled_area
    features['_'.join([class_label, 'major_axis_length'])] = properties.major_axis_length
    features['_'.join([class_label, 'minor_axis_length'])] = properties.minor_axis_length
    features['_'.join([class_label, 'perimeter'])] = properties.perimeter
    # features['_'.join([class_label, 'perimeter_crofton'])] = properties.perimeter_crofton
    features['_'.join([class_label, 'solidity'])] = properties.solidity
    features['_'.join([class_label, 'PA_ratio'])] = properties.perimeter / float(properties.area)
    return features
    # print(features)
    # exit()


def _get_single_class_largest_cc_bitmap(bitmap):
    labels, n = measure.label(bitmap, return_num=True)

    largest_area = 0
    associated_label = -1
    for label in range(1, n):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            associated_label = label
    if associated_label == -1:
        return None
    else:
        return (labels == label).astype(int)


def extract_feats(dir_, slide_name, class_list):
    class_bitmaps = dict()
    for class_ in class_list:
        class_bitmaps[class_] = np.array(Image.open(os.path.join(dir_, slide_name, class_ + '.png'))).squeeze()

    result_ = dict()
    get_classwise_regionprops(result_, class_bitmaps)
    extract_fraction_necrosis(result_, class_bitmaps)
    extract_ratio_necrosis_to_tumor(result_, class_bitmaps)
    extract_ratio_necrosis_to_stroma(result_, class_bitmaps)
    extract_shannon_entropy(result_, class_bitmaps)
    extract_tumor_stroma_entropy(result_, class_bitmaps)
    return {slide_name: result_}


if __name__ == '__main__':
    checkpoint_name = config.args.checkpoint_path.split('/')[-1].replace('.torch', '')
    bitmap_dir = 'bitmaps/{}'.format(checkpoint_name)
    feat_df_filename = '{}/tissue_tile_features/{}.csv'.format(config.args.base_path, checkpoint_name)
    SERIAL = False

    slide_list = os.listdir(bitmap_dir)

    # map_key = {'Stroma': 0,
    #            'Tumor': 1,
    #            'Fat': 2,
    #            'Vessel': 3,
    #            'Necrosis': 4}
    map_key = {'Stroma': 0,
               'Tumor': 1,
               'Fat': 2,
               'Necrosis': 3}
    classes = list(map_key.keys())

    results = {}
    if SERIAL:
        for slide in slide_list:
            print(slide)
            result = extract_feats(bitmap_dir, slide, classes)
            results.update(result)
    else:
        dicts = Parallel(n_jobs=32)(delayed(extract_feats)(bitmap_dir, slide, classes) for slide in slide_list)
        for dict_ in dicts:
            results.update(dict_)

    df = pd.DataFrame(results).T
    print(df)

    df.to_csv(feat_df_filename, index=True)

