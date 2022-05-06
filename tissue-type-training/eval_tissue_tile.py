from sklearn.metrics import average_precision_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import config
import json
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from PIL import Image
import os
import general_utils
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from confusion_matrix_analysis import cm_analysis
from general_utils import label_image_tissue_type, add_scale_bar


FONTSIZE = 7
plt.rc('legend',fontsize=FONTSIZE, title_fontsize=FONTSIZE)
plt.rc('xtick',labelsize=FONTSIZE)
plt.rc('ytick',labelsize=FONTSIZE)
plt.rc("axes", labelsize=FONTSIZE)


def get_auprc(df):
    try:
        assert 'score_1' in df.columns
        assert 'label' in df.columns
    except AssertionError:
        raise AssertionError('label and pred_score not in {}'.format(df.columns))
    preds = df['score_1'].tolist()
    is_hrd = df['label'].tolist()

    auprc = average_precision_score(y_true=is_hrd, y_score=preds)
    return {'auprc': auprc}


def get_random_auprc_from_df(df):
    random_df = df.copy(deep=True)
    random_df['pred_score'] = np.random.permutation(random_df['pred_score'].values)
    return {'random_auprc': get_auprc(random_df)}


def get_accuracy(df):
    return {'accuracy': accuracy_score(y_true=df.label, y_pred=df.predicted_class)}


def get_all_single_class_auprc_values(df):
    d = {}
    for class_ in df.label.unique():
        class_ = int(class_)
        #print('class {}'.format(class_))
        is_truly_class = df['label'] == class_
        df.loc[is_truly_class, 'temp_binary_truth'] = 1
        df.loc[~is_truly_class, 'temp_binary_truth'] = 0

        df['temp_binary_pred'] = df['score_{}'.format(class_)]
        d['auprc_{}'.format(class_)] = average_precision_score(y_true=df['temp_binary_truth'],
                                                               y_score=df['temp_binary_pred'])
    df.drop(columns=['temp_binary_pred', 'temp_binary_truth'], inplace=True)
    return d


def get_confusion_matrix(df):
    raise NotImplementedError
    return {'confusion_matrix': confusion_matrix(y_true=df.label, y_pred=df.predicted_class)}


def visualize(df, pred_file):
    sub_dir = os.path.join('visualizations', pred_file.split('/')[-1].replace('.csv', ''))
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    if len(df) == 0:
        return

    for image_path, sub_df in df.groupby('image_path'):
        _visualize(image_path, sub_df, sub_dir)


def _visualize(image_path, sub_df, sub_dir):
    desired_otsu_thumbnail_tile_size = 8 # 16
    scale_factor = config.args.tile_size / desired_otsu_thumbnail_tile_size
    print(image_path)
    try:
        slide = OpenSlide(image_path)
    except OpenSlideUnsupportedFormatError:
        print(image_path)
        exit()
    slide_mag = general_utils.get_magnification(slide)
    if slide_mag != config.args.magnification:
        if (slide_mag / config.args.magnification) == 2:
            scale = scale_factor * 2
        elif (slide_mag / config.args.magnification) == 4:
            scale = scale_factor * 4
        else:
            raise AssertionError('Invalid scale')
    else:
        scale = scale_factor
    thumbnail = general_utils.get_downscaled_thumbnail(slide, scale)
    sub_df['address'] = sub_df['tile_file_name'].apply(
        lambda x: [int(y) for y in x.replace('.png', '').split('/')[1].split('_')])

    # print(sub_df)
    thumbnail = general_utils.visualize_tile_scoring(thumbnail,
                                                     desired_otsu_thumbnail_tile_size,
                                                     sub_df.address.tolist(),
                                                     sub_df.predicted_class.tolist(),
                                                     overlap=int(config.args.overlap//scale_factor),
                                                     range_=[0, 3])
    thumbnail = Image.fromarray(thumbnail)
    thumbnail = label_image_tissue_type(thumbnail, map_reverse_key)
    thumbnail = add_scale_bar(thumbnail, scale, slide_mag)
    thumbnail.save('{}/{}_{}_eval.png'.format(sub_dir, image_path.split('/')[-1], set_name))


if __name__ == '__main__':
    all_preds = []
    all_acc = []
    for set_name, _pred_file in zip(['val'], [config.args.val_pred_file]):
        for fold in range(config.args.crossval):
            pred_file = _pred_file.format(fold)
            preds = pd.read_csv(pred_file)
            preds = preds.set_index('tile_file_name')
            n_classes = 4

            results = {}

            if n_classes == 2:
                results.update(get_auprc(preds))
                results.update(get_random_auprc_from_df(preds))
            print(preds)
            preds['predicted_class'] = preds.drop(columns='label').idxmax(axis='columns').str.replace(
                'score_', '').astype(int)
            preds['certainty'] = preds.drop(columns=['label', 'predicted_class']).max(axis='columns')

            results.update(get_all_single_class_auprc_values(preds))
            results.update(get_accuracy(preds))
            all_acc.append(results['accuracy'])
            
            map_key = {0: 'Stroma', 1: 'Tumor', 2: 'Fat', 3: 'Necrosis'}
            map_reverse_key = dict([(v, k) for k, v in map_key.items()])
            all_preds.append(preds)
            cm_analysis(y_true=preds.label,
                        y_pred=preds.predicted_class,
                        ymap=map_key,
                        labels=['Stroma', 'Tumor', 'Fat', 'Necrosis'],
                        filename='evals/{}'.format(pred_file.split('/')[-1].replace('.csv', '_confusion.png')))

            with open('evals/{}'.format(pred_file.split('/')[-1].replace('.csv', '.txt')), 'w') as f:
                json.dump(results, f)
            preds = preds.reset_index()
            preds['image_id'] = preds['tile_file_name'].str.split('/').map(lambda x: str(x[0]))
            df = pd.read_csv(config.args.preprocessed_cohort_csv_path)[['image_path']]
            df['image_id'] = df['image_path'].str.split('/').map(lambda x: str(x[-1][:-4]))
            df = df.set_index('image_id')
            preds = preds.join(df, on='image_id', how='left').drop(columns=['image_id'])
            visualize(preds, pred_file)

    print('{:4.3f} +/- {:4.3f}'.format(np.mean(all_acc), np.std(all_acc)))
    all_preds = pd.concat(all_preds, axis=0)
    all_preds.to_csv('evals/all_preds_{}'.format(config.args.val_pred_file.split('/')[-1].format('_all')))
    cm_analysis(y_true=all_preds.label,
                y_pred=all_preds.predicted_class,
                ymap=map_key,
                labels=['Stroma', 'Tumor', 'Fat', 'Necrosis'],
                filename='evals/integrated.svg')
