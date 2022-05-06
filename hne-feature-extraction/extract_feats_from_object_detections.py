import pandas as pd
import numpy as np
import os
import yaml
from joblib import Parallel, delayed
import sys
sys.path.append('../tissue-type-training')
import config


def get_density(obj_feats, regional_feats, result_dict, parent, class_):
    parent_area = regional_feats['{}_area'.format(parent)].item() * 64  # scale factor for 1/16 downsampling and 0.5 µm / pixel
    # print(parent_area)
    obj_mask = (obj_feats.Parent == parent) & (obj_feats.Class == class_)
    # print(obj_mask)
    obj_count = obj_mask.sum()
    # print(obj_count)
    key_ = '{}_{}_density'.format(parent, class_)
    try:
        result_dict[key_] = float(obj_count) / parent_area
    except ZeroDivisionError:
        result_dict[key_] = np.nan


def get_quantiles(object_feats, mask, feat, output_feat_name, results_dict):
    results_dict[output_feat_name.format('mean')] = object_feats.loc[mask, feat].mean()
    for quantile in np.arange(0.1, 1, 0.1):
        results_dict[output_feat_name.format('quantile{:2.1f}'.format(quantile))] = object_feats.loc[mask, feat].quantile(quantile)
    results_dict[output_feat_name.format('var')] = object_feats.loc[mask, feat].var()
    results_dict[output_feat_name.format('skew')] = object_feats.loc[mask, feat].skew()
    results_dict[output_feat_name.format('kurtosis')] = object_feats.loc[mask, feat].kurtosis()


def extract_feats(object_feat_fn, regional_feature_df_, slide_id):
    regional_feature_df = regional_feature_df_[regional_feature_df_.index.astype(str) == str(slide_id)]
    if len(regional_feature_df) != 1:
        return {}

    result_ = {}
    object_feats = pd.read_csv(object_feat_fn)
    if len(object_feats.columns) == 1:
        object_feats = pd.read_csv(object_feat_fn, delimiter='\t')
    object_feats = object_feats[object_feats['Detection probability'] > DETECTION_PROB_THRESHOLD]
    # print(object_feats.Parent.value_counts())
    get_density(object_feats,
                regional_feature_df,
                result_,
                parent='Tumor',
                class_='Lymphocyte'
                )
    get_density(object_feats,
                regional_feature_df,
                result_,
                parent='Tumor',
                class_='Other'
                )
    get_density(object_feats,
                regional_feature_df,
                result_,
                parent='Necrosis',
                class_='Other'
                )
    get_density(object_feats,
                regional_feature_df,
                result_,
                parent='Stroma',
                class_='Lymphocyte'
                )
    get_density(object_feats,
                regional_feature_df,
                result_,
                parent='Stroma',
                class_='Other'
                )
    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Nucleus: Area µm^2',
                  output_feat_name='Tumor_Other_{}_nuclear_area',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Nucleus: Circularity',
                  output_feat_name='Tumor_Other_{}_nuclear_circularity',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Nucleus: Solidity',
                  output_feat_name='Tumor_Other_{}_nuclear_solidity',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Nucleus: Max diameter µm',
                  output_feat_name='Tumor_Other_{}_nuclear_max_diameter',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Hematoxylin: Nucleus: Mean',
                  output_feat_name='Tumor_Other_{}_nuclear_hematoxylin_mean',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Hematoxylin: Nucleus: Median',
                  output_feat_name='Tumor_Other_{}_nuclear_hematoxylin_median',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Hematoxylin: Nucleus: Min',
                  output_feat_name='Tumor_Other_{}_nuclear_hematoxylin_min',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Hematoxylin: Nucleus: Max',
                  output_feat_name='Tumor_Other_{}_nuclear_hematoxylin_max',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Hematoxylin: Nucleus: Std.Dev.',
                  output_feat_name='Tumor_Other_{}_nuclear_hematoxylin_stdDev',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Eosin: Nucleus: Mean',
                  output_feat_name='Tumor_Other_{}_nuclear_eosin_mean',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Eosin: Nucleus: Median',
                  output_feat_name='Tumor_Other_{}_nuclear_eosin_median',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Eosin: Nucleus: Min',
                  output_feat_name='Tumor_Other_{}_nuclear_eosin_min',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Eosin: Nucleus: Max',
                  output_feat_name='Tumor_Other_{}_nuclear_eosin_max',
                  results_dict=result_)

    get_quantiles(object_feats=object_feats,
                  mask=(object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other'),
                  feat='Eosin: Nucleus: Std.Dev.',
                  output_feat_name='Tumor_Other_{}_nuclear_eosin_stdDev',
                  results_dict=result_)

    try:
        result_['ratio_Tumor_Lymphocyte_to_Tumor_Other'] = float(
            ((object_feats.Parent == 'Tumor') & (object_feats.Class == 'Lymphocyte')).sum()) / float(
            ((object_feats.Parent == 'Tumor') & (object_feats.Class == 'Other')).sum()
        )
    except ZeroDivisionError:
        result_['ratio_Tumor_Lymphocyte_to_Tumor_Other'] = np.nan

    return {int(slide_id): result_}


if __name__ == '__main__':

    checkpoint_name = config.args.checkpoint_path.split('/')[-1].replace('.torch', '')
    
    regional_feat_df_filename = 'tissue_tile_features/{}.csv'.format(checkpoint_name)

    object_detection_dir = 'final_objects/{}'.format(checkpoint_name)
    
    merged_feat_df_filename = 'tissue_tile_features/{}_merged.csv'.format(checkpoint_name)
    SERIAL = True
    DETECTION_PROB_THRESHOLD = 0.5

    slide_list = [x for x in os.listdir(object_detection_dir) if (('.csv' in x) or ('.tsv' in x))]
    regional_feat_df = pd.read_csv(regional_feat_df_filename).set_index('image_id')

    results = {}
    if SERIAL:
        for slide in slide_list:
            print(slide)
            result = extract_feats(os.path.join(object_detection_dir, slide), regional_feat_df, slide[:-4])
            results.update(result)
    else:
        dicts = Parallel(n_jobs=32)(delayed(extract_feats)(os.path.join(object_detection_dir, slide), regional_feat_df, slide[:-4]) for slide in slide_list)
        for dict_ in dicts:
            results.update(dict_)

    df = pd.DataFrame(results).T
    df = df.join(regional_feat_df, how='inner')
    print(df)

    df = df.reset_index().rename(columns={'index': 'image_id'})
    df['image_id'] = df['image_id'].astype(str)

    hne_df = pd.read_csv(config.args.preprocessed_cohort_csv_path)
    hne_df['image_id'] = hne_df['image_path'].apply(lambda x: x.split('/')[-1][:-4]).astype(str)
    df = df.join(hne_df[['image_id', 'Patient ID', 'n_foreground_tiles']].set_index('image_id'), on='image_id', how='left')
    df = df.drop(columns=['image_id'])
    df['Patient ID'] = df['Patient ID'].astype(str).apply(lambda x: x.zfill(3))
    df = df.fillna(df.median())
    df.to_csv(merged_feat_df_filename, index=False)

