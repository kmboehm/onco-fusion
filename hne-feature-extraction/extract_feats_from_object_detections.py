import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import config


def get_density(obj_feats, regional_feats, result_dict, parent, class_):
    parent_area = regional_feats['{}_area'.format(parent)].item() * 64  # scale factor for 1/16 downsampling and 0.5 µm / pixel
    # print(parent_area)
    obj_mask = (obj_feats.Parent == parent) & (obj_feats.Class == class_)
    # print(obj_mask)
    obj_count = obj_mask.sum()
    # print(obj_count)
    key_ = '{}_{}_density'.format(parent, class_)
    result_dict[key_] = float(obj_count) / parent_area


def get_quantiles(object_feats, mask, feat, output_feat_name, results_dict):
    results_dict[output_feat_name.format('mean')] = object_feats.loc[mask, feat].mean()
    for quantile in np.arange(0.1, 1, 0.1):
        results_dict[output_feat_name.format('quantile{:2.1f}'.format(quantile))] = object_feats.loc[mask, feat].quantile(quantile)
    results_dict[output_feat_name.format('var')] = object_feats.loc[mask, feat].var()
    results_dict[output_feat_name.format('skew')] = object_feats.loc[mask, feat].skew()
    results_dict[output_feat_name.format('kurtosis')] = object_feats.loc[mask, feat].kurtosis()


def extract_feats(object_feat_fn, regional_feature_df_, slide_id):
    regional_feature_df = regional_feature_df_[regional_feature_df_.index.isin([slide_id])]
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
    regional_feat_df_filename = os.path.join(config.args.base_path,
                                'tissue_tile_features/2021-02-06_14.26.33_fold-2_epoch018.csv')
    object_detection_dir = os.path.join(config.args.base_path, 'final_objects')
    merged_feat_df_filename = os.path.join(config.args.base_path,
                            'tissue_tile_features/2021-02-06_14.26.33_fold-2_epoch018_merged_scanned.csv')
    SERIAL = False
    DETECTION_PROB_THRESHOLD = 0.1

    slide_list = [x for x in os.listdir(object_detection_dir) if (('.csv' in x) or ('.tsv' in x))]
    regional_feat_df = pd.read_csv(regional_feat_df_filename).rename(columns={'Unnamed: 0': 'image_id'}).set_index('image_id')

    results = {}
    if SERIAL:
        for slide in slide_list:
            print(slide)
            result = extract_feats(os.path.join(object_detection_dir, slide), regional_feat_df, slide[:-4])
            results.update(result)
            # break
    else:
        dicts = Parallel(n_jobs=32)(delayed(extract_feats)(os.path.join(object_detection_dir, slide), regional_feat_df, slide[:-4]) for slide in slide_list)
        for dict_ in dicts:
            results.update(dict_)

    df = pd.DataFrame(results).T
    print(df)
    df = df.join(regional_feat_df, how='inner')
    print(df)

    df.to_csv(merged_feat_df_filename, index=True)

