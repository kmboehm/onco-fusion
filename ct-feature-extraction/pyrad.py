import pandas as pd
from radiomics import featureextractor
import numpy as np
import os
from joblib import Parallel, delayed
import logging


def define_all_lesions_for_one_site(df, params_fn, results_fn, _type='tumor'):
    print('-'*32)
    print(params_fn)
    
    #all_results = []
    #for _, row in df.iterrows():
    #    all_results.append(extract_single(_type, params_fn, row))
    #    break
    all_results = Parallel(n_jobs=64)(delayed(extract_single)(_type, params_fn, row) for _, row in df.iterrows())
    
    results = pd.DataFrame(all_results)
    results.to_csv(results_fn)


def extract_single(_type, params_fn, row):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_fn)
    # set level for all classes
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)
    print(row.windowed_img)
    print(row.pred_seg)
    try:
        result = extractor.execute(row.windowed_img, row.pred_seg)
    except ValueError:
        result = {}
        print('skipping {}'.format(row.ID))
    except RuntimeError:
        result = {}
        print('WARNING: RuntimeError for {}'.format(row.ID))

    result.update({'ID': row.ID})
    return result


if __name__ == '__main__':
    df = pd.read_csv('/gpfs/mskmind_ess/boehmk/adnexal-segmentation/integrated_ct_DF_preprocessed.csv')
    #df = pd.read_csv('integrated_ct_df.csv')
    bins = [25] #[5, 10, 25, 30, 35]
    print(df)
    for b in bins:
        print(b)
    
        define_all_lesions_for_one_site(df, 'params_auto{}.yaml'.format(b), 'features/auto_windowed_resampled_results_ovary_tumor_bin{}.csv'.format(b), _type='tumor')
        #define_all_lesions_for_one_site(df, 'params_left_ovary{}.yaml'.format(b), 'features/windowed_resampled_results_left_ovary_tumor_bin{}.csv'.format(b), _type='tumor')
        #define_all_lesions_for_one_site(df, 'params_right_ovary{}.yaml'.format(b), 'features/windowed_resampled_results_right_ovary_tumor_bin{}.csv'.format(b), _type='tumor')
        #define_all_lesions_for_one_site(df, 'params_omentum{}.yaml'.format(b), 'features/windowed_resampled_results_omentum_tumor_bin{}.csv'.format(b), _type='tumor')

