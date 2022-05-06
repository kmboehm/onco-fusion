"""
extract_ct_features.py extracts PyRadiomics-defined features from windowed CT volumes.
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml

from joblib import Parallel, delayed
from radiomics import featureextractor


def define_all_lesions_for_one_site(df, params_fn, results_fn):
    print('-'*32)
    print(params_fn)
    
    all_results = Parallel(n_jobs=16)(delayed(extract_single)(params_fn, row) for _, row in df.iterrows())
    #all_results = []
    #for _, row in df.iterrows():
    #    all_results.append(extract_single(params_fn, row))
    results = pd.DataFrame(all_results)
    results['Patient ID'] = results['Patient ID'].astype(str).apply(lambda x: x.zfill(3))
    results.to_csv(results_fn, index=False)


def extract_single(params_fn, row):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_fn)
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)
    try:
        result = extractor.execute(os.path.join(DATA_DIR, row.windowed_image_path),
                                   os.path.join(DATA_DIR, row.segmentation_path))
        print('Extracted features successfully for {}'.format(row.windowed_image_path))
    except ValueError:
        result = {}
        print('WARNING: ValueError for {}'.format(row.windowed_image_path))
    except RuntimeError:
        result = {}
        print('WARNING: RuntimeError for {}'.format(row.windowed_image_path))

    result.update({'Patient ID': str(row['Patient ID'])})
    return result


if __name__ == '__main__':
    with open('../global_config.yaml', 'r') as f:
        CONFIGS = yaml.safe_load(f)
        DATA_DIR = CONFIGS['data_dir']
    
    INPUT_DATAFRAME_PATH = os.path.join(DATA_DIR, 'data/dataframes/ct_df.csv')
    DF = pd.read_csv(INPUT_DATAFRAME_PATH)
    DF['Patient ID'] = DF['Patient ID'].astype(str)
    BINS = [25]

    for B in BINS:
        print('Extracting features for bin size {}'.format(B))
    
        define_all_lesions_for_one_site(DF,
        'params_left_ovary{}.yaml'.format(B),
        'features/_ct_left_ovary_bin{}.csv'.format(B))

        define_all_lesions_for_one_site(DF,
        'params_right_ovary{}.yaml'.format(B),
        'features/_ct_right_ovary_bin{}.csv'.format(B))
        
        define_all_lesions_for_one_site(DF,
        'params_omentum{}.yaml'.format(B),
        'features/_ct_omentum_bin{}.csv'.format(B))

