"""
make_windowed_vols.py applies the abdominal window to the raw MHD files in data/dataframes/ct_df.csv
and saves the resulting MHD files in data/ct/windowed_cts. TCGA MHD files must be downloaded before
running.
"""
import pandas as pd
import numpy as np
import os
import yaml

from medpy.io import load, save
from joblib import delayed, Parallel


with open('../global_config.yaml', 'r') as f:
    DATA_DIR = yaml.safe_load(f)['data_dir']

LEVEL = 50
WIDTH = 400
LOWER_BOUND = LEVEL - WIDTH//2
UPPER_BOUND = LEVEL + WIDTH//2
INPUT_DATAFRAME_PATH = os.path.join(DATA_DIR, 'data/dataframes/ct_df.csv')


def make_windowed(row):
    """
    Given row of CT data frame, load MHD file, apply window, and save windowed version.
    """
    input_tumor_img_fn = os.path.join(DATA_DIR, row['image_path'])
    output_tumor_img_fn = os.path.join(DATA_DIR, row['windowed_image_path'])
    try:
        tumor_img, header = load(input_tumor_img_fn)
        tumor_img = np.clip(tumor_img, a_min=LOWER_BOUND, a_max=UPPER_BOUND)
        sub_dir = '/'.join(output_tumor_img_fn.split('/')[:-1])
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        save(tumor_img, output_tumor_img_fn, header)
        print('{} succeeded'.format(input_tumor_img_fn))
    except:
        print('{} failed'.format(input_tumor_img_fn))


if __name__ == '__main__':
    df = pd.read_csv(INPUT_DATAFRAME_PATH)
    if not os.path.exists(os.path.join(DATA_DIR, 'data/ct/windowed_scans')):
        os.mkdir(os.path.join(DATA_DIR, 'data/ct/windowed_scans'))
    Parallel(n_jobs=16)(delayed(make_windowed)(row) for idx, row in df.iterrows())
    #for idx, row in df.iterrows():
    #    make_windowed(idx, row)
