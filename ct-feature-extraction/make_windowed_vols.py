import pandas as pd
import numpy as np
import os
from medpy.io import load, save
import medpy
from joblib import delayed, Parallel

LEVEL = 50
WIDTH = 400
LOWER_BOUND = LEVEL - WIDTH//2
UPPER_BOUND = LEVEL + WIDTH//2
OUTPUT_DIR = '/gpfs/mskmind_ess/boehmk/new-pyrad/windowed_scans'

def make_windowed(idx, row):
    input_tumor_img_fn = row['img']
    output_tumor_img_fn = row['windowed_img']
    try:
        tumor_img, header = load(input_tumor_img_fn)
        tumor_img = np.clip(tumor_img, a_min=LOWER_BOUND, a_max=UPPER_BOUND)
        sub_dir = '/'.join(output_tumor_img_fn.split('/')[:-1])
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        save(tumor_img, output_tumor_img_fn, header)
    except:
        print('{} failed'.format(row['img']))

if __name__ == '__main__':
    df_fn = 'integrated_ct_df.csv'
    df = pd.read_csv(df_fn)
    df = df.drop(columns=['windowed_img'])
    df.loc[df.cohort2, 'windowed_img'] = OUTPUT_DIR + '/' + df.img.apply(lambda x: x.split('/')[-3]) + '/windowed_' + df.img.apply(lambda x: x.split('/')[-1])
    df.loc[df.cohort1, 'windowed_img'] = OUTPUT_DIR + '/' + df.img.apply(lambda x: x.split('/')[-4]) + '/windowed_' + df.img.apply(lambda x: x.split('/')[-1])
    df.loc[df.cohort3, 'windowed_img'] = OUTPUT_DIR + '/' + df.img.apply(lambda x: x.split('/')[-3]) + '/windowed_' + df.img.apply(lambda x: x.split('/')[-1])
    df.to_csv('integrated_ct_df.csv')
    Parallel(n_jobs=64)(delayed(make_windowed)(idx, row) for idx, row in df.iterrows())
    #for idx, row in df.iterrows():
    #    make_windowed(idx, row)
