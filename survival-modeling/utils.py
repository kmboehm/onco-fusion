import pandas as pd
import numpy as np
import yaml
import os

with open('../global_config.yaml', 'r') as f:
    CONFIGS = yaml.safe_load(f)
    DATA_DIR = CONFIGS['data_dir']
    CODE_DIR = CONFIGS['code_dir']


def load_crs(binarize=False, drop_net=False):
    df = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'crs_df.csv'))
    df['Patient ID'] = df['Patient ID'].astype(str).apply(lambda x: x.zfill(3))
    df = df.set_index('Patient ID')
    if drop_net:
        df = df[df.CRS != 'NET']
    if binarize:
        df.loc[df.CRS=='2', 'CRS'] = '1/2'
        df.loc[df.CRS=='1', 'CRS'] = '1/2'
        df.loc[df.CRS=='3', 'CRS'] = '3/NET'
        df.loc[df.CRS=='NET', 'CRS'] = '3/NET'
        df.loc[df.CRS.str.contains('1'), 'CRS'] = '1/2'
    df['CRS'] = df['CRS'].astype(str)
    return df


def load_os():
    df = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'clin_df.csv'))
    df['Patient ID'] = df['Patient ID'].astype(str)
    df = df.set_index('Patient ID')
    df = df[['duration.OS', 'observed.OS']]
    df = df.rename(columns={'duration.OS': 'duration',
                            'observed.OS': 'observed'})
    return df

def load_pfs():
    df = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'clin_df.csv'))
    df['Patient ID'] = df['Patient ID'].astype(str)
    df = df.set_index('Patient ID')
    df = df[['duration.PFS', 'observed.PFS']]
    df = df.rename(columns={'duration.PFS': 'duration',
                            'observed.PFS': 'observed'})
    return df

def load_clin(cols=['Complete gross resection', 'stage', 'age', 'Type of surgery', 'adnexal_lesion', 'omental_lesion', 'Received PARPi']):
    df = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'clin_df.csv'))
    df['Patient ID'] = df['Patient ID'].astype(str)
    df = df.set_index('Patient ID')
    df = df[cols]
    return df   

def load_pathomic_features():
    df = pd.read_csv(os.path.join(CODE_DIR, 'code', 'hne-feature-extraction', 'tissue_tile_features', 'reference_hne_features.csv'))
    df['Patient ID'] = df['Patient ID'].astype(str)
    df = df.set_index('Patient ID')
    return df


def load_radiomic_features(site='omentum'):
    if site == 'omentum':
        df = pd.read_csv(os.path.join(CODE_DIR, 'code', 'ct-feature-extraction', 'features', 'ct_features_omentum.csv'))
    elif site == 'ovary':
        df = pd.read_csv(os.path.join(CODE_DIR, 'code', 'ct-feature-extraction', 'features', 'ct_features_ovary.csv'))
    else:
        raise NotImplementedError("Unknown radiomic site: {}".format(site))
    df['Patient ID'] = df['Patient ID'].astype(str)
    df = df.set_index('Patient ID')
    return df



def load_all_ids(imaging_only=False):
    radiomic_ids = set(load_radiomic_features('omentum').index).union(set(load_radiomic_features('ovary').index))
    pathomic_ids = set(load_pathomic_features().index)
    if imaging_only:
        ids = pathomic_ids.union(radiomic_ids)
    else:
        clinical_ids = set(load_clin().index)
        genomic_ids = set(load_genom().index)
        ids = pathomic_ids.union(radiomic_ids).union(genomic_ids).union(clinical_ids)
    return ids


def load_genom():
    genom = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'genomic_df.csv'))
    genom['Patient ID'] = genom['Patient ID'].astype(str)
    genom = genom.set_index('Patient ID')
    genom.loc[genom['HRD status'] == 'HRP', 'hrd_status'] = False
    genom.loc[genom['HRD status'] == 'HRD', 'hrd_status'] = True
    genom = genom.dropna(subset=['HRD status'])
    genom.hrd_status = genom.hrd_status.astype(bool)
    genom = genom[['hrd_status']]

    return genom
