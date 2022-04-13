import pandas as pd
import numpy as np


def load_crs(binarize=False, drop_net=True):
    df = pd.read_excel('/Users/boehmk/shahLab/thesis/nact-hgsc/NACT_IDS_List_for_Chemotherapy_Response_Score_08272019.xlsx')
    df = df.dropna(subset=['mrn'])
    df.mrn = df.mrn.astype(str)
    df = df.set_index('mrn')
    df = df[['CRS']]
    df = df.dropna(subset=['CRS'])
    if binarize:
        df.loc[df.CRS==2, 'CRS'] = '1/2'
        df.loc[df.CRS==1, 'CRS'] = '1/2'
        df.loc[df.CRS==3, 'CRS'] = '3/NET'
        df.loc[df.CRS=='NET', 'CRS'] = '3/NET'
        df.loc[df.CRS=='1 or 2', 'CRS'] = '1'
        df.loc[df.CRS=='2 or 3', 'CRS'] = '3/NET'
        df.loc[df.CRS.str.contains('1'), 'CRS'] = '1/2'
    else:
        df.loc[df.CRS==2, 'CRS'] = '2'
        df.loc[df.CRS==1, 'CRS'] = '1'
        df.loc[df.CRS==3, 'CRS'] = '3'
        df.loc[df.CRS=='1 or 2', 'CRS'] = '2'
        df.loc[df.CRS=='2 or 3', 'CRS'] = '2'
        df.loc[df.CRS.str.contains('1'), 'CRS'] = 1
        df['CRS'] = df['CRS'].astype(str)
    return df


def load_os():
    surv = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/integrated_surv_df_OS_updated9Nov2021.csv')
    surv['mrn'] = surv['mrn'].astype(str)
    surv = surv.drop_duplicates(subset=['mrn']).set_index('mrn')
    surv = surv[['duration', 'observed']]
    return surv

def load_pfs():
    surv = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/integrated_surv_df_PFS_updated_30Nov2021.csv')
    surv['mrn'] = surv['mrn'].astype(str)
    surv = surv.drop_duplicates(subset=['mrn']).set_index('mrn')
    surv = surv[['duration', 'observed']]
    return surv

def load_clin(cols=['cgr', 'stage', 'age', 'Type of surgery', 'adnexal_lesion', 'omental_lesion', 'parp_nact']):
    clin = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/integrated_surv_df_OS_updated9Nov2021.csv')
    clin['mrn'] = clin['mrn'].astype(str)
    clin['stage'] = clin['stage'].astype(str).apply(lambda x: x.strip().split(':')[-1].replace('A', '').replace('B', '').replace('C', '').replace('2', ''))
    clin['stage'] = clin['stage'].replace('', 'nan')
    clin = clin.drop_duplicates(subset=['mrn']).set_index('mrn')
    # print(clin['parp_nact'].value_counts(dropna=False))
    # print(clin)
    # print(clin.columns)
    # exit()
    clin = clin[cols]
    return clin    


def load_pathomic_features():
    df = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/hne_features.csv')
    df['mrn'] = df['mrn'].astype(str)
    df = df.set_index('mrn')
    return df


def load_radiomic_features(site='omentum'):
    if site == 'omentum':
        df = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/ct_features_omentum.csv')
    elif site == 'ovary':
        df = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/ct_features_ovary.csv')
    else:
        raise NotImplementedError("Unknown radiomic site: {}".format(site))
    df['mrn'] = df['mrn'].astype(str)
    df = df.set_index('mrn')
    return df


def load_dmp_mrn_map():
    df = pd.read_csv('raw_features/dmp_mrn_map.csv')
    df = df.drop_duplicates(subset=['mrn'])
    return df


def load_all_mrns(imaging_only=False):
    radiomic_mrns = set(load_radiomic_features('omentum').index).union(set(load_radiomic_features('ovary').index))
    pathomic_mrns = set(load_pathomic_features().index)
    if imaging_only:
        mrns = pathomic_mrns.union(radiomic_mrns)
    else:
        clinical_mrns = set(load_clin().index)
        genomic_mrns = set(load_genom().index)
        mrns = pathomic_mrns.union(radiomic_mrns).union(genomic_mrns).union(clinical_mrns)
    return mrns


def load_genom(set_unknown_to_false=True, set_ambiguous_to_false=False):
    genom = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/full_genomic_analysis_df.csv')
    genom['mrn'] = genom['mrn'].astype(str)
    genom = genom.set_index('mrn')
    genom.loc[genom.status == 'HRD', 'hrd_status'] = True
    genom.loc[genom.status == 'HRP', 'hrd_status'] = False
    if set_unknown_to_false:
        genom.loc[genom.status == 'unknown', 'hrd_status'] = False
    else:
        genom.loc[genom.status == 'unknown', 'hrd_status'] = np.nan
        
    if set_ambiguous_to_false:
        genom.loc[genom.status == 'ambiguous', 'hrd_status'] = False
    else:
        genom.loc[genom.status == 'ambiguous', 'hrd_status'] = np.nan
    genom = genom[['hrd_status']]

    genom_tcga = pd.read_csv('/Users/boehmk/shahLab/thesis/reanalysis/features/tcga_hrd_statuses.csv').set_index('ID').rename(columns={'target': 'hrd_status'})
    genom_tcga = genom_tcga[['hrd_status']]

    genom = pd.concat([genom, genom_tcga])

    genom = genom[~genom.hrd_status.isna()]
    genom.hrd_status = genom.hrd_status.astype(bool)
    genom = genom[~genom.index.duplicated(keep='last')]

    return genom