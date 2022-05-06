from utils import load_os, load_clin, load_pathomic_features, load_radiomic_features, load_genom, load_crs, load_all_ids, load_pfs
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test, logrank_test
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations, product
from lifelines.exceptions import ConvergenceError
from copy import deepcopy
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from statannot import add_stat_annotation
from scipy.stats import mannwhitneyu, kruskal, percentileofscore, rankdata, chi2_contingency
import os
import sys
import yaml
sys.path.append('../feature-selection')
from select_features import preprocess_features, evaluate_features

with open('../global_config.yaml', 'r') as f:
    CONFIGS = yaml.safe_load(f)
    DATA_DIR = CONFIGS['data_dir']
    CODE_DIR = CONFIGS['code_dir']


FONTSIZE = 7
plt.rc('legend',fontsize=FONTSIZE-2, title_fontsize=FONTSIZE-2)
plt.rc('xtick',labelsize=FONTSIZE)
plt.rc('ytick',labelsize=FONTSIZE)
plt.rc("axes", labelsize=FONTSIZE)

def plot_genom_KM(df, plot_file_name, df_file_name=None, source_file_name=None, ylab="proportion (OS)"):
    median_surv_dict = {}
    mask_ = df.hrd_status
    results = logrank_test(df.loc[mask_, 'duration'],
                            df.loc[~mask_, 'duration'],
                            df.loc[mask_, 'observed'],
                            df.loc[~mask_, 'observed'])
    kmf = KaplanMeierFitter()
    fig = plt.figure(figsize=(1.73, 1) if 'g.svg' not in plot_file_name else (2,2), constrained_layout=True)

    labels = ['HRD', 'HRP']
    colors = ['#005a8a', '#443500']
    masks = [mask_, ~mask_]
    for idx, (mask, label, color) in enumerate(zip(masks, labels, colors)):
        kmf.fit(df.loc[mask, 'duration'].to_numpy().ravel(), df.loc[mask, 'observed'].to_numpy().ravel(), label='{} (n={})'.format(label, sum(mask)))
        median_surv_dict[label] = {'median_survival_time': kmf.median_survival_time_}
        # print('Median survival time for {}: {:4.1f} months'.format(label, kmf.median_survival_time_))
        kmf.plot(ci_show=False, show_censors=True, color=color, linewidth=1, censor_styles={'alpha': 0.75, 'ms': 4}) # linewidth=2
        plt.xlabel('time (m)')
        plt.ylabel(ylab)
    if results.p_value > 5e-4:
        plt.text(0, 0, 'p = {:4.3f}'.format(results.p_value), fontsize=FONTSIZE)
    else:
        plt.text(0, 0, 'p < 5e-4', fontsize=FONTSIZE)
    plt.ylim(-0.05, 1.05)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(plot_file_name, dpi=300 if '.png' in plot_file_name else None)
    plt.close()
    if df_file_name:    
        pd.DataFrame(median_surv_dict).T.reset_index().rename(columns={'index': 'risk_group'}).to_csv(df_file_name, index=False)
    if source_file_name:
        df[['hrd_status', 'observed', 'duration']].to_csv(source_file_name, index=False)


def calc_c_dev(x,y, col):
    _df_ = x.join(y, how='inner')
    c = concordance_index(_df_.duration, _df_[col], _df_.observed)
    return np.abs(c - 0.5)

def split_train_test(df):
    test_ids = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'test_ids.csv'))['Patient ID'].astype(str)
    train_ids = pd.read_csv(os.path.join(DATA_DIR, 'data', 'dataframes', 'train_ids.csv'))['Patient ID'].astype(str)
    assert not set(train_ids).intersection(set(test_ids))
    test_df = df[df.index.isin(test_ids)]
    train_df = df[df.index.isin(train_ids)]
    return train_df, test_df

def encode_clinical_features(train_df, test_df):
    feats = [x for x in ['stage', 'Type of surgery'] if x in train_df.columns]
    enc = OneHotEncoder(sparse=False)
    enc.fit(pd.concat([train_df[feats], test_df[feats]]).values)

    train_df_categorical_feats = enc.transform(train_df[feats].values)
    cols = [feats[0] + '_' + str(x) for x in enc.categories_[0].tolist()]
    for ind in range(1, len(enc.categories_)):
        cols.extend([feats[ind] + '_' + str(x) for x in enc.categories_[ind].tolist()])
    train_df_categorical_feats = pd.DataFrame(train_df_categorical_feats, columns=cols, index=train_df.index)
    train_df = train_df.drop(columns=feats).join(train_df_categorical_feats)

    test_df_categorical_feats = enc.transform(test_df[feats].values)
    test_df_categorical_feats = pd.DataFrame(test_df_categorical_feats, columns=cols, index=test_df.index)
    test_df = test_df.drop(columns=feats).join(test_df_categorical_feats)
    return train_df, test_df

def remove_low_iqr(df, threshold=0.1):
    temp_df = df.copy()
    temp_scaler = MinMaxScaler()
    temp_df = pd.DataFrame(temp_scaler.fit_transform(temp_df.values), columns=temp_df.columns, index=temp_df.index)
    
    iqr_vals, feat_names = [], []
    for col in temp_df.columns:
        var_ = temp_df[col].quantile(0.75) - temp_df[col].quantile(0.25)
        iqr_vals.append(var_)
        feat_names.append(col)

    iqr_vals = np.array(iqr_vals)
    feat_names = np.array(feat_names)
    feat_names = feat_names[iqr_vals > threshold]
    return list(feat_names)

def select_imaging_features(X, y, feature_table, filter_=None, remove_filter=None, thresh=0.05, strategy='reject_news', verbose=False, modality='radiology', use_corrected=True):
    assert strategy in ['reject_news', 'reject_max']
    features = remove_low_iqr(X[feature_table.feat.tolist()], threshold=0.1)
    feature_table = feature_table.loc[feature_table.feat.isin(features)]
    if filter_:
        feature_table = feature_table[feature_table.feat.str.contains(filter_, regex=True)]
    if remove_filter:
        feature_table = feature_table[~feature_table.feat.str.contains(remove_filter, regex=True)]

    # save corrected p values
    reject, pvals_corrected, _, _ = multipletests(pvals=feature_table.p, alpha=0.05, method='fdr_bh', is_sorted=True)
    feature_table['p_corrected'] = pvals_corrected
    feature_table.drop(columns=['Unnamed: 0']).to_csv('results/p_corrected_{}.csv'.format(modality), index=False)

    if use_corrected:
        feature_table = feature_table.loc[feature_table.p_corrected < thresh]
    else:
        feature_table = feature_table.loc[feature_table.p < thresh]
    # print(feature_table)
    # exit()
    feats = feature_table.feat.tolist()

    frame = X.join(y, how='inner')

    return identify_statistically_significant_feats_on_multivariate_cph(feats, frame, thresh, strategy, verbose)


def identify_statistically_significant_feats_on_multivariate_cph(feats_, frame, thresh, strategy, verbose):
    feats = deepcopy(feats_)

    feat_to_consider = feats.pop(0)
    current_feats = [feat_to_consider]

    model = CoxPHFitter(penalizer=0.01, l1_ratio=0)

    while True:
        current_feats.extend(['duration', 'observed'])
        temp_frame = frame.copy()[current_feats]
        model.fit(temp_frame, duration_col='duration', event_col='observed')
        current_feats.remove('duration')
        current_feats.remove('observed')

        if strategy == 'reject_max':
            p_max = model.summary['p'].max()
            p_max_feat = temp_frame.iloc[model.summary['p'].argmax()].index[0]
            if verbose:
                print(p_max_feat)
                print(model.summary)
            if p_max > thresh:
                if verbose:
                    print('reject {} with {:4.3f}'.format(p_max_feat, p_max))
                current_feats.remove(p_max_feat)
            else:
                print('accept {} with {:4.3f}'.format(p_max_feat, p_max))
        else:
            p_new = model.summary.loc[model.summary.index == feat_to_consider, 'p'].item()
            if p_new > thresh:
                if verbose:
                    print('reject {} with {:4.3f}'.format(feat_to_consider, p_new))
                current_feats.remove(feat_to_consider)
            else:
                if verbose:
                    print('accept {} with {:4.3f}'.format(feat_to_consider, p_new))

        if len(feats) == 0:
            break
        else:
            feat_to_consider = feats.pop(0)
            current_feats.append(feat_to_consider)

    # print(current_feats)
    current_feats.extend(['duration', 'observed'])
    temp_frame = frame.copy()[current_feats]
    model.fit(temp_frame, duration_col='duration', event_col='observed')
    current_feats.remove('duration')
    current_feats.remove('observed')
    # if verbose:
    #     model.print_summary()

    return current_feats

def scale(train_df, test_df, col_to_scale):
    scaler = MinMaxScaler()

    train_df_scaled_feat = scaler.fit_transform(train_df[[col_to_scale]].values.reshape(-1,1))
    train_df_scaled_feat = pd.DataFrame(train_df_scaled_feat, columns=[col_to_scale], index=train_df.index)
    train_df = train_df.drop(columns=[col_to_scale]).join(train_df_scaled_feat)

    test_df_scaled_feat = scaler.transform(test_df[[col_to_scale]].values.reshape(-1,1))
    test_df_scaled_feat = pd.DataFrame(test_df_scaled_feat, columns=[col_to_scale], index=test_df.index)
    test_df = test_df.drop(columns=[col_to_scale]).join(test_df_scaled_feat)
    return train_df, test_df



def select_clinical_features(X, y, threshold=0.1):
    df = X.join(y).dropna(how='any')
    feat_table = evaluate_features(df,
                                   feats_to_consider=X.columns,
                                   method='cph',
                                   n_jobs=1)
    feat_table.to_csv('results/p_clinical.csv', index=False)
    feats = feat_table.loc[feat_table['p']<threshold, 'feat'].tolist()
    return identify_statistically_significant_feats_on_multivariate_cph(feats, df, threshold, 'reject_news', False)


def train(X, y, feat_acronym, penalty=0.5, make_forest_plot=True):
    model = CoxPHFitter(penalizer=penalty, l1_ratio=0)
    model.fit(X.join(y, how='inner'), duration_col='duration', event_col='observed', robust=True)
    if make_forest_plot:
        make_forest_plot_(model.summary, 'figures/forest_plots/forest_{}.svg'.format(feat_acronym))
        model.summary.to_csv('results/model_summaries/{}_summary.csv'.format(feat_acronym))
    return model

def train_stratified(X, y, feat_acronym, penalty=0.5):
    df = X.join(y, how='inner')
    # strata = [x for x in ['G_score', 'C_score'] if x in df.columns]
    strata = ['G_score']
    df.loc[:, strata] = df.loc[:, strata].astype(float)

    model = CoxPHFitter(penalizer=penalty, l1_ratio=0)
    model.fit(df, duration_col='duration', event_col='observed', strata=strata, robust=True)
    # if not ((len(strata) == 2) and (len(feat_acronym) == 2)):
    make_forest_plot_(model.summary, 'figures/forest_plots/forest_{}.svg'.format(feat_acronym))

    return model

def make_forest_plot_(summary, output_file_name):
    plt.rc('ytick',labelsize=FONTSIZE)
    coef = []
    modality = []
    if summary.index.str.contains('wavelet-').any():
        summary['abbreviated_feature'] = summary.index.str.replace('wavelet-', '')
        summary['abbreviated_feature'] = summary['abbreviated_feature'].str.replace('log-sigma-1-0-mm-3D_', 'LoG_')
        summary['abbreviated_feature'] = summary['abbreviated_feature'].str.replace('log-sigma-3-0-mm-3D_', 'LoG_')
        summary['abbreviated_feature'] = summary['abbreviated_feature'].apply(lambda x: '_'.join([x.split('_')[0], x.split('_')[-1]]))
        fig_len = 2.75
    elif summary.index.str.contains('Tumor').any():
        summary['abbreviated_feature'] = summary.index.str.replace('Tumor_Other_mean_nuclear_area', 'Mean tumor nuc. area')
        summary['abbreviated_feature'] = summary['abbreviated_feature'].str.replace('Stroma_major_axis_length', 'Stroma maj. axis len.')
        summary = summary.sort_values(by=['abbreviated_feature'], ascending=True)
        fig_len = 1.97
    else:
        summary['abbreviated_feature'] = summary.index
        fig_len  = 2
    
    for covar, row in summary.iterrows():
        modality.extend(3*[row['abbreviated_feature'].replace('_',' ').replace(' score', '')])
        coef.append(float(summary.loc[summary.index == covar, 'coef lower 95%']))
        coef.append(float(summary.loc[summary.index == covar, 'coef']))
        coef.append(float(summary.loc[summary.index == covar, 'coef upper 95%']))
    fig = plt.figure(figsize=(fig_len, len(summary)*0.25+0.25), constrained_layout=True)
    feat_coef = pd.DataFrame({'modality': modality,
                          'coefficient': coef})
    with plt.rc_context({'lines.linewidth': 0.5, 'lines.markersize': 0.5}):
        ax = sns.pointplot(x="coefficient", y="modality", data=feat_coef, join=False, color='.2')
    plt.axvline(x=0, color='.2', linestyle=':', linewidth=1.0)
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(output_file_name, dpi=300 if '.svg' not in output_file_name else None)
    plt.close()
    plt.rc('ytick',labelsize=FONTSIZE-2)

def infer(X, y, model, col_name):
    return pd.DataFrame({col_name: -model.predict_partial_hazard(X.join(y))})

def perm_c(df_, score_col):
    df = df_.copy()
    randomized_score_col = pd.DataFrame({score_col: df[score_col].sample(frac=1.0, replace=False, ignore_index=True).tolist()}, index=df.index)
    # randomized_score_col.index== df.index.tolist()#.reindex(index=df.index)
    df.loc[:, score_col] = randomized_score_col.loc[:, score_col]
    return concordance_index(event_times=df['duration'],
                      predicted_scores=df[score_col],
                      event_observed=df['observed'])

def perm_p(c, df_, score_col, k=1000):
    perm_c_values = Parallel(n_jobs=-1)(delayed(perm_c)(df_, score_col) for _ in range(k))
    p = (100 - percentileofscore(perm_c_values, c)) / 100.
    return p

def bootstrap_c(df_, score_col):
    df = df_.sample(n=len(df_)-1, replace=False)
    return concordance_index(event_times=df['duration'],
                          predicted_scores=df[score_col],
                          event_observed=df['observed'])

def c_index(X, y, score_col='score', label='G train', n_bootstraps=100):
    df = X.join(y, how='inner').dropna()
    df[score_col] = df[score_col].astype(int)
    c = concordance_index(event_times=df['duration'],
                          predicted_scores=df[score_col],
                          event_observed=df['observed'])
    p = perm_p(c, df, score_col)
    bootstraps = np.array(Parallel(n_jobs=-1)(delayed(bootstrap_c)(df, score_col) for _ in range(n_bootstraps)))
    return {'c': c, 'lower_ci': c - np.quantile(bootstraps, 0.025), 'upper_ci': np.quantile(bootstraps, 0.975) - c, 'permutation_p': p}

def plot_features(X, y, feats):
    df = X.join(y, how='inner')
    # df = df[df.observed]
    for feat in feats:
        # print(df[feat].unique())
        # print(len(df[feat].unique()))
        # print(df[feat].describe())
        # print(df[feat].value_counts())
        # exit()
        fig = plt.figure(figsize=(3, 1.5), constrained_layout=True)
        # sns.regplot(data=df, x=feat, y='duration',
        #             scatter=False, color='.2',
        #             line_kws={'linewidth':1},
        #             ci=95)
        sns.scatterplot(data=df, x=feat, y='duration',
                        hue='observed', palette=['#d2b04c', '#8f4cd2'],
                        style='observed', markers={True: '+', False: '+'},
                        alpha=0.7,
                        legend='full',
                        s=12)
        # sns.kdeplot(
        #     data=df,
        #     x=feat,
        #     y="duration",
        #     levels=5,
        #     fill=True,
        #     alpha=0.3,
        #     cut=1,
        #     color='gray'
        # )
        plt.ylabel('OS (m)')
        plt.xlabel(feat.replace('wavelet-HLL_glcm_Autocorrelation', 'HLL Autocorrelation'))
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # plt.savefig('figures/feature_plots/{}.png'.format(feat), dpi=300)
        plt.savefig('figures/feature_plots/{}.svg'.format(feat))
        plt.close()

def _calc_logrank_p_for_quantile(quant, df_):
    df = df_.copy()
    thresh = df.score.quantile(quant)
    df.loc[df.score < thresh, 'risk group'] = 'higher risk'
    df.loc[df.score >= thresh, 'risk group'] = 'lower risk'
    if ((df['risk group'] == 'higher risk').sum() == 0) or ((df['risk group'] == 'lower risk').sum() == 0):
        return 1.0
    return multivariate_logrank_test(df['duration'], df['risk group'], df['observed']).p_value

def calc_threshold_that_maximizes_two_group_separation(df_):
    quantiles = np.arange(0.33,0.67,0.01) #CHANGED from (0.2,0.81,0.01)
    p_values = np.array(Parallel(n_jobs=-1)(delayed(_calc_logrank_p_for_quantile)(quant, df_) for quant in quantiles))
    arg_min = np.argmin(p_values)
    quantile = quantiles[arg_min]
    best_thresh = df_.score.quantile(quantile)
    return best_thresh

def plot_halves_km(df, plot_filename, df_filename, benchmark_df, ylab, width, height):
    median_surv_dict = {}

    if benchmark_df is None:
        benchmark_df = df
    brightline = calc_threshold_that_maximizes_two_group_separation(benchmark_df)
    # brightline = benchmark_df.score.quantile(0.5)
    df.loc[df.score < brightline, 'risk group'] = 'higher risk'
    df.loc[df.score >= brightline, 'risk group'] = 'lower risk'
    df.to_csv(df_filename.replace('results/', 'results/scores-'))

    p = multivariate_logrank_test(df['duration'], df['risk group'], df['observed']).p_value

    kmf = KaplanMeierFitter()

    color_map = {'lower risk': '#000078', 'higher risk': '#8a034f'}
    fig = plt.figure(figsize=(width, height), constrained_layout=True) #1.88 for multimodal
    plt.rcParams["axes.labelsize"] = FONTSIZE
    for risk_group, sub_df in df.groupby('risk group'):
        kmf.fit(sub_df['duration'], sub_df['observed'], label='{} (n={})'.format(risk_group, len(sub_df)))
        median_surv_dict[risk_group] = {'median_survival_time': kmf.median_survival_time_,
                                        '36m_frac': kmf.predict(36, interpolate=True),
                                        '48m_frac': kmf.predict(48, interpolate=True),
                                        '24m_frac': kmf.predict(24, interpolate=True),
                                        '12m_frac': kmf.predict(12, interpolate=True)}
        kmf.plot(ci_show=False, show_censors=True, color=color_map[risk_group], linewidth=2, censor_styles={'alpha': 0.75, 'ms': 7})
    plt.xlabel('time (m)')
    plt.ylabel(ylab)
    if 'GRH' in plot_filename and 'GRHC' not in plot_filename:
        plt.gca().legend().set_title("$\\bf{GRH}$ $\\bf{model}$")

    if 'GRH' in plot_filename and 'GRHC' not in plot_filename and 'pfs' not in plot_filename:
        pval_yloc = 0.4
    else:
        pval_yloc = 0.25

    plt.gca().set_ylim(-0.05, 1.05)
    if p > 5e-4:
        plt.text(0, pval_yloc, 'p = {:4.3f}'.format(p), fontsize=FONTSIZE)
    else:
        plt.text(0, pval_yloc, 'p < 5e-4', fontsize=FONTSIZE)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(plot_filename, dpi=300 if '.svg' not in plot_filename else None)
    pd.DataFrame(median_surv_dict).T.reset_index().rename(columns={'index': 'risk_group'}).to_csv(df_filename, index=False)
    plt.close()

def make_km_plots(scores_dict, outcomes, file_name_prefix, benchmark_scores_dict=None, ylab='proportion (OS)'):
    for feat_combo, score_df in scores_dict.items():
        df = score_df.rename(columns={'{}_score'.format(feat_combo): 'score'}).join(outcomes, how='inner')
        df['score'] = df['score'].astype(float)
        if benchmark_scores_dict:
            benchmark_df = benchmark_scores_dict[feat_combo].rename(columns={'{}_score'.format(feat_combo): 'score'}).join(outcomes, how='inner')
            benchmark_df['score'] = benchmark_df['score'].astype(float)
        else:
            benchmark_df = None

        if feat_combo != 'G':
            plot_halves_km(df.dropna(),
                           'figures/km_plots/{}_km_{}.svg'.format(file_name_prefix + 'halves', feat_combo), # .png
                           'results/{}_{}_2groups_survival.csv'.format(file_name_prefix, feat_combo),
                           benchmark_df,
                           ylab=ylab,
                           width=2.5 if feat_combo == 'R' else 2.25,
                           height=1.88 if (feat_combo in ['GRH', 'RH']) and ('OS' in ylab) else 2)

def make_unimodal_c_barplots(train_df_, test_df_):
    train_df = train_df_.copy()
    train_df['split'] = 'train'
    test_df = test_df_.copy()
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], axis=0)
    plt.rc('xtick',labelsize=FONTSIZE)
    for modalities, plotting_df in df.groupby('model'):
        fig = plt.figure(figsize=(0.75, 2), constrained_layout=True)
        sns.barplot(data=plotting_df, x='split', y='c', palette='Greys')
        plt.errorbar(x=range(len(plotting_df)),
                     y=plotting_df.c,
                     fmt='none',
                     yerr=plotting_df[['lower_ci', 'upper_ci']].values.T,
                     ecolor='.2',
                     elinewidth=1.0,
                     alpha=0.75)
        ax = plt.gca()
        ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1.0)
        # plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
        y_lower_lim = min(0.49, (plotting_df['c'] - plotting_df['lower_ci']).min() - 0.01)
        y_upper_lim = (plotting_df['c'] + plotting_df['upper_ci']).max() + 0.01
        ax.set_ylim(y_lower_lim, y_upper_lim)
        plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
        plt.xlabel('')
        plt.ylabel('c-Index')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.savefig('figures/barplots/{}_c_barplot.svg'.format(modalities.replace('\n', '')))
    plt.rc('xtick',labelsize=FONTSIZE-2)
    plt.close()

def make_c_barplot(plotting_df, file_name_prefix):
    fig = plt.figure(figsize=(2, 2), constrained_layout=True)
    sns.barplot(data=plotting_df, x='model', y='c', palette='muted')
    plt.errorbar(x=range(len(plotting_df)),
                 y=plotting_df.c,
                 fmt='none',
                 yerr=plotting_df[['lower_ci', 'upper_ci']].values.T,
                 ecolor='.2',
                 elinewidth=1.0,
                 alpha=0.75)
    ax = plt.gca()
    ax.axhline(0.5, color='k', linestyle='dashed', linewidth=1.0)
    # plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    y_lower_lim = min(0.49, (plotting_df['c'] - plotting_df['lower_ci']).min() - 0.01)
    y_upper_lim = (plotting_df['c'] + plotting_df['upper_ci']).max() + 0.02
    for i, (_, row) in enumerate(plotting_df.iterrows()):
        if row['permutation_p'] <= 0.05:
            plt.text(i-0.29, row['c'] + row['upper_ci'], '*', fontsize=FONTSIZE, color='.2')
    ax.set_ylim(y_lower_lim, y_upper_lim)
    plt.xlabel('')
    plt.ylabel('c-Index')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig('figures/barplots/{}_c_barplot.svg'.format(file_name_prefix))
    # plt.savefig('figures/barplots/{}_c_barplot.png'.format(file_name_prefix), dpi=300)
    plotting_df['model'] = plotting_df['model'].apply(lambda x: x.replace('\n', ''))
    plotting_df['lower_bound'] = plotting_df['c'] - plotting_df['lower_ci']
    plotting_df['upper_bound'] = plotting_df['c'] + plotting_df['upper_ci']
    plotting_df.to_csv('results/{}_c.csv'.format(file_name_prefix), index=False)
    plt.close()

def make_crs_plot(df, output_file_name, n_classes):
    assert n_classes in [2, 4]
    if n_classes == 2:
        order = ["1/2", "3/NET"]
        figsize = (1,2.25)
    else:
        order = ['1', '2', '3', 'NET']
        figsize = (2,2)

    fig = plt.figure(figsize=figsize, constrained_layout=n_classes==2)
    ax = plt.gca()
    sns.boxplot(data=df, x='CRS', y='score', showfliers=False, palette='mako', order=order, linewidth=0.5, ax=ax)
    sns.swarmplot(data=df, x='CRS', y='score', color=".2", order=order, sizes=[2]*len(df), ax=ax)
    #print(output_file_name)
    if n_classes == 2:
        add_stat_annotation(ax, data=df, x='CRS', y='score', order=order,
                        box_pairs=[("1/2", "3/NET")],
                        perform_stat_test=False, pvalues=[mannwhitneyu(df.loc[df['CRS'] == "1/2", 'score'], df.loc[df['CRS'] == "3/NET", 'score'], alternative='less').pvalue],
                         text_format='star', loc='outside', verbose=0) # test='Mann-Whitney', comparisons_correction=None,
    else:
        p = kruskal(*[df.loc[df['CRS'] == CRS, 'score'].values.ravel() for CRS in order]).pvalue
        plt.text(1, df.score.max() + 0.1, "P={:4.3f}".format(p), fontsize=FONTSIZE-1)
        plt.subplots_adjust(top=0.8)
        plt.tight_layout()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(output_file_name, dpi=300 if '.svg' not in output_file_name else None)
    plt.close()

def make_crs_plots(scores_dict, file_name_prefix):
    crs_scores = load_crs(binarize=True)
    #print(crs_scores)
    for feat_combo, score_df in scores_dict.items():
        df = score_df.rename(columns={'{}_score'.format(feat_combo): 'score'})
        df.score = df.score.astype(float)
        df = df.join(crs_scores, how='inner')
        df.to_csv('results/crs/{}_crs_{}.csv'.format(file_name_prefix, feat_combo))
        #print('{}: {} cases'.format(feat_combo, len(df)))
        make_crs_plot(df,
                      'figures/crs_plots/{}_crs_{}.svg'.format(file_name_prefix, feat_combo), #.png
                      n_classes=2)


def make_multimodal_correlation_plot(scores, file_name_prefix, method='kendall'):
    df = scores['G']
    df['G_score'] = df['G_score'].astype(float)
    for col in scores.keys():
        if col == 'G':
            continue
        if col not in ['R', 'H', 'C', 'GRH']:
            continue
        df = df.join(scores[col], how='outer')
    df = df.rename(columns=dict([(x, x.replace('_score','')) for x in df.columns]))
    df = df.corr(method)
    mask = np.triu(np.ones_like(df, dtype=bool))
    np.fill_diagonal(mask, 0)

    # Draw the heatmap with the mask and correct aspect ratio
    fig, ax = plt.subplots(figsize=(2.17,2.17), constrained_layout=True)
    sns.heatmap(df, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .4}, ax=ax)
    # plt.tight_layout()
    # plt.savefig('figures/multimodal/{}_corr.png'.format(file_name_prefix), dpi=300)
    plt.savefig('figures/multimodal/{}_corr.svg'.format(file_name_prefix))
    df.to_csv('results/{}_corr.csv'.format(file_name_prefix))
    plt.close()


def plot_unimodal_scores(scores_dict, outcomes, file_name_prefix):
    df = scores_dict['G'].join(scores_dict['H'], how='inner').join(scores_dict['R'], how='inner').join(outcomes, how='inner')
    df = df[df.observed]
    df['R_rank'] = rankdata(df['R_score'])
    df['R_rank'] = 1 - (df['R_rank'] / (df['R_rank'].max() - df['R_rank'].min()))
    df['H_rank'] = rankdata(df['H_score'])
    df['H_rank'] = 1 - (df['H_rank'] / (df['H_rank'].max() - df['H_rank'].min()))

    df.loc[df.G_score == 1, 'subtype'] = 'HRD'
    df.loc[df.G_score == 0, 'subtype'] = 'HRP'
    df = df.rename(columns={'H_rank': 'histologic risk quantile', 'R_rank': 'radiologic risk quantile', 'subtype': 'genomic type'})
    h_col = 'histologic risk quantile'
    r_col = 'radiologic risk quantile'

    df = df[[h_col, r_col, 'genomic type', 'duration']]

    df['OS percentile'] = df.duration.apply(lambda x: percentileofscore(df.duration, x))
    df.to_csv('results/{}_unimodal_scores.csv'.format(file_name_prefix))
    fig = plt.figure(figsize=(3.35, 1.88), constrained_layout=True)
    sns.scatterplot(data=df, x=h_col, y=r_col, style='genomic type', hue='OS percentile', palette='vlag_r', edgecolor="0.2", alpha=0.8, sizes=[10.]*len(df),
        markers = {'HRD': 'o', 'HRP': 's'})
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plt.savefig('figures/multimodal/{}_scores.png'.format(file_name_prefix), dpi=300)
    plt.savefig('figures/multimodal/{}_scores.svg'.format(file_name_prefix))
    plt.close()

def train_multimodal(feature_set, multimodal_train_features, multimodal_test_features, strategy):
    if strategy == 'late':
        cols = [col + '_score' for col in feature_set]
    else:
        cols = []
        if 'G' in feature_set:
            cols.extend(train_genomic_features.columns) 
        if 'H' in feature_set:
            cols.extend(train_pathomic_features.columns) 
        if 'R' in feature_set:
            cols.extend(train_radiomic_features.columns) 
        if 'C' in feature_set:
            cols.extend(train_clinical_features.columns) 

    feat_acronym = ''.join(feature_set)
    if False:#('G' in feat_acronym):# or ('C' in feat_acronym):
        model = train_stratified(multimodal_train_features[cols].dropna(), survival_outcomes, feat_acronym, penalty=0.5)
    else:
        model = train(multimodal_train_features[cols].dropna(), survival_outcomes, feat_acronym, penalty=0.5)
    
    _train_scores = {feat_acronym: infer(multimodal_train_features[cols].dropna(),
                                       survival_outcomes, model,
                                       '{}_score'.format(feat_acronym))}
    _train_c = {feat_acronym: c_index(X=_train_scores[feat_acronym],
            y=survival_outcomes,
            score_col='{}_score'.format(feat_acronym),
            label='{} train'.format(feat_acronym))}
    _pfs_train_c = {feat_acronym: c_index(X=_train_scores[feat_acronym],
            y=pfs,
            score_col='{}_score'.format(feat_acronym),
            label='{} train'.format(feat_acronym))}


    _test_scores = {feat_acronym: infer(multimodal_test_features[cols].dropna(),
                                      survival_outcomes, model,
                                      '{}_score'.format(feat_acronym))}
    _test_c = {feat_acronym: c_index(X=_test_scores[feat_acronym],
            y=survival_outcomes,
            score_col='{}_score'.format(feat_acronym),
            label='{} test'.format(feat_acronym))}
    _pfs_test_c = {feat_acronym: c_index(X=_test_scores[feat_acronym],
            y=pfs,
            score_col='{}_score'.format(feat_acronym),
            label='{} test'.format(feat_acronym))}


    return {'train_score_update': _train_scores,
            'test_score_update': _test_scores,
            'train_c_update': _train_c,
            'pfs_train_c_update': _pfs_train_c,
            'test_c_update': _test_c,
            'pfs_test_c_update': _pfs_test_c}


def convert_to_df(c_dict_, file_name_prefix):
    c_dict = deepcopy(c_dict_)
    c_dict = list(c_dict.items())
    c_dict.sort(key=lambda x: x[-1]['c'])
    plotting_df = pd.DataFrame({'model': ['\n'.join(x[0]) for x in c_dict],
                                'c': [x[1]['c'] for x in c_dict],
                                'lower_ci': [x[1]['lower_ci'] for x in c_dict],
                                'upper_ci': [x[1]['upper_ci'] for x in c_dict],
                                'permutation_p': [x[1]['permutation_p'] for x in c_dict]})
    plotting_df['model'].apply(lambda x: x.replace('\n', '')).to_csv('results/{}_c.csv'.format(file_name_prefix), index=False)
    return plotting_df

if __name__ == '__main__':
    survival_outcomes = load_os()
    pfs = load_pfs()

    train_scores = {}
    test_scores = {}
    train_c = {}
    test_c = {}
    pfs_train_c = {}
    pfs_test_c = {}

###### GENOMIC ###
    genomic_features = load_genom()
    #genomic_features = genomic_features.loc[genomic_features.index.isin(load_all_ids(imaging_only=True))]
    train_genomic_features, test_genomic_features = split_train_test(genomic_features)


    # plot KM by HRD status alone
    plot_genom_KM(genomic_features.join(survival_outcomes, how='inner'),
                  'figures/km_plots/km_both_g.svg',
                  'results/both_G_2groups_survival.csv',
                  ylab="proportion (OS)")
    plot_genom_KM(train_genomic_features.join(survival_outcomes, how='inner'),
                  'figures/km_plots/km_train_g.svg',
                  'results/train_G_2groups_survival.csv',
                  ylab="proportion (OS)",
                  source_file_name=None)#'/Users/boehmk/shahLab/thesis/manuscript_submission/editorial edits/Source Data/assembly/Ex2d.csv')
    plot_genom_KM(test_genomic_features.join(survival_outcomes, how='inner'),
                  'figures/km_plots/km_test_g.svg',
                  'results/test_G_2groups_survival.csv',
                  ylab="proportion (OS)",
                  source_file_name=None)#'/Users/boehmk/shahLab/thesis/manuscript_submission/editorial edits/Source Data/assembly/Ex2e.csv')

    # evaluate
    train_scores['G'] = train_genomic_features.rename(columns={'hrd_status': 'G_score'})
    train_c['G'] = c_index(X=train_scores['G'], y=survival_outcomes, score_col='G_score', label='G train')
    pfs_train_c['G'] = c_index(X=train_scores['G'], y=pfs, score_col='G_score', label='G train')

    test_scores['G'] = test_genomic_features.rename(columns={'hrd_status': 'G_score'})
    test_c['G'] = c_index(X=test_scores['G'], y=survival_outcomes, score_col='G_score', label='G test')
    pfs_test_c['G'] = c_index(X=test_scores['G'], y=pfs, score_col='G_score', label='G test')

###### RADIOMIC ###
    train_radiomic_features, test_radiomic_features = split_train_test(load_radiomic_features())

    # scale features
    train_radiomic_features, radiomic_scaler = preprocess_features(train_radiomic_features, 5)
    test_radiomic_features, _ = preprocess_features(test_radiomic_features, 5, scaler=radiomic_scaler)

    # select features
    radiomic_features = select_imaging_features(train_radiomic_features,
                                                survival_outcomes,
                                                pd.read_csv(os.path.join(CODE_DIR, 'code', 'feature-selection', 'results', 'hr_ct_features_omentum.csv')),
                                                filter_='wavelet',
                                                remove_filter='firstorder',
                                                thresh=0.05,
                                                modality='radiology')
    #print(radiomic_features)
    plot_features(train_radiomic_features, survival_outcomes, radiomic_features)
    train_radiomic_features = train_radiomic_features[radiomic_features]
    test_radiomic_features = test_radiomic_features[radiomic_features]

    # train radiomic model
    radiomic_model = train(train_radiomic_features, survival_outcomes, 'R')

    # evaluate
    train_scores['R'] = infer(train_radiomic_features, survival_outcomes, radiomic_model, 'R_score')
    train_c['R'] = c_index(X=train_scores['R'], y=survival_outcomes, score_col='R_score', label='R train')
    pfs_train_c['R'] = c_index(X=train_scores['R'], y=pfs, score_col='R_score', label='R train')

    test_scores['R'] = infer(test_radiomic_features, survival_outcomes, radiomic_model, 'R_score')
    test_c['R'] = c_index(X=test_scores['R'], y=survival_outcomes, score_col='R_score', label='R test')
    pfs_test_c['R'] = c_index(X=test_scores['R'], y=pfs, score_col='R_score', label='R test')


###### PATHOMIC ###
    train_pathomic_features, test_pathomic_features = split_train_test(load_pathomic_features())
    train_pathomic_features, pathomic_scaler = preprocess_features(train_pathomic_features, 5)
    test_pathomic_features, _ = preprocess_features(test_pathomic_features, 5, scaler=pathomic_scaler)

    # select features
    pathomic_features = select_imaging_features(train_pathomic_features,
                                                survival_outcomes,
                                                pd.read_csv(os.path.join(CODE_DIR, 'code', 'feature-selection', 'results', 'hr_hne_features.csv')),
                                                thresh=0.05,
                                                modality='pathology',
                                                filter_='Stroma|Tumor|Necrosis',
                                                use_corrected=False)
    #print(pathomic_features)
    plot_features(train_pathomic_features, survival_outcomes, pathomic_features)
    train_pathomic_features = train_pathomic_features[pathomic_features]
    test_pathomic_features = test_pathomic_features[pathomic_features]

    # train pathomic model
    pathomic_model = train(train_pathomic_features, survival_outcomes, 'H')

    # evaluate
    train_scores['H'] = infer(train_pathomic_features, survival_outcomes, pathomic_model, 'H_score')
    train_c['H'] = c_index(X=train_scores['H'], y=survival_outcomes, score_col='H_score', label='H train')
    pfs_train_c['H'] = c_index(X=train_scores['H'], y=pfs, score_col='H_score', label='H train')

    test_scores['H'] = infer(test_pathomic_features, survival_outcomes, pathomic_model, 'H_score')
    test_c['H'] = c_index(X=test_scores['H'], y=survival_outcomes, score_col='H_score', label='H test')
    pfs_test_c['H'] = c_index(X=test_scores['H'], y=pfs, score_col='H_score', label='H test')

###### CLINICAL ###
    clin = load_clin(cols=['Complete gross resection', 'stage', 'age', 'Type of surgery', 'adnexal_lesion', 'omental_lesion', 'Received PARPi'])#, 'date_diagnosis'
    #clin = clin.loc[clin.index.isin(load_all_ids(True))]
    train_clinical_features, test_clinical_features = split_train_test(clin)

    train_clinical_features, test_clinical_features = encode_clinical_features(train_clinical_features, test_clinical_features)

    # scale clinical features
    train_clinical_features, test_clinical_features = scale(train_clinical_features, test_clinical_features, 'age')

    # select clinical features
    clinical_feats = select_clinical_features(train_clinical_features, survival_outcomes, threshold=0.05) #CHANGED 13/12/2021
    train_clinical_features = train_clinical_features[clinical_feats].dropna()
    test_clinical_features = test_clinical_features[clinical_feats].dropna()
    #print(clinical_feats)
    
    # train clinical model
    clinical_model = train(train_clinical_features, survival_outcomes, 'C')

    # evaluate
    train_scores['C'] = infer(train_clinical_features, survival_outcomes, clinical_model, 'C_score')
    train_c['C'] = c_index(X=train_scores['C'], y=survival_outcomes, score_col='C_score', label='C train')
    pfs_train_c['C'] = c_index(X=train_scores['C'], y=pfs, score_col='C_score', label='C train')

    test_scores['C'] = infer(test_clinical_features, survival_outcomes, clinical_model, 'C_score')
    test_c['C'] = c_index(X=test_scores['C'], y=survival_outcomes, score_col='C_score', label='C test')
    pfs_test_c['C'] = c_index(X=test_scores['C'], y=pfs, score_col='C_score', label='C test')


###### MULTIMODAL ###
    strategy = 'late'
    if strategy == 'late':
        # late fusion
        multimodal_train_features = pd.concat([df for df in train_scores.values()], axis=1)
        multimodal_test_features = pd.concat([df for df in test_scores.values()], axis=1)
    else:
        # early fusion
        multimodal_train_features = train_genomic_features.join(
                                    train_pathomic_features, how='inner').join(
                                    train_radiomic_features, how='inner').join(
                                    train_clinical_features, how='inner')
        multimodal_test_features = test_genomic_features.join(
                                   test_pathomic_features, how='inner').join(
                                   test_radiomic_features, how='inner').join(
                                   test_clinical_features, how='inner')

    feature_sets = list(combinations(train_scores.keys(), 4))
    feature_sets.extend(list(combinations(train_scores.keys(), 3)))
    feature_sets.extend(list(combinations(train_scores.keys(), 2)))
    # train and infer
    updates = Parallel(n_jobs=len(feature_sets))(delayed(
        train_multimodal)(feature_set, multimodal_train_features, multimodal_test_features, strategy) for feature_set in feature_sets)
    for update in updates:
        train_scores.update(update['train_score_update'])
        train_c.update(update['train_c_update'])
        pfs_train_c.update(update['pfs_train_c_update'])

        test_scores.update(update['test_score_update'])
        test_c.update(update['test_c_update'])
        pfs_test_c.update(update['pfs_test_c_update'])


###### PLOTS AND RESULTS ###
    # test CRS correspondence
    make_crs_plots(train_scores, 'train')
    make_crs_plots(test_scores, 'test')
    #all_scores = {}
    #for key in train_scores.keys():
    #    all_scores[key] = pd.concat([train_scores[key], test_scores[key]], axis=0)
    #make_crs_plots(all_scores, 'all_')

    # make c-index barplots for PFS
    pfs_train_c = convert_to_df(pfs_train_c, 'train')
    pfs_test_c = convert_to_df(pfs_test_c, 'test')
    make_c_barplot(pfs_train_c, 'pfs_train')
    make_c_barplot(pfs_test_c, 'pfs_test')

    # make c-index barplots for OS
    train_c = convert_to_df(train_c, 'train')
    test_c = convert_to_df(test_c, 'test')
    make_unimodal_c_barplots(train_c, test_c)
    make_c_barplot(train_c, 'train')
    make_c_barplot(test_c, 'test')

    # make PFS KM plots
    make_km_plots(test_scores, pfs, 'pfs_test_', benchmark_scores_dict=train_scores, ylab='proportion (PFS)')
    make_km_plots(train_scores, pfs, 'pfs_train_', ylab='proportion (PFS)')

    # make OS KM plots
    make_km_plots(test_scores, survival_outcomes, 'test_', benchmark_scores_dict=train_scores)
    make_km_plots(train_scores, survival_outcomes, 'train_')

    # make multimodal correlation plot
    make_multimodal_correlation_plot(test_scores, 'test')
    make_multimodal_correlation_plot(train_scores, 'train')
    make_multimodal_correlation_plot(test_scores, 'test_spearman', method='spearman')
    make_multimodal_correlation_plot(test_scores, 'test_pearson', method='pearson')
    make_multimodal_correlation_plot(train_scores, 'train_spearman', method='spearman')
    make_multimodal_correlation_plot(train_scores, 'train_pearson', method='pearson')

    # plot unimodal scores
    plot_unimodal_scores(test_scores, survival_outcomes, 'test')
    plot_unimodal_scores(train_scores, survival_outcomes, 'train')

