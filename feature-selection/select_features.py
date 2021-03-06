import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lifelines
import yaml
import os
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import spearmanr, pearsonr, kendalltau, zscore, percentileofscore, chisquare, power_divergence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pingouin import partial_corr
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from joblib import Parallel, delayed 
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests

warnings.simplefilter("ignore")

FONTSIZE = 7
plt.rc('legend',fontsize=FONTSIZE, title_fontsize=FONTSIZE)
plt.rc('xtick',labelsize=FONTSIZE)
plt.rc('ytick',labelsize=FONTSIZE)
plt.rc("axes", labelsize=FONTSIZE)

def evaluate_feature_partial_correlation(df, feat_col, y_col, covariate_col, x_covariate_col, y_covariate_col, method):
    """
    :param df: pandas DataFrame with each row being a patient and each column being a feature or outcome, containing column "duration" (float) of survival time
    :param feat_col: column name (str) of feature values (float)
    :param y_col: column name (str) of outcomes (float)
    :param covariate_col: column name (str) of XY covariate (float)
    :param x_covariate_col: column name (str) of X covariate (float)
    :param y_covariate_col: column name (str) of Y covariate (float)
    :param method: name (str) of method supported by pingouin.partial_corr
    :return: single-entry dict with {feat_col (str): [p (float), corr (float)]}
    """
    results = partial_corr(data=df,
                           x=feat_col,
                           y=y_col,
                           y_covar=y_covariate_col,
                           x_covar=x_covariate_col,
                           covar=covariate_col,
                           method=method)
    corr = results['r'].item()
    p = results['p-val'].item()
    if np.isnan(p):
        p = 1.0
        corr = 0.0
    return {feat_col: [p, corr]}


def evaluate_feature_cox(df, feat_col, covar):
    """
    :param df: pandas DataFrame with each row being a patient and each column being a feature or outcome, containing columns "duration" (float) of survival time and "observed" (float) to delineate observed [1.0] or censored [0.0] outcomes 
    :param feat_col: column name (str) of feature values (float)
    :param covar_col: column name (str) of XY covariate (float)
    :return: single-entry dict with {feat_col (str): [p (float), log(partial_hazard_ratio) (float)]}
    """

    col_list = ['duration', 'observed', feat_col]
    if covar:
        col_list.append(covar)

    model = CoxPHFitter(penalizer=0.0)
    try:
        model.fit(df[col_list], duration_col='duration', event_col='observed')
    except (lifelines.exceptions.ConvergenceError, lifelines.exceptions.ConvergenceWarning) as e:
        try:
            model = CoxPHFitter(penalizer=0.2)
            model.fit(df[col_list], duration_col='duration', event_col='observed')
        except (lifelines.exceptions.ConvergenceError, lifelines.exceptions.ConvergenceWarning) as e:
            return {feat_col: [1.0, 0.0]}

    coef = model.summary.coef[feat_col]
    p = model.summary.p[feat_col]

    return {feat_col: [p, coef]}    


def evaluate_feature_concordance(df, feat_col, k_permutations=100):
    """
    :param df: pandas DataFrame with each row being a patient and each column being a feature or outcome, containing columns "duration" (float) of survival time and "observed" (float) to delineate observed [1.0] or censored [0.0] outcomes 
    :param feat_col: column name (str) of feature values (float)
    :param k_permutations: number of interations (int) for permutation test to assess statistical significance
    :return: single-entry dict with {feat_col (str): [p (float), log(partial_hazard_ratio) (float)]}
    """

    c = lifelines.utils.concordance_index(event_times=df['duration'],
                                          predicted_scores=df[feat_col],
                                          event_observed=df['observed'])
    directional_deviance = c - 0.5
    absolute_deviance = np.abs(directional_deviance)

    random_absolute_deviances = []
    for _ in range(k_permutations):
        random_c = lifelines.utils.concordance_index(event_times=df['duration'],
                                          predicted_scores=df[feat_col].sample(frac=1),
                                          event_observed=df['observed'])
        random_absolute_deviances.append(np.abs(random_c - 0.5))
    p = (np.array(random_absolute_deviances) >= absolute_deviance).mean()
    return {feat_col: [p, directional_deviance]}


def evaluate_features(df, feats_to_consider, method='kendall', covar=None, x_covar=None, y_covar=None, n_jobs=-1):
    """
    :param df: pandas DataFrame with each row being a patient and each column being a feature or outcome
    :param feats_to_consider: list of column names (str) identifying feature columns
    :param method: name (str) of method supported by pingouin.partial_corr, "cph" for Cox regression, or "c-index" for concordance assessment
    :param covar: column name (str) of XY covariate (float)
    :param x_covar: column name (str) of X covariate (float)
    :param y_covar: column name (str) of Y covariate (float)
    :param n_jobs: number of parallel jobs to run for feature assessment
    :return: pandas DataFrame with columns ['feat', 'p', 'stat']
    """
    assert not df.isna().sum().any()
    assert ('duration' in df.columns) and ('observed' in df.columns)

    if method == 'cph':
        assert (not x_covar) and (not y_covar)
        dicts = Parallel(n_jobs=n_jobs)(delayed(evaluate_feature_cox)
            (df, effect, covar) for effect in feats_to_consider)
    elif method == 'c-index':
        assert (not covar) and (not x_covar) and (not y_covar)
        dicts = Parallel(n_jobs=n_jobs)(delayed(evaluate_feature_concordance)
            (df, effect) for effect in feats_to_consider)
    elif method in ['pearson', 'spearman']:
        assert (not (covar and x_covar)) and (not (covar and y_covar))
        dicts = Parallel(n_jobs=n_jobs)(delayed(evaluate_feature_partial_correlation)
            (df[df.observed], effect, 'duration', covar, x_covar, y_covar, method) for effect in feats_to_consider)
    else:
        raise RuntimeError("Unknown method {}".format(method))

    results = {}
    for dict_ in dicts:
        results.update(dict_)
    results = list(results.items())

    feats = np.array([x[0] for x in results])
    p_values = np.array([x[1][0] for x in results])
    stat_values = np.array([x[1][1] for x in results])
    results_df = pd.DataFrame({'feat': feats, 'p': p_values, 'stat': stat_values})

    if method == 'response-agnostic':
        results_df['p'] = results_df.stat.apply(lambda x: (100 - percentileofscore(results_df.stat, x))/100)
    results_df = results_df.sort_values(by='p')

    return results_df


def _get_features_to_consider(all_columns, args):
    """
    :param all_columns: list of columns in input dataframe
    :param args: parsed arguments
    :return: list of actual feature-containing columns (excluding covariates and outcomes)
    """
    columns_to_exclude = ['duration', 'observed', 'Unnamed: 0']
    for col in [args.xy_covar, args.x_covar, args.y_covar, args.index_col]:
        if col:
            columns_to_exclude.append(col)
    return list(set(all_columns) - set(columns_to_exclude))


def preprocess_features(df, outlier_threshold, feature_names=None, scaler=None):
    """
    :param df: input dataframe with features
    :outlier threshold: number of std devs above which we define an outlier
    :feature_names: list of feature names (str)
    :return: dataframe with processed features (removed outliers, scaled to zero mean, unit variance)
    """
    if feature_names:
        x = df[feature_names].copy(deep=True)
        y = df.drop(columns=feature_names).copy(deep=True)
    else:
        x = df

    # x[(zscore(x.values.astype(float))) < -outlier_threshold] = -outlier_threshold
    # x[(zscore(x.values.astype(float))) > outlier_threshold] = outlier_threshold
    if outlier_threshold != -1:
        x[(np.abs(zscore(x.values.astype(float))) > outlier_threshold)] = np.nan
        x = x.fillna(x.median())
    if not scaler:
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler = RobustScaler()
        x = pd.DataFrame(scaler.fit_transform(x.values), columns=x.columns, index=x.index)
    else:
        x = pd.DataFrame(scaler.transform(x.values), columns=x.columns, index=x.index)
    # scaler = RobustScaler()

    if outlier_threshold != -1:
        x[x>outlier_threshold]=outlier_threshold
        x[x<-outlier_threshold]=-outlier_threshold
    else:
        x[x>5]=5
        x[x<-5]=-5

    if feature_names:
        df = x.join(y)
    else:
        df = x
    return df, scaler


def _get_x_axis_name_volcano(method):
    if method == 'kendall':
        x_axis_name = "Kendall's $\\tau$"
    elif method == 'spearman':
        x_axis_name = "Spearman's $\\rho$"
    elif method == 'pearson':
        x_axis_name = "Pearson's $\\rho$"
    elif method == 'response-agnostic':
        x_axis_name = "IQR"
    elif method == 'cph':
        x_axis_name = "log(Hazard ratio)"
    elif method == 'c-index':
        x_axis_name = "concordance (deviation from random)"
    else:
        raise RuntimeError("Cannot generate volcano plot x-axis name for method {}".format(method))
    return x_axis_name


def _make_results_df_pretty_for_plotting(df_, x_axis_name, modality, eps=1e-30):
    df_.loc[df_.p < eps, 'p'] = eps
    df = pd.DataFrame({'feature': df_.feat,
                       '-log(p)': -np.log10(df_.p),
                       x_axis_name: df_.stat})
    if modality == 'radiology':
        try:
            for feat_name in df.feature:
                assert 'original-' not in feat_name
                assert 'diagnostic' not in feat_name
        except AssertionError:
            raise RuntimeError("Feature {} appears not to be a wavelet feature. The volcano plot color coding only supports wavelet feature.".format(feat_name))

        df['abbreviated_feature'] = df.feature.str.replace('wavelet-', '')
        df['abbreviated_feature'] = df.abbreviated_feature.str.replace('log-sigma-1-0-mm-3D_', 'LoG_')
        df['abbreviated_feature'] = df.abbreviated_feature.str.replace('log-sigma-3-0-mm-3D_', 'LoG_')
        
        df['abbreviated_feature'] = df['abbreviated_feature'].apply(lambda x: '_'.join([x.split('_')[0], x.split('_')[-1]]))

        df.Matrix = 'Other'
        for matrix in ['glszm', 'ngtdm', 'glcm', 'glrlm', 'gldm']: #, 'firstorder'
            mask = df.feature.str.contains(matrix)
            count_ = mask.sum()
            df.loc[mask, 'Matrix'] = matrix
    else:
        df['Feature'] = 'Other'
        for feat_type in ['Tumor_Other', 'Tumor_Lymphocyte', 'Stroma_Other', 'Stroma_Lymphocyte', 'Tumor', 'Necrosis', 'Stroma', 'Fat']:
            mask = (df.feature.str.contains(feat_type) |  df.feature.str.contains(feat_type.lower())) & (df['Feature'] == 'Other')
            count_ = mask.sum()
            # df.loc[mask, 'Matrix'] = '{} (n={})'.format(matrix, count_)
            df.loc[mask, 'Feature'] = feat_type.replace('_Other', ' Nuclei').replace('_Lymphocyte', ' Lymphocyte')
        df['abbreviated_feature'] = df['feature'].str.replace('_Other_', ' Nuclei ' ).str.replace('_Lymphocyte_', ' Lymphocyte ')

    df = df.sort_values(by='feature')
    return df    

def make_volcano_plot(df_, method, output_plot_path, modality, top_k_to_label=1):
    x_axis_name = _get_x_axis_name_volcano(method)

    df = _make_results_df_pretty_for_plotting(df_, x_axis_name, modality)
    df.loc[df[x_axis_name] < -3, x_axis_name] = -3
    df.loc[df[x_axis_name] > 3, x_axis_name] = 3

    if modality == 'radiology':
        hue_col = 'Matrix'
    else:
        hue_col = 'Feature'

    if modality == 'radiology':
        df = df.sort_values(by='-log(p)', ascending=True)
    
    # plt.rcParams["axes.labelsize"] = 13
    fig = plt.figure(figsize=(3, 2), constrained_layout=True)
    g = sns.scatterplot(data=df,
                        x=x_axis_name,
                        y='-log(p)',
                        hue=hue_col,
                        alpha=0.7,
                        palette='dark',
                        hue_order=['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'] if modality == 'radiology' else None,
                        s=6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_col)

    df = df.sort_values(by='-log(p)', ascending=False)
    ax = plt.gca()
    if modality == 'radiology':
        sig_threshold = get_ci_95_pval(df)
        if sig_threshold != -1:
            plt.axhline(y=sig_threshold, color='.2', linewidth=0.5, linestyle='-.')
        else:
            plt.gca().set_ylim((-0.20185706772068027, 4.2681049500679045))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if '.svg' in output_plot_path:
        plt.savefig(output_plot_path)
    else:
        plt.savefig(output_plot_path, dpi=300)
    plt.close()

def get_ci_95_pval(df):
    p_vals = df['-log(p)'].apply(lambda x: float(10)**(-x)).tolist()
    reject, pvals_corrected, _, _ = multipletests(pvals=p_vals, alpha=0.05, method='fdr_bh', is_sorted=True)
    if reject.sum() == 0:
        return -1
    arg_ = np.argmax(pvals_corrected>=0.05)
    return -np.log10((p_vals[arg_ - 1] + p_vals[arg_])/2)


if __name__ == '__main__':
    PARSER = ArgumentParser(description='select features for survival analysis')
    PARSER.add_argument('feature_df_path', type=str,
        help='path to pandas DataFrame with features and outcomes. all columns in the DF will be evaluated except "duration", "observed," and any covariates')
    PARSER.add_argument('--outcome_df_path', type=str, help='path to pd df with outcomes, optional', default=None)
    PARSER.add_argument('--train_id_df_path', type=str, help='path to pd df with train IDs, optional', default=None)

    PARSER.add_argument('--output_df_path', type=str, default='feature_evaluation.csv', help='path at which to save feature evaluation df')
    PARSER.add_argument('--output_plot_path', type=str, default='feature_evaluation.png', help='path at which to save feature evaluation volcano plot')
    PARSER.add_argument('--index_col', type=str, default='Patient ID', help="name of column to set as index")
    PARSER.add_argument('--outlier_std_threshold', type=float, default=5, help="number of standard deviations to use for clipping")
    PARSER.add_argument('--method', type=str, default='cph',
        help="'kendall' for Kendall's Tau, 'cph' for Cox Proportional Hazards, 'c-index' for Concordance, 'spearman,' 'response-agnostic', or 'pearson'")
    PARSER.add_argument('--xy_covar', type=str, default=None, help="XY covariate column name")
    PARSER.add_argument('--x_covar', type=str, default=None, help="X covariate column name")
    PARSER.add_argument('--y_covar', type=str, default=None, help="Y covariate column name")
    PARSER.add_argument('--n_jobs', type=int, default=-1, help="number of parallel jobs to use")
    PARSER.add_argument('--modality', type=str, default='radiology', help="radiology or pathology")
    ARGS = PARSER.parse_args()

    with open('../global_config.yaml', 'r') as f:
        CONFIGS = yaml.safe_load(f)
        DATA_DIR = CONFIGS['data_dir']
        CODE_DIR = CONFIGS['code_dir']

    DF = pd.read_csv(os.path.join(DATA_DIR, ARGS.feature_df_path)).set_index(ARGS.index_col)
    
    if ARGS.modality == 'radiology':
        DF = DF[[x for x in DF.columns if 'firstorder' not in x]]

    if ARGS.outcome_df_path:
        OUTCOME_DF = pd.read_csv(os.path.join(DATA_DIR, ARGS.outcome_df_path)).set_index(ARGS.index_col)[['duration.OS', 'observed.OS']]
        OUTCOME_DF = OUTCOME_DF.rename(columns={'duration.OS': 'duration', 'observed.OS': 'observed'})
        DF = DF.join(OUTCOME_DF, how='inner')
    
    if ARGS.train_id_df_path:
        DF = DF[DF.index.isin(pd.read_csv(os.path.join(DATA_DIR, ARGS.train_id_df_path))[ARGS.index_col])]

    try:
        assert not DF.isna().sum().any()
    except AssertionError:
        raise RuntimeError("Input dataframe must not contain any NaN values.")

    FEATURE_NAMES = _get_features_to_consider(DF.columns.tolist(), ARGS)

    DF, _ = preprocess_features(DF,
                                outlier_threshold=ARGS.outlier_std_threshold,
                                feature_names=FEATURE_NAMES)

    RESULTS = evaluate_features(df=DF,
                                feats_to_consider=FEATURE_NAMES,
                                method=ARGS.method,
                                covar=ARGS.xy_covar,
                                x_covar=ARGS.x_covar,
                                y_covar=ARGS.y_covar,
                                n_jobs=ARGS.n_jobs)
    RESULTS.to_csv(ARGS.output_df_path)
    make_volcano_plot(RESULTS, ARGS.method, ARGS.output_plot_path, ARGS.modality)
