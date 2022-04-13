import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from utils import remove_collinear, remove_low_variance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from scipy.stats import spearmanr, pearsonr, kendalltau, zscore, percentileofscore, chisquare, power_divergence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pingouin import partial_corr

import lifelines
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from joblib import Parallel, delayed 
from argparse import ArgumentParser

from sklearn.cluster import KMeans

FONTSIZE = 8
plt.rc('legend',fontsize=FONTSIZE-2, title_fontsize=FONTSIZE-2)
plt.rc('xtick',labelsize=FONTSIZE-2)
plt.rc('ytick',labelsize=FONTSIZE-2)
plt.rc("axes", labelsize=FONTSIZE)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


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
    except lifelines.exceptions.ConvergenceError:
        try:
            model = CoxPHFitter(penalizer=0.2)
            model.fit(df[col_list], duration_col='duration', event_col='observed')
        except lifelines.exceptions.ConvergenceError:
            return {feat_col: [1.0, 0.0]}

    coef = model.summary.coef[feat_col]
    p = model.summary.p[feat_col]

    return {feat_col: [p, coef]}    


def evaluate_feature_concordance(df, feat_col, k_permutations=1000):
    """
    :param df: pandas DataFrame with each row being a patient and each column being a feature or outcome, containing columns "duration" (float) of survival time and "observed" (float) to delineate observed [1.0] or censored [0.0] outcomes 
    :param feat_col: column name (str) of feature values (float)
    :param k_permutations: number of interations (int) for permutation test to assess statistical significance
    :return: single-entry dict with {feat_col (str): [p (float), log(partial_hazard_ratio) (float)]}
    """

    c = lifelines.utils.concordance_index(event_times=df['duration'],
                                          predicted_scores=df[feat_col],
                                          event_observed=df['observed'])
    random_c_vals = []
    for _ in range(k_permutations):
        random_c = lifelines.utils.concordance_index(event_times=df['duration'],
                                          predicted_scores=df[feat_col].sample(frac=1),
                                          event_observed=df['observed'])
        random_c_vals.append(np.abs(random_c - 0.5))
    p = (np.array(random_c_vals) > np.abs(c - 0.5)).mean()
    return {feat_col: [p, c - 0.5]}


def evaluate_feature_response_agnostic(df, feat_col, covar=None):
    # var_ = df[feat_col].var()
    var_ = df[feat_col].quantile(0.75) - df[feat_col].quantile(0.25)

    # counts = np.histogram(df[feat_col], bins=50)[0]
    # if counts.max() > 0.5 * len(df):
    #     var_ = 0
    # expected = np.full_like(counts, float(len(df))/float(len(counts)))
    # _, chisq_p = chisquare(f_obs=counts, f_exp=expected)
    # _, chisq_p = power_divergence(f_obs=counts)
    # print(counts)
    # print(expected)
    # print(chisq_p)
    # exit()
    assert not df[feat_col].isna().sum()
    # kmeans = KMeans(n_clusters=2).fit(df[feat_col].values.reshape(-1, 1))
    # clumpiness = np.abs(df.loc[(kmeans.labels_==0).reshape(-1, 1), feat_col].mean() - df.loc[(kmeans.labels_==1).reshape(-1, 1), feat_col].mean())
    # clumpiness = kmeans.score(df[feat_col].values.reshape(-1,1))
    # print(clumpiness)
    # if clumpiness < -30:
        # var_ = 0
    n_duplicates = df[feat_col].duplicated(keep=False).sum()
    print(n_duplicates)
    if n_duplicates > len(df)/5:
        var_ = 0

    if covar:
        _, p = spearmanr(df[feat_col], df[covar])
        # if feat_col == 'Fat_largest_component_eccentricity':
        #     print(pearsonr(df[feat_col], df[covar]))
        #     print(feat_col)
        #     print(p)
        #     exit()
        if p<0.01:
            var_ = 0
    # print(var_)
    # exit()
    return {feat_col: [None, var_]}


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
    elif method == 'response-agnostic':
        assert (not x_covar) and (not y_covar)
        dicts = Parallel(n_jobs=n_jobs)(delayed(evaluate_feature_response_agnostic)
            (df, effect, covar) for effect in feats_to_consider)
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


def preprocess_features(df, outlier_threshold, feature_names):
    """
    :param df: input dataframe with features
    :outlier threshold: number of std devs above which we define an outlier
    :feature_names: list of feature names (str)
    :return: dataframe with processed features (removed outliers, scaled to zero mean, unit variance)
    """
    x = df[feature_names].copy(deep=True)
    y = df.drop(columns=feature_names).copy(deep=True)

    # x[(zscore(x.values.astype(float))) < -outlier_threshold] = -outlier_threshold
    # x[(zscore(x.values.astype(float))) > outlier_threshold] = outlier_threshold
    if outlier_threshold != -1:
        x[(np.abs(zscore(x.values.astype(float))) > outlier_threshold)] = np.nan
        x = x.fillna(x.median())
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    x = pd.DataFrame(scaler.fit_transform(x.values), columns=x.columns, index=x.index)

    if outlier_threshold != -1:
        x[x>outlier_threshold]=outlier_threshold
        x[x<-outlier_threshold]=-outlier_threshold
    else:
        x[x>5]=5
        x[x<-5]=-5

    return x.join(y)


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


def _make_results_df_pretty_for_plotting(df_, x_axis_name, modality, eps=1e-6):
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
        # df['abbreviated_feature'] = df.abbreviated_feature.str.replace('log-sigma-3-0-mm-3D_', 'LoG_')
        
        df['abbreviated_feature'] = df['abbreviated_feature'].apply(lambda x: '_'.join([x.split('_')[0], x.split('_')[-1]]))

        df.Matrix = 'Other'
        for matrix in ['glszm', 'ngtdm', 'glcm', 'glrlm', 'firstorder', 'gldm']:
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

    if modality == 'radiology':
        hue_col = 'Matrix'
    else:
        hue_col = 'Feature'

    # plt.rcParams["axes.labelsize"] = 13
    fig = plt.figure(figsize=(3, 2), constrained_layout=True)
    g = sns.scatterplot(data=df,
                        x=x_axis_name,
                        y='-log(p)',
                        hue=hue_col,
                        alpha=0.7,
                        palette='Set2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_col)
    # plt.tight_layout()

    ax = plt.gca()
    for index, row in df.sort_values(by='-log(p)', ascending=False)[:top_k_to_label].iterrows():
        x, y = row[x_axis_name], row['-log(p)']
        feat = row['abbreviated_feature']
        if x > 0:
            align = 'right'
            x_offset = -0.01 
        else:
            align = 'left'
            x_offset = 0.01
        ax.text(x+x_offset, y, feat, horizontalalignment=align, size=4, color='black')

    plt.savefig(output_plot_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    PARSER = ArgumentParser(description='select features for survival analysis')
    PARSER.add_argument('feature_df_path', type=str,
        help='path to pandas DataFrame with features and outcomes. all columns in the DF will be evaluated except "duration", "observed," and any covariates')
    PARSER.add_argument('--output_df_path', type=str, default='feature_evaluation.csv', help='path at which to save feature evaluation df')
    PARSER.add_argument('--output_plot_path', type=str, default='feature_evaluation.png', help='path at which to save feature evaluation volcano plot')
    PARSER.add_argument('--index_col', type=str, default='ID', help="name of column to set as index")
    PARSER.add_argument('--outlier_std_threshold', type=float, default=4.0, help="number of standard deviations to use as threshold for outlier detection and replacement by median")
    PARSER.add_argument('--method', type=str, default='kendall',
        help="'kendall' for Kendall's Tau, 'cph' for Cox Proportional Hazards, 'c-index' for Concordance, 'spearman,' 'response-agnostic', or 'pearson'")
    PARSER.add_argument('--xy_covar', type=str, default=None, help="XY covariate column name")
    PARSER.add_argument('--x_covar', type=str, default=None, help="X covariate column name")
    PARSER.add_argument('--y_covar', type=str, default=None, help="Y covariate column name")
    PARSER.add_argument('--n_jobs', type=int, default=-1, help="number of parallel jobs to use")
    PARSER.add_argument('--modality', type=str, default='radiology', help="radiology or pathology")
    PARSER.add_argument('--bootstrap', type=bool, default=False, help="bootstrap")
    ARGS = PARSER.parse_args()

    DF = pd.read_csv(ARGS.feature_df_path).set_index(ARGS.index_col)
    DF = DF[[x for x in DF.columns if 'firstorder' not in x]]

    if ARGS.bootstrap:
        DF = DF.sample(frac=0.95)


    try:
        assert not DF.isna().sum().any()
    except AssertionError:
        raise RuntimeError("Input dataframe must not contain any NaN values.")

    FEATURE_NAMES = _get_features_to_consider(DF.columns.tolist(), ARGS)

    DF = preprocess_features(DF, 
                             outlier_threshold=ARGS.outlier_std_threshold,
                             feature_names=FEATURE_NAMES)

    RESULTS = evaluate_features(df=DF,
                                feats_to_consider=FEATURE_NAMES,
                                method=ARGS.method,
                                covar=ARGS.xy_covar,
                                x_covar=ARGS.x_covar,
                                y_covar=ARGS.y_covar,
                                n_jobs=ARGS.n_jobs)
    print(RESULTS)

    RESULTS.to_csv(ARGS.output_df_path)

    features = RESULTS
    features = features[~features.feat.str.contains('firstorder')]
    features = features[~features.feat.str.contains('original')]
    features = features[~features.feat.str.contains('Unnamed')]
    print(features.feat.head().tolist())

    make_volcano_plot(RESULTS, ARGS.method, ARGS.output_plot_path, ARGS.modality)
