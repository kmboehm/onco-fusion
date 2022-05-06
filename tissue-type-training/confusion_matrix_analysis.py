import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import binom_test


FONTSIZE = 8
plt.rc('legend',fontsize=FONTSIZE, title_fontsize=FONTSIZE)
plt.rc('xtick',labelsize=FONTSIZE)
plt.rc('ytick',labelsize=FONTSIZE)
plt.rc("axes", labelsize=FONTSIZE)



def cm_analysis(y_true, y_pred, labels=None, ymap=None, figsize=(2.66,2.66), filename='confusion_matrix.pdf', acc_pval=False):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[int(yi)] for yi in y_pred]
        y_true = [ymap[int(yi)] for yi in y_true]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '{:.0f}%'.format(p)
            elif c == 0:
                annot[i, j] = ''
            else:
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '{:.0f}%'.format(p)

    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, square=True, cbar=False, annot_kws={'fontsize': FONTSIZE})
    if acc_pval:
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        _, counts = np.unique(y_true, return_counts=True)
        no_information_rate = np.max(counts) / float(len(y_true))
        plt.text(0, 0, 'p = {:4.3f}'.format(binom_test(x=np.sum(y_pred == y_true),
                                                       n=len(y_pred),
                                                       p=no_information_rate)))
    plt.savefig(filename, dpi=300 if '.svg' not in filename else None)
