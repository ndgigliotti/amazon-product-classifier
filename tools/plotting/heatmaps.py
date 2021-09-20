from typing import Callable, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from tools import utils
from tools.plotting.utils import HEATMAP_STYLE, heatmap_figsize


def pair_corr_heatmap(
    *,
    data: pd.DataFrame,
    ignore: Union[str, list] = None,
    annot: bool = True,
    high_corr: float = None,
    scale: float = 0.5,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a heatmap of the pairwise correlations in `data`.

    Parameters
    ----------
    data : DataFrame
        Data for pairwise correlations.
    ignore : str or list, optional
        Column or columns to ignore, by default None.
    annot : bool, optional
        Whether to annotate cells, by default True.
    high_corr : float, optional
        Threshold for high correlations, by default None. Causes cells
        to be colored in all-or-nothing fashion.
    scale : float, optional
        Scale multiplier for figure size, by default 0.5.
    ax : Axes, optional
        Axes to plot on, by default None.

    Returns
    -------
    Axes
        The heatmap.
    """
    if not ignore:
        ignore = []
    corr_df = data.drop(columns=ignore).corr()
    title = "Correlations Between Features"
    if ax is None:
        figsize = heatmap_figsize(corr_df.shape, scale)
        fig, ax = plt.subplots(figsize=figsize)
    if high_corr is not None:
        if annot:
            annot = corr_df.values
        corr_df = corr_df.abs() > high_corr
        kwargs["center"] = None
        title = f"High {title}"
    mask = np.triu(np.ones_like(corr_df, dtype="int64"), k=0)
    style = dict(HEATMAP_STYLE)
    style.update(kwargs)
    style.update({"annot": annot})
    ax = sns.heatmap(
        data=corr_df,
        mask=mask,
        ax=ax,
        **style,
    )
    ax.set_title(title, pad=10)
    return ax


def cat_corr_heatmap(
    *,
    data: pd.DataFrame,
    categorical: str,
    transpose: bool = False,
    high_corr: float = None,
    scale: float = 0.5,
    no_prefix: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a correlation heatmap of categorical vs. numeric features.

    Args:
        data (DataFrame): Frame containing categorical and numeric data.
        categorical (str): Name or list of names of categorical features.
        high_corr (float): Threshold for high correlation. Defaults to None.
        scale (float, optional): Multiplier for determining figsize. Defaults to 0.5.
        no_prefix (bool, optional): If only one cat, do not prefix dummies. Defaults to True.
        ax (Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        Axes: Axes of the plot.
    """
    if isinstance(categorical, str):
        ylabel = utils.title(categorical)
        categorical = [categorical]
        single_cat = True
    else:
        ylabel = "Categorical Features"
        single_cat = False
    title = "Correlation with Numeric Features"
    cat_df = data.filter(categorical, axis=1)
    if no_prefix and single_cat:
        dummies = pd.get_dummies(cat_df, prefix="", prefix_sep="")
    else:
        dummies = pd.get_dummies(cat_df)
    corr_df = dummies.apply(lambda x: data.corrwith(x))
    if not transpose:
        corr_df = corr_df.T
    if high_corr is not None:
        if "annot" not in kwargs or kwargs.get("annot"):
            kwargs["annot"] = corr_df.values
        corr_df = corr_df.abs() > high_corr
        kwargs["center"] = None
        title = f"High {title}"
    if ax is None:
        fig, ax = plt.subplots(figsize=heatmap_figsize(corr_df.shape, scale=scale))
    style = dict(HEATMAP_STYLE)
    style.update(kwargs)
    ax = sns.heatmap(corr_df, ax=ax, **style)
    xlabel = "Numeric Features"
    if transpose:
        xlabel, ylabel = ylabel, xlabel
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, pad=10)
    return ax


def confusion_matrix(
    estimator,
    X_test,
    y_test,
    *,
    labels=None,
    sample_weight=None,
    normalize="true",
    title_scorer=("accuracy", "balanced_accuracy"),
    center=0.5,
    annot=True,
    annot_kws=None,
    fmt=".2f",
    cmap="Blues",
    cbar=False,
    linewidths=0,
    linecolor="w",
    ax=None,
    size=None,
):

    y_pred = estimator.predict(X_test)

    if labels is None:
        labels = unique_labels(y_test, y_pred)

    cm = metrics.confusion_matrix(
        y_test, y_pred, sample_weight=sample_weight, labels=labels, normalize=normalize
    )

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    if size is None:
        size = heatmap_figsize(cm.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=size)

    sns.heatmap(
        cm,
        center=center,
        annot=annot,
        annot_kws=annot_kws,
        fmt=fmt,
        cmap=cmap,
        cbar=cbar,
        linewidths=linewidths,
        linecolor=linecolor,
    )

    if isinstance(title_scorer, (str, Callable)):
        title_scorer = [title_scorer]
    title = []
    for scorer in title_scorer:
        func = metrics.get_scorer(scorer)
        score = func(estimator, X_test, y_test, sample_weight=sample_weight)
        name = scorer.__name__ if hasattr(scorer, "__name__") else scorer
        name = name.title().replace("_", " ")
        title.append(f"{name}: {score:.2f}")
    title = ", ".join(title)

    ax.set(title=title, xlabel="Predicted Value", ylabel="True Value")
    return ax
