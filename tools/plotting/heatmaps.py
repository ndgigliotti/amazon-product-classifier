from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .. import utils
from .utils import HEATMAP_STYLE, heatmap_figsize


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
