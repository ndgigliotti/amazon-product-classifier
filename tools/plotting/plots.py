from functools import partial, singledispatch
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from .._validation import _validate_orient, _validate_sort
from .annotate import add_tukey_marks, annot_bars
from .utils import cat_palette, flip_axis, heat_palette, set_invisible, smart_subplots


def mirror_plot(
    *,
    data: pd.DataFrame,
    x: str,
    y: str,
    left_estimator: Callable = np.sum,
    right_estimator: Callable = np.mean,
    sort_side="right",
    sort_dir="desc",
    size: Tuple[float, float] = (4, 8),
    **kwargs,
) -> plt.Figure:
    """Plot two horizontal bar graphs aligned back-to-back on the vertical axis.

    Parameters
    ----------
    data : DataFrame
        Data for plotting.
    x : str
        Variable for horizontal axis.
    y : str
        Variable for vertical axis.
    left_estimator : Callable, optional
        Estimator for left graph, by default np.sum.
    right_estimator : Callable, optional
        Estimator for right graph, by default np.mean.
    sort_side : str, optional
        Side to sort on, 'left' or 'right' (default).
    sort_dir : str, optional
        Sort direction, 'asc' or 'desc' (default).
    size : Tuple[float, float], optional
        Size of each subplot, by default (4, 8).

    Returns
    -------
    Figure
        Figure for plot.
    """
    if sort_side.lower() not in {"right", "left"}:
        raise ValueError("`sort_side` must be 'right' or 'left'")
    sort_est = left_estimator if sort_side.lower() == "left" else right_estimator
    order = data.groupby(y)[x].agg(sort_est).sort_values().index.to_numpy()
    if sort_dir.lower() == "desc":
        order = order[::-1]

    palette = cat_palette("deep", data.loc[:, y])
    barplot = partial(
        sns.barplot, data=data, y=y, x=x, order=order, palette=palette, **kwargs
    )
    fig, (ax1, ax2) = smart_subplots(nplots=2, size=size, ncols=2, sharey=True)
    barplot(ax=ax1, estimator=left_estimator)
    barplot(ax=ax2, estimator=right_estimator)

    ax1.set_ylabel(None)
    ax2.set_ylabel(None)
    flip_axis(ax1)
    fig.tight_layout()
    return fig


def grouper_plot(
    *,
    data: pd.DataFrame,
    grouper: str = None,
    x: str = None,
    y: str = None,
    kind: str = "line",
    ncols: int = 3,
    height: int = 4,
    **kwargs,
) -> plt.Figure:
    """Plot data by group; one subplot per group.

    Parameters
    ----------
    data : DataFrame
        Data to plot, by default None.
    grouper : str, optional
        Column to group by, by default None.
    x : str, optional
        Variable for x-axis, by default None.
    y : str, optional
        Variable for y-axis, by default None.
    kind : str, optional
        Kind of plot for Dataframe.plot().
        Options: 'line' (default), 'bar', 'barh', 'hist', 'box',
        'kde', 'density', 'area', 'pie', 'scatter', 'hexbin'.
    ncols : int, optional
        Number of subplot columns, by default 3
    height : int, optional
        Height of square subplots, by default 4

    Returns
    -------
    Figure
        The figure.
    """
    data.sort_values(x, inplace=True)
    grouped = data.groupby(grouper)
    fig, axs = smart_subplots(nplots=len(grouped), ncols=ncols, size=(height, height))
    set_invisible(axs)

    for ax, (label, group) in zip(axs.flat, grouped):
        group.plot(x=x, y=y, ax=ax, kind=kind, **kwargs)
        ax.set_title(label)
        ax.set_visible(True)

    fig.tight_layout()
    return fig


def multi_rel(
    *,
    data: pd.DataFrame,
    x: Union[str, list],
    y: str,
    kind="line",
    ncols: int = 3,
    size: Tuple[float, float] = (5.0, 5.0),
    sharey: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plot each `x`-value against `y` on line graphs.
    Parameters
    ----------
    data : DataFrame
        Data with distributions to plot.
    x : str or list-like of str
        Dependent variable(s).
    y : str
        Independent variable.
    kind : str, optional
        Kind of plot: 'line' (default), 'scatter', 'reg', 'bar'.
    ncols : int, optional
        Number of columns for subplots, defaults to 3.
    size : Tuple[float, float], optional.
        Size of each subpot, by default (5.0, 5.0).
    sharey: bool, optional
        Share the y axis between subplots. Defaults to True.
    Returns
    -------
    Figure
        Multiple relational plots.
    """
    fig, axs = smart_subplots(
        nplots=data.columns.size,
        ncols=ncols,
        size=size,
        sharey=sharey,
    )
    set_invisible(axs)
    kinds = dict(
        line=sns.lineplot, scatter=sns.scatterplot, reg=sns.regplot, bar=sns.barplot
    )
    plot = kinds[kind.lower()]

    for ax, column in zip(axs.flat, x):
        ax.set_visible(True)
        ax = plot(data=data, x=column, y=y, ax=ax, **kwargs)

    if axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.set_ylabel(None)
    elif axs.size > 1:
        for ax in axs[1:]:
            ax.set_ylabel(None)
    fig.tight_layout()
    return fig


def multi_dist(
    *,
    data: pd.DataFrame,
    tukey_marks: bool = False,
    ncols: int = 3,
    height: int = 5,
    **kwargs,
) -> plt.Figure:
    """Plot histograms for all numeric variables in `data`.

    Parameters
    ----------
    data : DataFrame
        Data with distributions to plot.
    tukey_marks : bool, optional
        Annotate histograms with IQR and Tukey's fences, by default False.
    ncols : int, optional
        Number of columns for subplots, by default 3.
    height : int, optional
        Subpot height, by default 5.

    Returns
    -------
    Figure
        Multiple histograms.
    """
    data = data.select_dtypes("number")
    fig, axs = smart_subplots(
        nplots=data.columns.size,
        ncols=ncols,
        size=(height, height),
    )
    set_invisible(axs)

    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        ax = sns.histplot(data=data, x=column, ax=ax, **kwargs)
        if tukey_marks:
            add_tukey_marks(data[column], ax, annot=False)
        ax.set_title(f"Distribution of '{column}'")

    if axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.set_ylabel(None)
    elif axs.size > 1:
        for ax in axs[1:]:
            ax.set_ylabel(None)
    fig.tight_layout()
    return fig


@singledispatch
def countplot(
    data: pd.Series,
    *,
    normalize: bool = False,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    orient: str = "h",
    sort: str = "desc",
    topn: int = None,
    annot: bool = True,
    size: Tuple[float, float] = (5, 5),
    ncols: int = 3,
    ax: plt.Axes = None,
    **kwargs,
) -> Union[plt.Axes, plt.Figure]:
    """Plot value counts of every feature in `data`.

    Parameters
    ----------
    data : Series
        Data to plot.
    normalize : bool, optional
        Show fractions instead of counts, by default False.
    heat : str, optional
        Color palette for heat, by default "coolwarm".
    heat_desat : float, optional
        Saturation of heat color palette, by default 0.6.
    orient : str, optional
        Bar orientation, by default "h".
    sort : str, optional
        Direction for sorting bars. Can be 'asc' or 'desc' (default).
    annot : bool, optional
        Annotate bars, True by default.
    size : int, optional
        Figure size, or size of each subplot. Ignored if `ax` is passed.
        Defaults to (5, 5).
    ncols : int, optional
        Number of columns for subplots, by default 3. Only relevant for DataFrames.
    ax : Axes, optional
        Axes to plot Series on. Raises ValueError if passed with DataFrame.

    Returns
    -------
    Axes or Figure
        Axes if passed Series, Figure if DataFrame.
    """
    _validate_orient(orient)
    _validate_sort(sort)
    orient = orient.lower()

    if ax is None:
        _, ax = plt.subplots(figsize=size)

    df = data.value_counts(normalize=normalize).to_frame("Count")
    if topn is not None:
        if topn <= df.shape[0]:
            df = df.iloc[:topn]
        else:
            raise ValueError("`topn` must be <= number of unique values.")
    df.index.name = data.name or "Series"
    df.reset_index(inplace=True)
    pal = heat_palette(df["Count"], heat, desat=heat_desat)
    ax = barplot(
        data=df,
        x=data.name or "Series",
        y="Count",
        ax=ax,
        orient=orient,
        sort=sort.lower(),
        palette=pal,
        **kwargs,
    )
    title = f"'{data.name}' Value Counts" if data.name else "Value Counts"
    ax.set(title=title)
    format_spec = "{x:.0%}" if normalize else "{x:,.0f}"
    if annot:
        annot_bars(ax, orient=orient, format_spec=format_spec)

    if orient == "h":
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(format_spec))
        ax.set(ylabel=None)
    else:
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(format_spec))
        ax.set(xlabel=None)
    return ax


@countplot.register
def _(
    data: pd.DataFrame,
    *,
    normalize: bool = False,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    orient: str = "h",
    sort: str = "desc",
    topn: int = None,
    annot: bool = True,
    size: Tuple[float, float] = (5, 5),
    ncols: int = 3,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Figure:
    if ax is not None:
        raise ValueError("`ax` not supported for DataFrame input")
    fig, axs = smart_subplots(
        nplots=data.columns.size,
        ncols=ncols,
        size=size,
    )
    sort = sort.lower()
    data = data.loc[:, data.nunique().sort_values(ascending=False).index]
    set_invisible(axs)
    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        countplot(
            data.loc[:, column],
            normalize=normalize,
            heat=heat,
            heat_desat=heat_desat,
            orient=orient,
            sort=sort,
            topn=topn,
            annot=annot,
            ax=ax,
        )

    fig.tight_layout()
    return fig


@countplot.register
def _(
    data: SeriesGroupBy,
    *,
    normalize: bool = False,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    orient: str = "h",
    sort: str = "desc",
     topn: int = None,
    annot: bool = True,
    size: Tuple[float, float] = (5, 5),
    ncols: int = 3,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Figure:
    if ax is not None:
        raise ValueError("`ax` not supported for SeriesGroupBy input")
    fig, axs = smart_subplots(
        nplots=len(data.groups),
        ncols=ncols,
        size=size,
    )
    sort = sort.lower()
    group_keys = data.nunique().sort_values(ascending=False).index
    set_invisible(axs)
    for ax, key in zip(axs.flat, group_keys):
        ax.set_visible(True)
        countplot(
            data.get_group(key),
            normalize=normalize,
            heat=heat,
            heat_desat=heat_desat,
            orient=orient,
            sort=sort,
            topn=topn,
            annot=annot,
            ax=ax,
        )
        ax.set(title=f"'{key}' Value Counts")

    fig.tight_layout()
    return fig


def heated_barplot(
    *,
    data: pd.Series,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    figsize: tuple = (6, 8),
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a sharply divided barplot ranking positive and negative values.

    Args:
        data (pd.Series): Data to plot.
        heat (str): Name of color palette to be passed to Seaborn.
        heat_desat (float, optional): Saturation of color palette. Defaults to 0.6.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        plt.Axes: Axes for the plot.
    """
    data.index = data.index.astype(str)
    data.sort_values(ascending=False, inplace=True)
    palette = heat_palette(data, heat, desat=heat_desat)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(
        x=data.values, y=data.index, palette=palette, orient="h", ax=ax, **kwargs
    )
    ax.axvline(0.0, color="k", lw=1, ls="-", alpha=0.33)
    return ax


def barplot(
    *,
    data: pd.DataFrame,
    x: str,
    y: str,
    sort="asc",
    orient="v",
    estimator: Callable = np.mean,
    figsize: tuple = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a barplot with sorted bars and switchable orientation.

    Parameters
    ----------
    data : DataFrame
        Data for plotting.
    x : str
        Variable for x-axis.
    y : str
        Variable for y-axis.
    sort : str, optional
        Sort direction, by default "asc".
    orient : str, optional
        Bar orientation: 'h' or 'v' (default).
    estimator : Callable, optional
        Estimator for calculating bar heights, by default np.mean.
    figsize : tuple, optional
        Figure size. Defaults to (8, 5) if not specified.
    ax : Axes, optional
        Axes to plot on, by default None.

    Returns
    -------
    Axes
        Barplot.
    """
    _validate_orient(orient)
    _validate_sort(sort)
    if ax is None:
        if figsize is None:
            width, height = (8, 5)
            figsize = (width, height) if orient == "v" else (height, width)
        fig, ax = plt.subplots(figsize=figsize)
    if sort is not None:
        asc = sort.lower() == "asc"
        order = data.groupby(x)[y].agg(estimator)
        order = order.sort_values(ascending=asc).index.to_list()
    else:
        order = None

    if orient.lower() == "h":
        x, y = y, x
    ax = sns.barplot(
        data=data,
        x=x,
        y=y,
        estimator=estimator,
        orient=orient,
        order=order,
        ax=ax,
        **kwargs,
    )
    return ax
