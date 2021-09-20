from functools import singledispatch
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import wordcloud as wc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from tools.plotting.utils import smart_subplots


@singledispatch
def wordcloud(
    word_scores: Union[Series, DataFrame],
    *,
    cmap: Union[str, List[str], Dict[str, str]] = "Greys",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    ax: Axes = None,
    **kwargs,
) -> Union[Axes, Figure]:
    """Plot wordcloud(s) from word frequencies or scores.

    Parameters
    ----------
    word_scores : Series or DataFrame
        Word frequencies or scores indexed by word. Plots multiple wordclouds
        if passed a DataFrame.
    cmap : str, or list of str or dict {cols -> cmaps}, optional
        Name of Matplotlib colormap to use, by default 'Greys'.
    size : tuple of floats, optional
        Size of (each) wordcloud, by default (5, 3).
    ncols : int, optional
        Number of columns, if passing a DataFrame. By default 3.
    ax : Axes, optional
        Axes to plot on, if passing a Series. By default None.

    Returns
    -------
    Axes or Figure
        Axes of single wordcloud of Figure of multiple wordclouds.
        Returns Axes if `word_scores` is Series, Figure if a DataFrame.
    """
    # This is the dispatch if `word_scores` is neither Series nor DataFrame.
    raise TypeError(f"Expected Series or DataFrame, got {type(word_scores)}")


@wordcloud.register
def _(
    word_scores: Series,
    *,
    cmap: str = "Greys",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    ax: Axes = None,
    **kwargs,
) -> Axes:
    """Dispatch for Series. Returns single Axes with wordcloud."""
    # Create new Axes if none received
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    # Calculate size of wordcloud image
    width, height = np.array(size) * 100

    cloud = wc.WordCloud(
        colormap=cmap,
        width=width,
        height=height,
        **kwargs,
    )

    # Create wordcloud from scores and put on `ax`
    cloud = cloud.generate_from_frequencies(word_scores)
    ax.imshow(cloud.to_image(), interpolation="bilinear", aspect="equal")

    if word_scores.name is not None:
        ax.set(title=word_scores.name)

    # Hide grid lines and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


@wordcloud.register
def _(
    word_scores: DataFrame,
    *,
    cmap: Union[str, List[str], Dict[str, str]] = "Greys",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    ax: Axes = None,
    **kwargs,
) -> Figure:
    """Dispatch for DataFrames. Plots each column on a subplot."""
    if ax is not None:
        raise ValueError("`ax` not supported for DataFrame input")

    # Create subplots
    fig, axs = smart_subplots(nplots=word_scores.shape[1], size=size, ncols=ncols)

    # Wrangle `cmap` into a dict
    if isinstance(cmap, str):
        cmap = dict.fromkeys(word_scores.columns, cmap)
    elif isinstance(cmap, list):
        cmap = dict(zip(word_scores.columns, cmap))
    elif not isinstance(cmap, dict):
        raise TypeError("`cmap` must be str, list, or dict {cols -> cmaps}")

    # Plot each column
    for ax, column in zip(axs.flat, word_scores.columns):
        wordcloud(
            word_scores.loc[:, column],
            cmap=cmap[column],
            size=size,
            ncols=ncols,
            ax=ax,
            **kwargs,
        )
    fig.tight_layout()
    return fig
