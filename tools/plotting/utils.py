from os.path import normpath
from types import MappingProxyType
from typing import Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import minmax_scale

from ..typing import SeedLike

# Default style settings for heatmaps
HEATMAP_STYLE = MappingProxyType(
    {
        "square": True,
        "annot": True,
        "fmt": ".2f",
        "cbar": False,
        "center": 0,
        "cmap": sns.color_palette("coolwarm", n_colors=100, desat=0.6),
        "linewidths": 0.1,
        "linecolor": "k",
        "annot_kws": MappingProxyType({"fontsize": 10}),
    }
)

# Matplotlib rcParams to (optionally) set
MPL_DEFAULTS = MappingProxyType({"axes.labelpad": 10, "axes.titlepad": 5})


def heatmap_figsize(shape: Tuple[int, int], scale: float = 0.85) -> Tuple[float, float]:
    """Calculate heatmap figure size based on data shape.

    Args:
        data (pd.DataFrame): Ndarray, Series, or Dataframe for figsize.
        scale (float, optional): Scale multiplier for figsize. Defaults to 0.85.

    Returns:
        Tuple[float, float].
    """
    figsize = np.array(shape, dtype=np.float64)[::-1] * scale
    return tuple(figsize)


def smart_subplots(
    *,
    nplots: int,
    size: Tuple[float, float],
    ncols: int = None,
    nrows: int = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Wrapper for `plt.subplots` which calculates the specifications.

    Parameters
    ----------
    nplots : int
        Number of subplots.
    size : Tuple[float, float]
        Size of each subplot in format (width, height).
    ncols : int, optional
        Number of columns in figure. Derived from `nplots` and `nrows` if not specified.
    nrows : int, optional
        Number of rows in figure. Derived from `nplots` and `ncols` if not specified.
    **kwargs
        Keyword arguments passed to `plt.subplots`.
    Returns
    -------
    fig: Figure
        Figure for the plot.
    axs: Axes or array of Axes
        Axes for the plot.
    """
    if ncols and not nrows:
        nrows = int(np.ceil(nplots / ncols))
    elif nrows and not ncols:
        ncols = int(np.ceil(nplots / nrows))
    elif not (nrows or ncols):
        raise ValueError("Must pass either `ncols` or `nrows`")

    figsize = (ncols * size[0], nrows * size[1])
    kwargs.update(nrows=nrows, ncols=ncols, figsize=figsize)
    fig, axs = plt.subplots(**kwargs)
    return fig, axs


def set_invisible(axs: np.ndarray) -> None:
    """Sets all axes to invisible."""
    for ax in axs.flat:
        ax.set_visible(False)


def flip_axis(ax: plt.Axes, axis: str = "x") -> None:
    """Flip axis so it extends in the opposite direction.

    Parameters
    ----------
    ax : Axes
        Axes object with axis to flip.
    axis : str, optional
        Which axis to flip, by default "x".
    """
    if axis.lower() == "x":
        ax.set_xlim(reversed(ax.get_xlim()))
    elif axis.lower() == "y":
        ax.set_ylim(reversed(ax.get_ylim()))
    else:
        raise ValueError("`axis` must be 'x' or 'y'")


def heat_palette(data: pd.Series, palette_name: str, desat: float = 0.6) -> np.ndarray:
    """Return Series of heat-colors corresponding to values in `data`.

    Parameters
    ----------
    data : Series
        Series of numeric values to associate with heat colors.
    palette_name : str
        Name of Seaborn color palette.
    desat : float, optional
        Saturation of Seaborn color palette, by default 0.6.

    Returns
    -------
    ndarray
        Heat colors aligned with `data`.
    """
    heat = pd.Series(
        sns.color_palette(palette_name, desat=desat, n_colors=201),
        index=pd.RangeIndex(-100, 101),
    )
    idx = np.around(minmax_scale(data, feature_range=(-100, 100))).astype(np.int64)
    return heat.loc[idx].to_numpy()


def cat_palette(
    name: str,
    keys: list,
    shuffle: bool = False,
    offset: int = 0,
    seed: SeedLike = None,
    **kwargs,
) -> dict:
    """Create a color palette dictionary for a categorical variable.

    Args:
        name (str): Color palette name to be passed to Seaborn.
        keys (list): Keys for mapping to colors.
        shuffle (bool, optional): Shuffle the palette. Defaults to False.
        offset (int, optional): Number of initial colors to skip over. Defaults to 0.

    Returns:
        dict: Categorical-style color mapping.
    """
    n_colors = len(keys) + offset
    pal = sns.color_palette(name, n_colors=n_colors, **kwargs)[offset:]
    if shuffle:
        if isinstance(seed, np.random.RandomState):
            seed.shuffle(pal)
        else:
            np.random.default_rng(seed).shuffle(pal)
    return dict(zip(keys, pal))


def save(fig: Figure, dst: str, bbox_inches: Union[str, float] = "tight", **kwargs):
    dst = normpath(dst)
    fig.savefig(dst, bbox_inches=bbox_inches)
    return dst
