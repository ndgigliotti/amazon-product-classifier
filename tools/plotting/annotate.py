from functools import singledispatch
from typing import NoReturn, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas.core.series import Series

from tools.outliers import tukey_fences
from tools import utils


def draw_line(
    ax: Axes,
    *,
    x: Union[int, float] = None,
    y: Union[int, float] = None,
    annot: bool = True,
    line_color: str = "k",
    line_style: str = "--",
    pad_title=20,
    num_format: str = ",.0f",
):
    if not (x is None or y is None):
        raise ValueError("Cannot pass both `x` and `y`.")
    if x is not None:
        ax.axvline(x, ls=line_style, c=line_color)
        text_y = ax.get_ylim()[1] * 1.01
        ax.text(x, text_y, f"{x:{num_format}}", ha="center")
    elif y is not None:
        ax.axhline(y, ls=line_style, c=line_color)
        text_x = ax.get_xlim()[1] * 1.01
        ax.text(text_x, y, f"{y:{num_format}}", ha="center")
    else:
        raise ValueError("Must specify either `x` or `y`.")
    if annot and pad_title:
        ax.set_title(ax.get_title(), pad=pad_title)
    return ax


def add_tukey_marks(
    data: Series,
    ax: Axes,
    annot: bool = True,
    iqr_color: str = "r",
    fence_color: str = "k",
    fence_style: str = "--",
    annot_quarts: bool = False,
    num_format: str = ".1f",
) -> Axes:
    """Add IQR box and fences to a histogram-like plot.

    Args:
        data (pd.Series): Data for calculating IQR and fences.
        ax (plt.Axes): Axes to annotate.
        iqr_color (str, optional): Color of shaded IQR box. Defaults to "r".
        fence_color (str, optional): Fence line color. Defaults to "k".
        fence_style (str, optional): Fence line style. Defaults to "--".
        annot_quarts (bool, optional): Annotate Q1 and Q3. Defaults to False.

    Returns:
        Axes: Annotated Axes object.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    ax.axvspan(q1, q3, color=iqr_color, alpha=0.2)
    iqr_mp = q1 + ((q3 - q1) / 2)
    lower, upper = tukey_fences(data)
    ax.axvline(lower, c=fence_color, ls=fence_style)
    ax.axvline(upper, c=fence_color, ls=fence_style)
    text_yval = ax.get_ylim()[1]
    text_yval *= 1.01
    if annot:
        ax.text(iqr_mp, text_yval, "IQR", ha="center")
        if annot_quarts:
            ax.text(q1, text_yval, "Q1", ha="center")
            ax.text(q3, text_yval, "Q3", ha="center")
        ax.text(upper, text_yval, f"{upper:{num_format}}", ha="center")
        ax.text(lower, text_yval, f"{lower:{num_format}}", ha="center")
    return ax


@singledispatch
def annot_bars(
    ax: Union[ndarray, Axes],
    dist: float = 0.15,
    pad: float = 0,
    color: str = "k",
    compact: bool = False,
    orient: str = "h",
    format_spec: str = "{x:.2f}",
    fontsize: int = 12,
    alpha: float = 0.5,
    drop_last: int = 0,
) -> NoReturn:
    """Annotate a bar graph with the bar values.

    Parameters
    ----------
    ax : Axes
        Axes object to annotate.
    dist : float, optional
        Distance from ends as fraction of max bar. Defaults to 0.15.
    pad : float, optional
        Fraction by which to pad axis bounds.
    color : str, optional
        Text color. Defaults to "k".
    compact : bool, optional
        Annotate inside the bars. Defaults to False.
    orient : str, optional
        Bar orientation. Defaults to "h".
    format_spec : str, optional
        Format string for annotations. Defaults to "{x:.2f}".
    fontsize : int, optional
        Font size. Defaults to 12.
    alpha : float, optional
        Opacity of text. Defaults to 0.5.
    drop_last : int, optional
        Number of bars to ignore on tail end. Defaults to 0.
    """
    # This is the last-resort dispatch.
    raise TypeError(f"Expected Axes or ndarray of Axes, got {type(ax)}.")


@annot_bars.register
def _(
    ax: Axes,
    dist: float = 0.15,
    pad: float = 0,
    color: str = "k",
    compact: bool = False,
    orient: str = "h",
    format_spec: str = "{x:.2f}",
    fontsize: int = 12,
    alpha: float = 0.5,
    drop_last: int = 0,
) -> NoReturn:
    """Dispatch for Axes."""
    if not compact:
        dist = -dist

    max_bar = np.abs([b.get_width() for b in ax.patches]).max()
    dist = dist * max_bar
    for bar in ax.patches[: -drop_last or len(ax.patches)]:
        if orient.lower() == "h":
            xb = np.array(ax.get_xbound()) * (1 + pad)
            ax.set_xbound(*xb)
            x = bar.get_width()
            x = x + dist if x < 0 else x - dist
            y = bar.get_y() + bar.get_height() / 2
            text = format_spec.format(x=bar.get_width())
        elif orient.lower() == "v":
            yb = np.array(ax.get_ybound()) * (1 + pad)
            ax.set_ybound(*yb)
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            y = y + dist if y < 0 else y - dist
            text = format_spec.format(x=bar.get_height())
        else:
            raise ValueError("`orient` must be 'h' or 'v'")

        ax.annotate(
            text,
            (x, y),
            ha="center",
            va="center",
            c=color,
            fontsize=fontsize,
            alpha=alpha,
        )


@annot_bars.register
def _(
    ax: ndarray,
    dist: float = 0.15,
    pad: float = 0,
    color: str = "k",
    compact: bool = False,
    orient: str = "h",
    format_spec: str = "{x:.2f}",
    fontsize: int = 12,
    alpha: float = 0.5,
    drop_last: int = 0,
) -> NoReturn:
    """Dispatch for ndarray of Axes."""
    axs = utils.flat_map(
        annot_bars,
        ax,
        dist=dist,
        pad=pad,
        color=color,
        compact=compact,
        orient=orient,
        format_spec=format_spec,
        fontsize=fontsize,
        alpha=alpha,
        drop_last=drop_last,
    )
    return axs
