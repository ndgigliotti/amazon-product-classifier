from functools import singledispatch
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def _format_big_number(num, dec):
    """Format large number using abreviations like 'K' and 'M'."""
    abb = ""
    if num != 0:
        mag = np.log10(np.abs(num))
        if mag >= 12:
            num = num / 10 ** 12
            abb = "T"
        elif mag >= 9:
            num = num / 10 ** 9
            abb = "B"
        elif mag >= 6:
            num = num / 10 ** 6
            abb = "M"
        elif mag >= 3:
            num = num / 10 ** 3
            abb = "K"
        num = round(num, dec)
    return f"{num:,.{dec}f}{abb}"


def big_number_formatter(dec: int = 0) -> ticker.FuncFormatter:
    """Formatter for large numbers; uses abbreviations like 'K' and 'M'.

    Parameters
    ----------
    dec : int, optional
        Decimal precision, by default 0.

    Returns
    -------
    FuncFormatter
        Tick formatter.
    """

    @ticker.FuncFormatter
    def formatter(num, pos):
        return _format_big_number(num, dec)

    return formatter


def big_money_formatter(dec: int = 0) -> ticker.FuncFormatter:
    """Formatter for large monetary numbers; uses abbreviations like 'K' and 'M'.

    Parameters
    ----------
    dec : int, optional
        Decimal precision, by default 0.

    Returns
    -------
    FuncFormatter
        Tick formatter.
    """

    @ticker.FuncFormatter
    def formatter(num, pos):
        return f"${_format_big_number(num, dec)}"

    return formatter


def basic_formatter(spec=",.0f") -> ticker.StrMethodFormatter:
    """Simple string-based tick formatter.

    Parameters
    ----------
    dec : int, optional
        Decimal precision, by default 0.

    Returns
    -------
    StrMethodFormatter
        Tick formatter.
    """
    return ticker.StrMethodFormatter(f"{{x:{spec}}}")


@singledispatch
def rotate_ticks(ax: plt.Axes, deg: float, axis: str = "x"):
    """Rotate ticks on `axis` by `deg`.

    Parameters
    ----------
    ax : Axes or ndarray of Axes
        Axes object or objects to rotate ticks on.
    deg : float
        Degree of rotation.
    axis : str, optional
        Axis on which to rotate ticks, 'x' (default) or 'y'.
    """
    get_labels = getattr(ax, f"get_{axis}ticklabels")
    for label in get_labels():
        label.set_rotation(deg)


@rotate_ticks.register
def _(ax: np.ndarray, deg: float, axis: str = "x"):
    """Process ndarrays"""
    axs = ax
    for ax in axs:
        rotate_ticks(ax, deg=deg, axis=axis)


def map_ticklabels(ax: plt.Axes, mapper: Callable, axis: str = "x") -> None:
    """Apply callable to tick labels.

    Parameters
    ----------
    ax : Axes
        Axes object to apply function on.
    mapper : Callable
        Callable to apply to tick labels.
    axis : str, optional
        Axis on which to apply callable, 'x' (default) or 'y'.
    """
    axis = getattr(ax, f"{axis}axis")
    labels = [x.get_text() for x in axis.get_ticklabels()]
    labels = list(map(mapper, labels))
    axis.set_ticklabels(labels)
