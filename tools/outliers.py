from functools import singledispatch
import warnings
from tools.typing import FrameOrSeries, Strings
from typing import Iterable, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.preprocessing import scale
from tools._validation import _invalid_value
from tools import utils


def _create_mask(data: FrameOrSeries, value=False, dtype=np.bool_):
    if not isinstance(value, (bool, int)):
        raise TypeError(
            f"Expected bool or int for `value`, got {type(value).__name__}."
        )
    if value:
        mask = np.ones_like(data, dtype=dtype)
    else:
        mask = np.zeros_like(data, dtype=dtype)

    if isinstance(data, ndarray):
        pass
    elif mask.ndim > 1:
        mask = DataFrame(mask, index=data.index, columns=data.columns)
    else:
        mask = Series(mask, index=data.index, name=data.name)
    return mask


def _warn_subset_not_impl(subset, input_type):
    if subset is not None:
        warnings.warn(f"`subset` not implemented for {input_type} input.")


@singledispatch
def _display_report(outliers: DataFrame, verb: str) -> None:
    """Display report of modified observations.

    Parameters
    ----------
    outliers : Series or DataFrame
        Boolean mask which marks outliers as True.
    verb : str
        Outlier adjustment verb (past tense), e.g. 'trimmed'.
    """
    report = outliers.sum()
    n_modified = outliers.any(axis=1).sum()
    report["total_obs"] = n_modified
    report = report.to_frame(f"n_{verb}")
    report[f"pct_{verb}"] = (report.squeeze() / outliers.shape[0]) * 100
    report = report.astype(np.float64)
    print(report.to_string(float_format="{:,.0f}".format))


@_display_report.register
def _(outliers: Series, verb: str) -> None:
    """Process Series"""
    # simply convert Series to DataFrame
    _display_report(outliers.to_frame(), verb)


@singledispatch
def winsorize(data: Series, outliers: Series, show_report: bool = True) -> Series:
    """Reset outliers to outermost inlying values.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    outliers : Series or DataFrame
        Boolean mask of outliers.
    show_report : bool, optional
        Show number of modified observations.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    min_in, max_in = data[~outliers].agg(["min", "max"])
    data = data.clip(lower=min_in, upper=max_in)
    if show_report:
        _display_report(outliers, "winsorized")
    return data


@winsorize.register
def _(data: DataFrame, outliers: DataFrame, show_report: bool = True) -> DataFrame:
    """Process DataFrames"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    bounds = data.mask(outliers).agg(["min", "max"]).T
    data = data.clip(lower=bounds["min"], upper=bounds["max"], axis=1)
    if show_report:
        _display_report(outliers, "winsorized")
    return data


@winsorize.register
def _(data: ndarray, outliers: ndarray, show_report: bool = True) -> ndarray:
    """Process ndarrays"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")

    # convert to DataFrame or Series
    data = DataFrame(data).squeeze()
    outliers = DataFrame(outliers).squeeze()

    # dispatch to relevant function
    data = winsorize(data, outliers, show_report=show_report)
    return data.to_numpy()


@singledispatch
def trim(data: Series, outliers: Series, show_report: bool = True) -> Series:
    """Remove outliers from data.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    outliers : Series or DataFrame
        Boolean mask of outliers.
    show_report : bool, optional
        Show number of trimmed observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    data = data.loc[~outliers].copy()
    if show_report:
        _display_report(outliers, "trimmed")
    return data


@trim.register
def _(data: DataFrame, outliers: DataFrame, show_report: bool = True) -> DataFrame:
    """Process DataFrames"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    data = data.loc[~outliers.any(axis=1)].copy()
    if show_report:
        _display_report(outliers, "trimmed")
    return data


@trim.register
def _(data: ndarray, outliers: ndarray, show_report: bool = True) -> ndarray:
    """Process ndarrays"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    # convert to DataFrame or Series
    data = DataFrame(data).squeeze()
    outliers = DataFrame(outliers).squeeze()

    # dispatch to relevant function and convert back to ndarray
    return trim(data, outliers).to_numpy()


def tukey_fences(data: Series, mult: float = 1.5, interp: str = "linear") -> tuple:
    """Get the lower and upper Tukey fences.

    Tukey's fences are located at 1.5 * IQR from the IQR on either side.
    While 1.5 is standard, a custom multiplier can be specifed using the
    `mult` parameter.

    Parameters
    ----------
    data : Series
        Distribution for calculating fences.
    mult : float
        Multiplier (`mult` * IQR) which determines the margins between
        the IQR and fences. Defaults to 1.5.
    interp : str, optional
        Method to use when quantile lies between two data points,
        by default 'linear'. Possible values: 'linear', 'lower',
        'higher', 'nearest', 'midpoint'. See the Pandas documentation
        for Series.quantile.

    Returns
    -------
    lower : float
        Lower Tukey fence.
    upper : float
        Upper Tukey fence.
    """
    q1 = data.quantile(0.25, interpolation=interp)
    q3 = data.quantile(0.75, interpolation=interp)
    iqr = q3 - q1
    lower = q1 - mult * iqr
    upper = q3 + mult * iqr
    return lower, upper


@singledispatch
def tukey_outliers(data: Series, mult: float = 1.5, subset: Strings = None) -> Series:
    """Returns boolean mask of Tukey-fence outliers.

    Inliers are anything between Tukey's fences (inclusive).
    Tukey's fences are located at 1.5 * IQR from the IQR on either side.
    Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.
    mult : float
        Multiplier (`mult` * IQR) which determines the margins between
        the IQR and fences. Defaults to 1.5.
    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    _warn_subset_not_impl(subset, "Series")
    lower, upper = tukey_fences(data, mult=mult)
    return (data < lower) | (data > upper)


@tukey_outliers.register
def _(data: DataFrame, mult: float = 1.5, subset: Strings = None) -> DataFrame:
    """Process DataFrames"""
    # Slice out subset
    check = utils.get_columns(data, subset)

    # Create base outlier mask
    mask = _create_mask(data)

    # Update mask with checked subset
    mask.update(check.apply(tukey_outliers, mult=mult))
    return mask


@tukey_outliers.register
def _(data: ndarray, mult: float = 1.5, subset: Strings = None) -> ndarray:
    """Process ndarrays"""
    _warn_subset_not_impl(subset, "ndarray")

    # convert to DataFrame or Series
    data = DataFrame(data).squeeze()

    # route to relevant function
    outliers = tukey_outliers(data, mult=mult)

    # convert back to ndarray
    return outliers.to_numpy()


@singledispatch
def z_outliers(
    data: DataFrame,
    thresh: float = 3.0,
    subset: Strings = None,
) -> DataFrame:
    """Returns boolean mask of z-score outliers.

    Inliers are anything with an absolute z-score less than
    or equal to `thresh`. Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.
    thresh : float, optional
        Z-score threshold for outliers, by default 3.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    check = utils.get_columns(data, subset)
    mask = _create_mask(data)
    mask.update(check.apply(scale, raw=True).abs() > thresh)
    return mask.astype(np.bool_)


@z_outliers.register
def _(data: Series, thresh: int = 3, subset: Strings = None) -> Series:
    """Process Series"""
    _warn_subset_not_impl(subset, "Series")
    return z_outliers(data.to_frame(), thresh=thresh).squeeze()


@z_outliers.register
def _(data: ndarray, thresh: int = 3, subset: Strings = None) -> ndarray:
    """Process ndarrays"""
    _warn_subset_not_impl(subset, "ndarray")
    mask = np.abs(scale(data)) > thresh
    return mask.astype(np.bool_)


@singledispatch
def quantile_outliers(
    data: Series,
    inner: float = None,
    lower: float = None,
    upper: float = None,
    subset: Strings = None,
    interp: str = "linear",
) -> Series:
    """Returns boolean mask of observations outside the specified range.

    The `lower` and `upper` quantiles mark the boundaries for inliers
    (inclusive). The parameter `inner` allows you to specify the central
    quantile range of the inliers, and is equivalent to setting symmetrical
    `lower` and `upper` bounds. For DataFrames, outliers are determined
    independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    inner : float, optional
        Quantile range of inliers (i.e. `upper` - `lower`), by default None.
        Shorthand for specifying symmetrical `upper` and `lower` bounds.
        Does nothing if `lower` or `upper` are specified.
    lower : float, optional
        Lower quantile boundary, by default None. Overrides `inner`.
    upper : float, optional
        Upper quantile boundary, by default None. Overrides `inner`.
    interp : str, optional
        Method to use when quantile lies between two data points,
        by default 'linear'. Possible values: 'linear', 'lower',
        'higher', 'nearest', 'midpoint'. See the Pandas documentation
        for Series.quantile.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    _warn_subset_not_impl(subset, "Series")

    if lower or upper:
        lower = 0.0 if lower is None else lower
        upper = 1.0 if upper is None else upper
    elif inner:
        lower = (1 - inner) / 2
        upper = 1 - lower
    else:
        raise ValueError(
            "Must pass either `inner` or (one or both of) `lower` and `upper`"
        )

    lower, upper = data.quantile([lower, upper], interpolation=interp)
    inliers = data.between(lower, upper, inclusive=False) | data.isna()
    return ~inliers


@quantile_outliers.register
def _(
    data: DataFrame,
    inner: float = None,
    lower: float = None,
    upper: float = None,
    subset: Strings = None,
    interp: str = "linear",
) -> DataFrame:
    """Process DataFrames"""
    check = utils.get_columns(data, subset)
    mask = _create_mask(data)
    mask.update(
        check.apply(
            quantile_outliers,
            inner=inner,
            lower=lower,
            upper=upper,
            interp=interp,
        )
    )
    return mask


@quantile_outliers.register
def _(
    data: ndarray,
    inner: float = None,
    lower: float = None,
    upper: float = None,
    interp: str = "linear",
) -> DataFrame:
    """Process ndarrays"""
    # convert to DataFrame or Series
    data = DataFrame(data).squeeze()

    # dispatch to relevant function
    outliers = quantile_outliers(
        data,
        inner=inner,
        lower=lower,
        upper=upper,
        interp=interp,
    )
    return outliers.to_numpy()


def tukey_winsorize(
    data: DataFrame, mult: float = 1.5, subset: Strings = None, show_report: bool = True
) -> DataFrame:
    """Reset outliers to outermost values within Tukey fences.

    For DataFrames, outliers are Winsorized independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    mult : float
        Multiplier (`mult` * IQR) which determines the margins between
        the IQR and fences. Defaults to 1.5.
    show_report : bool, optional
        Show number of modified observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    outliers = tukey_outliers(data, mult=mult, subset=subset)
    return winsorize(data, outliers, show_report)


def tukey_trim(
    data: DataFrame, mult: float = 1.5, subset: Strings = None, show_report: bool = True
) -> DataFrame:
    """Remove observations beyond the Tukey fences.

    For DataFrames, outliers are found independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    mult : float
        Multiplier (`mult` * IQR) which determines the margins between
        the IQR and fences. Defaults to 1.5.
    show_report : bool, optional
        Show number of trimmed observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    outliers = tukey_outliers(data, mult=mult, subset=subset)
    return trim(data, outliers, show_report)


def z_winsorize(
    data: DataFrame, thresh: int = 3, subset: Strings = None, show_report: bool = True
) -> DataFrame:
    """Reset outliers to outermost values within z-score threshold.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    thresh : int, optional
        Z-score threshold for outliers, by default 3.
    show_report : bool, optional
        Show number of modified observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    outliers = z_outliers(data, thresh=thresh, subset=subset)
    return winsorize(data, outliers, show_report)


def z_trim(
    data: DataFrame, thresh: int = 3, subset: Strings = None, show_report: bool = True
) -> DataFrame:
    """Remove observations beyond the z-score threshold.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    thresh : int, optional
        Z-score threshold for outliers, by default 3.
    show_report : bool, optional
        Show number of trimmed observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    outliers = z_outliers(data, thresh=thresh, subset=subset)
    return trim(data, outliers, show_report)


def quantile_winsorize(
    data: DataFrame,
    inner: float = None,
    lower: float = None,
    upper: float = None,
    subset: Strings = None,
    interp: str = "linear",
    show_report: bool = True,
) -> DataFrame:
    """Reset outliers to outermost values within the specified range.

    The `lower` and `upper` quantiles mark the boundaries for inliers
    (inclusive). The parameter `inner` allows you to specify the central
    quantile range of the inliers, and is equivalent to setting symmetrical
    `lower` and `upper` bounds. For DataFrames, outliers are determined
    independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    inner : float, optional
        Quantile range of inliers (i.e. `upper` - `lower`), by default 0.9.
        Shorthand for specifying symmetrical `upper` and `lower` bounds.
        Does nothing if `lower` or `upper` are specified.
    lower : float, optional
        Lower quantile boundary, by default None. Overrides `inner`.
    upper : float, optional
        Upper quantile boundary, by default None. Overrides `inner`.
    interp : str, optional
        Method to use when quantile lies between two data points,
        by default 'linear'. Possible values: 'linear', 'lower',
        'higher', 'nearest', 'midpoint'. See the Pandas documentation
        for Series.quantile.
    show_report : bool, optional
        Show number of modified observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    outliers = quantile_outliers(
        data,
        inner=inner,
        lower=lower,
        upper=upper,
        subset=subset,
        interp=interp,
    )
    return winsorize(data, outliers, show_report)


def quantile_trim(
    data: DataFrame,
    inner: float = None,
    lower: float = None,
    upper: float = None,
    subset: Strings = None,
    interp: str = "linear",
    show_report: bool = True,
) -> DataFrame:
    """Remove observations outside the specified quantile range.

    The `lower` and `upper` quantiles mark the boundaries for inliers
    (inclusive). The parameter `inner` allows you to specify the central
    quantile range of the inliers, and is equivalent to setting symmetrical
    `lower` and `upper` bounds. For DataFrames, outliers are determined
    independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    inner : float, optional
        Quantile range of inliers (i.e. `upper` - `lower`), by default 0.9.
        Shorthand for specifying symmetrical `upper` and `lower` bounds.
        Does nothing if `lower` or `upper` are specified.
    lower : float, optional
        Lower quantile boundary, by default None. Overrides `inner`.
    upper : float, optional
        Upper quantile boundary, by default None. Overrides `inner`.
    interp : str, optional
        Method to use when quantile lies between two data points,
        by default 'linear'. Possible values: 'linear', 'lower',
        'higher', 'nearest', 'midpoint'. See the Pandas documentation
        for Series.quantile.
    show_report : bool, optional
        Show number of trimmed observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    outliers = quantile_outliers(
        data,
        inner=inner,
        lower=lower,
        upper=upper,
        subset=subset,
        interp=interp,
    )
    return trim(data, outliers, show_report)
