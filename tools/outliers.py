from functools import singledispatch

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import scale


@singledispatch
def _display_report(outliers: pd.DataFrame, verb: str) -> None:
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
    display(report)


@_display_report.register
def _(outliers: pd.Series, verb: str) -> None:
    """Process Series"""
    # simply convert Series to DataFrame
    _display_report(outliers.to_frame(), verb)


@singledispatch
def winsorize(
    data: pd.Series, outliers: pd.Series, show_report: bool = True
) -> pd.Series:
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
def _(
    data: pd.DataFrame, outliers: pd.DataFrame, show_report: bool = True
) -> pd.DataFrame:
    """Process DataFrames"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    bounds = data.mask(outliers).agg(["min", "max"]).T
    data = data.clip(lower=bounds["min"], upper=bounds["max"], axis=1)
    if show_report:
        _display_report(outliers, "winsorized")
    return data


@winsorize.register
def _(data: np.ndarray, outliers: np.ndarray, show_report: bool = True) -> np.ndarray:
    """Process ndarrays"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")

    # convert to DataFrame or Series
    data = pd.DataFrame(data).squeeze()
    outliers = pd.DataFrame(outliers).squeeze()

    # dispatch to relevant function
    data = winsorize(data, outliers, show_report=show_report)
    return data.to_numpy()


@singledispatch
def trim(data: pd.Series, outliers: pd.Series, show_report: bool = True) -> pd.Series:
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
def _(
    data: pd.DataFrame, outliers: pd.DataFrame, show_report: bool = True
) -> pd.DataFrame:
    """Process DataFrames"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    data = data.loc[~outliers.any(axis=1)].copy()
    if show_report:
        _display_report(outliers, "trimmed")
    return data


@trim.register
def _(data: np.ndarray, outliers: np.ndarray, show_report: bool = True) -> np.ndarray:
    """Process ndarrays"""
    if type(data) != type(outliers):
        raise TypeError("`data` and `outliers` must be same type")
    # convert to DataFrame or Series
    data = pd.DataFrame(data).squeeze()
    outliers = pd.DataFrame(outliers).squeeze()

    # dispatch to relevant function and convert back to ndarray
    return trim(data, outliers).to_numpy()


def tukey_fences(data: pd.Series, interpolation: str = "linear") -> tuple:
    """Get the lower and upper Tukey fences.

    Tukey's fences are located at 1.5 * IQR from the IQR on either side.

    Parameters
    ----------
    data : Series
        Distribution for calculating fences.
    interpolation : str, optional
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
    q1 = data.quantile(0.25, interpolation=interpolation)
    q3 = data.quantile(0.75, interpolation=interpolation)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


@singledispatch
def tukey_outliers(data: pd.Series) -> pd.Series:
    """Returns boolean mask of Tukey-fence outliers.

    Inliers are anything between Tukey's fences (inclusive).
    Tukey's fences are located at 1.5 * IQR from the IQR on either side.
    Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    lower, upper = tukey_fences(data)
    return (data < lower) | (data > upper)


@tukey_outliers.register
def _(data: pd.DataFrame) -> pd.DataFrame:
    """Process DataFrames"""
    # simply map Series function across DataFrame
    return data.apply(tukey_outliers)


@tukey_outliers.register
def _(data: np.ndarray) -> np.ndarray:
    """Process ndarrays"""
    # convert to DataFrame or Series
    data = pd.DataFrame(data).squeeze()

    # route to relevant function
    outliers = tukey_outliers(data)

    # convert back to ndarray
    return outliers.to_numpy()


@singledispatch
def z_outliers(data: pd.DataFrame, thresh: int = 3) -> pd.DataFrame:
    """Returns boolean mask of z-score outliers.

    Inliers are anything with an absolute z-score less than
    or equal to `thresh`. Missing values are considered inliers.

    Parameters
    ----------
    data : Series or DataFrame
        Data to examine for outliers.
    thresh : int, optional
        Z-score threshold for outliers, by default 3.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
    z_data = scale(data)
    z_data = pd.DataFrame(z_data, index=data.index, columns=data.columns)
    return z_data.abs() > thresh


@z_outliers.register
def _(data: pd.Series, thresh: int = 3) -> pd.Series:
    """Process Series"""
    # convert to DataFrame and then squeeze back into Series
    return z_outliers(data.to_frame(), thresh=thresh).squeeze()


@z_outliers.register
def _(data: np.ndarray, thresh: int = 3) -> np.ndarray:
    """Process ndarrays"""
    z_data = scale(data)
    return np.abs(z_data) > thresh


@singledispatch
def quantile_outliers(
    data: pd.Series,
    inner: float = 0.9,
    lower: float = None,
    upper: float = None,
    interpolation: str = "linear",
) -> pd.Series:
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
        Quantile range of inliers (i.e. `upper` - `lower`), by default 0.9.
        Shorthand for specifying symmetrical `upper` and `lower` bounds.
        Does nothing if `lower` or `upper` are specified.
    lower : float, optional
        Lower quantile boundary, by default None. Overrides `inner`.
    upper : float, optional
        Upper quantile boundary, by default None. Overrides `inner`.
    interpolation : str, optional
        Method to use when quantile lies between two data points,
        by default 'linear'. Possible values: 'linear', 'lower',
        'higher', 'nearest', 'midpoint'. See the Pandas documentation
        for Series.quantile.

    Returns
    -------
    Series or DataFrame
        Boolean mask of outliers, same type as input.
    """
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

    lower, upper = data.quantile([lower, upper], interpolation=interpolation)
    inliers = data.between(lower, upper, inclusive=True) | data.isna()
    return ~inliers


@quantile_outliers.register
def _(
    data: pd.DataFrame,
    inner: float = 0.9,
    lower: float = None,
    upper: float = None,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Process DataFrames"""
    kwargs = {k: v for k, v in locals().items() if k != "data"}
    # simply map Series function across DataFrame
    return data.apply(quantile_outliers, **kwargs)


@quantile_outliers.register
def _(
    data: np.ndarray,
    inner: float = 0.9,
    lower: float = None,
    upper: float = None,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Process ndarrays"""
    # convert to DataFrame or Series
    data = pd.DataFrame(data).squeeze()
    kwargs = {k: v for k, v in locals().items() if k != "data"}
    # dispatch to relevant function
    outliers = quantile_outliers(data, **kwargs)
    return outliers.to_numpy()


def tukey_winsorize(data: pd.DataFrame, show_report: bool = True) -> pd.DataFrame:
    """Reset outliers to outermost values within Tukey fences.

    For DataFrames, outliers are Winsorized independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to Winsorize.
    show_report : bool, optional
        Show number of modified observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Winsorized data, same type as input.
    """
    outliers = tukey_outliers(data)
    return winsorize(data, outliers, show_report)


def tukey_trim(data: pd.DataFrame, show_report: bool = True) -> pd.DataFrame:
    """Remove observations beyond the Tukey fences.

    For DataFrames, outliers are found independently for each feature.

    Parameters
    ----------
    data : Series or DataFrame
        Data to trim.
    show_report : bool, optional
        Show number of trimmed observations, defaults to True.

    Returns
    -------
    Series or DataFrame
        Trimmed data, same type as input.
    """
    outliers = tukey_outliers(data)
    return trim(data, outliers, show_report)


def z_winsorize(
    data: pd.DataFrame, thresh: int = 3, show_report: bool = True
) -> pd.DataFrame:
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
    outliers = z_outliers(data, thresh=thresh)
    return winsorize(data, outliers, show_report)


def z_trim(
    data: pd.DataFrame, thresh: int = 3, show_report: bool = True
) -> pd.DataFrame:
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
    outliers = z_outliers(data, thresh=thresh)
    return trim(data, outliers, show_report)


def quantile_winsorize(
    data: pd.DataFrame,
    inner: float = 0.9,
    lower: float = None,
    upper: float = None,
    interpolation: str = "linear",
    show_report: bool = True,
) -> pd.DataFrame:
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
    interpolation : str, optional
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
    kwargs = ["inner", "lower", "upper", "interpolation"]
    kwargs = pd.Series(locals()).loc[kwargs]
    outliers = quantile_outliers(data, **kwargs)
    return winsorize(data, outliers, show_report)


def quantile_trim(
    data: pd.DataFrame,
    inner: float = 0.9,
    lower: float = None,
    upper: float = None,
    interpolation: str = "linear",
    show_report: bool = True,
) -> pd.DataFrame:
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
    interpolation : str, optional
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
    kwargs = ["inner", "lower", "upper", "interpolation"]
    kwargs = pd.Series(locals()).loc[kwargs]
    outliers = quantile_outliers(data, **kwargs)
    return trim(data, outliers, show_report)