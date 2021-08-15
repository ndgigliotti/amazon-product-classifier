from functools import wraps
from typing import Union
from IPython.display import display, HTML
import pandas as pd
from deprecation import deprecated
import numpy as np


def null_rows(
    data: pd.DataFrame, subset: list = None, total: bool = False
) -> pd.DataFrame:
    """Get rows with missing values.

    Parameters
    ----------
    data : DataFrame
        Data for getting rows with missing values.
    subset: list, optional
        List of columns to consider. None by default.
    total : bool, optional
        Only get rows which are totally null, defaults to False.
    Returns
    -------
    DataFrame
        Table of rows with missing values.
    """

    if subset is None:
        subset = data.columns
    elif isinstance(subset, str):
        subset = [subset]
    if total:
        null_mask = data.loc[:, subset].isnull().all(axis=1)
    else:
        null_mask = data.loc[:, subset].isnull().any(axis=1)
    return data.loc[null_mask].copy()


@wraps(null_rows)
@deprecated(details="use `null_rows` instead")
def nan_rows(data: pd.DataFrame, total: bool = False) -> pd.DataFrame:
    return null_rows(**locals())


def dup_rows(
    data: pd.DataFrame, subset: list = None, keep: Union[str, bool] = False
) -> pd.DataFrame:
    """Get duplicate rows.

    Parameters
    ----------
    data : DataFrame
        Data for getting duplicate rows.
    subset: list, optional
        List of columns to consider. None by default.
    keep: {'first', 'last', False}, default ``False``
        Determines which duplicates (if any) to mark.
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.
    Returns
    -------
    DataFrame
        Table of duplicate rows.
    """
    mask = data.duplicated(subset=subset, keep=keep)
    return data.loc[mask].copy()


def who_is_null(
    data: pd.DataFrame, column: str = None, index: str = None, total: bool = False
) -> np.ndarray:
    """Get indices of rows with missing values.

    Parameters
    ----------
    data : DataFrame
        DataFrame for looking up missing values.
    column : str, optional
        Column for looking up missing values. Uses all columns if not provided.
    index : str, optional
        Column to use as index. Defaults to index of `data` if None.
    total : bool, optional
        Only look up rows which are totally null, defaults to False.
        Irrelevant if `column` is specified.

    Returns
    -------
    ndarray
        Indices of rows with missing values.
    """
    if index is not None:
        data = data.set_index(index)
    if column is None:
        if total:
            null_mask = data.isnull().all(axis=1)
        else:
            null_mask = data.isnull().any(axis=1)
    else:
        null_mask = data[column].isnull()
    return data.loc[null_mask].index.to_numpy()


@wraps(who_is_null)
@deprecated(details="use `who_is_null` instead")
def who_is_nan(
    data: pd.DataFrame, column: str = None, index: str = None, total: bool = False
) -> np.ndarray:
    return who_is_null(**locals())


def class_distrib(data: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """Get class membership counts for the smallest and largest classes.

    Sometimes the class distribution of a categorical variable is imbalanced,
    meaning that some classes have many more members than others. This function
    returns the minimum member count (members of the smallest class), maximum
    member count (members of the largest class), and number of classes to
    indicate the balance.

    Parameters
    ----------
    data : DataFrame
        DataFrame for getting class membership counts.
    normalize : bool, optional
        Return relative membership fractions instead of counts, by default False.

    Returns
    -------
    DataFrame
        Class membership information.
    """
    funcs = ["min", "max", "count"]
    df = data.apply(lambda x: x.value_counts(normalize).agg(funcs))
    df.rename(
        {"count": "classes", "min": "min_members", "max": "max_members"}, inplace=True
    )
    return df.T.sort_values("min_members")


def info(data: pd.DataFrame, round_pct: int = 2) -> pd.DataFrame:
    """Get counts of nulls, uniques, and duplicate observations.

    Parameters
    ----------
    data : DataFrame
        DataFrame for getting information.
    round_pct : int, optional
        Decimals for rounding percentages, by default 2.

    Returns
    -------
    DataFrame
        Table of information.
    """
    n_rows = data.shape[0]
    null = data.isnull().sum().to_frame("null")
    dup = pd.DataFrame(
        index=data.columns, data=data.duplicated().sum(), columns=["dup"]
    )
    uniq = data.nunique().to_frame("uniq")
    info = pd.concat([null, dup, uniq], axis=1)
    pcts = (info / n_rows) * 100
    pcts.columns = pcts.columns.map(lambda x: f"{x}_%")
    pcts = pcts.round(round_pct)
    info = pd.concat([info, pcts], axis=1)
    order = ["null", "null_%", "uniq", "uniq_%", "dup", "dup_%"]
    info = info.loc[:, order]
    info.sort_values("null", ascending=False, inplace=True)
    return info


def show_uniques(data: pd.DataFrame, cut: int = 10, columns: list = None) -> None:
    """Display the unique values for each column of `data`.

    Parameters
    ----------
    data : DataFrame
        DataFrame for viewing unique values.
    cut : int, optional
        Show only columns with this many or fewer uniques, by default 10.
    columns : list, optional
        Columns to show, by default None. Ignores `cut` if specified.
    """
    if columns:
        data = data.loc[:, columns]
    elif cut:
        data = data.loc[:, data.nunique() <= cut]
    cols = [pd.Series(y.dropna().unique(), name=x) for x, y in data.iteritems()]
    table = pd.concat(cols, axis=1)
    table = HTML(table.to_html(index=False, na_rep="", notebook=True))
    display(table)


def impute(data: pd.DataFrame, strategy="mode") -> pd.DataFrame:
    """Impute missing values according to `strategy`.

    Parameters
    ----------
    data : DataFrame
        Data for imputation.
    strategy : str, optional
        Method for calculating fill values, by default "mode".

    Returns
    -------
    DataFrame
        Data with

    Raises
    ------
    ValueError
        Could not fill values in some columns using `strategy`.
    """
    if strategy == "mode":
        filler = data.mode().loc[0]
    elif strategy == "mean":
        filler = data.mean()
    elif strategy == "median":
        filler = data.median()
    data = data.fillna(filler)
    has_na = data.isnull().any(axis=0)
    if has_na.any():
        failed = has_na[has_na].index.to_list()
        raise ValueError(f"Could not fill values in {failed} with {strategy}")
    return data