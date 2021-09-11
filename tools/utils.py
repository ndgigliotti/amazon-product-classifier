import inspect
import os
from functools import singledispatch
from typing import Callable, Collection, Iterable, List, Union

import numpy as np
import pandas as pd
import requests
from IPython.display import display
from numpy import ndarray
from pandas._typing import ArrayLike, FrameOrSeries
from pandas.api.types import (
    is_categorical_dtype,
    is_float,
    is_hashable,
    is_integer,
    is_list_like,
)
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.series import Series
from tqdm.notebook import tqdm

from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_consistent_length, compute_sample_weight
from tools._validation import _check_1d


def get_columns(data: DataFrame, subset: Union[str, Iterable[str]]):
    if subset is None:
        pass
    elif isinstance(subset, str):
        data = data.loc[:, [subset]]
    elif isinstance(subset, Iterable):
        data = data.loc[:, list(subset)]
    else:
        raise TypeError(
            f"Expected str or iterable of str, got {type(subset).__name__}."
        )
    return data


def numeric_cols(data: pd.DataFrame) -> list:
    """Returns a list of all numeric column names.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        All and only the numeric column names.
    """
    return data.select_dtypes("number").columns.to_list()


def true_numeric_cols(data: pd.DataFrame, min_unique=3) -> list:
    """Returns numeric columns with at least `min_unique` unique values.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        Numeric column names.
    """
    num = data.select_dtypes("number")
    return num.columns[min_unique <= num.nunique()].to_list()


def hashable_cols(data: pd.DataFrame) -> list:
    valid_idx = data.apply(lambda x: x.first_valid_index() or x.index[0])
    test_row = data.loc[valid_idx].fillna(method="bfill").iloc[0]
    hashable = data.columns[test_row.map(is_hashable)]
    return hashable.to_list()


def cat_cols(data: pd.DataFrame, min_cats: int = None, max_cats: int = None) -> list:
    """Returns a list of categorical column names.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.
    min_cats : int, optional
        Minimum number of categories, by default None.
    max_cats : int, optional
        Maximum number of categories, by default None.

    Returns
    -------
    list
        Categorical column names.
    """
    cats = data.select_dtypes("category")
    cat_counts = cats.nunique()
    if min_cats is None:
        min_cats = cat_counts.min()
    if max_cats is None:
        max_cats = cat_counts.max()
    keep = (min_cats <= cat_counts) & (cat_counts <= max_cats)
    return cats.columns[keep].to_list()


def multicat_cols(data: pd.DataFrame) -> list:
    """Returns column names of categoricals with 3+ categories.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        Categorical (3+) column names.
    """
    cats = data.select_dtypes("category")
    return cats.columns[3 <= cats.nunique()].to_list()


def noncat_cols(data: pd.DataFrame) -> list:
    """Returns a list of all non-categorical column names.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        All and only the non-categorical column names.
    """
    return data.columns.drop(cat_cols(data)).to_list()


def binary_cols(data: pd.DataFrame) -> list:
    """Returns a list of columns with exactly 2 unique values.

    Parameters
    ----------
    data : DataFrame
        DataFrame to get column names from.

    Returns
    -------
    list
        All and only the binary column names.
    """
    return data.columns[data.nunique() == 2].to_list()


def get_defaults(func: Callable) -> dict:
    """Returns dict of parameters with their default values, if any.

    Parameters
    ----------
    func : Callable
        Callable to look up parameters for.

    Returns
    -------
    dict
        Parameters with default values, if any.

    Raises
    ------
    TypeError
        `callable` must be Callable.
    """
    if not isinstance(func, Callable):
        raise TypeError(f"`callable` must be Callable, not {type(func)}")
    params = pd.Series(inspect.signature(func).parameters)
    defaults = params.map(lambda x: x.default)
    return defaults.to_dict()


def get_param_names(func: Callable, include_self=False) -> list:
    """Returns list of parameter names.

    Parameters
    ----------
    func : Callable
        Callable to look up parameter names for.

    Returns
    -------
    list
        List of parameter names.
    """
    params = list(inspect.signature(func).parameters.keys())
    if "self" in params:
        params.remove("self")
    return params


def pandas_heatmap(
    frame: pd.DataFrame,
    subset=None,
    na_rep="",
    precision=3,
    cmap="vlag",
    low=0,
    high=0,
    vmin=None,
    vmax=None,
    axis=None,
):
    """Style DataFrame as a heatmap."""
    table = frame.style.background_gradient(
        subset=subset, cmap=cmap, low=low, high=high, vmin=vmin, vmax=vmax, axis=axis
    )
    table.set_na_rep(na_rep)
    table.set_precision(precision)
    return table


def filter_pipe(
    data: FrameOrSeries,
    like: List[str] = None,
    regex: List[str] = None,
    axis: int = None,
) -> FrameOrSeries:
    """Subset the DataFrame or Series labels with more than one filter at once.

    Parameters
    ----------
    data: DataFrame or Series
        DataFrame or Series to filter labels on.
    like : list of str
        Keep labels from axis for which "like in label == True".
    regex : list of str
        Keep labels from axis for which re.search(regex, label) == True.
    axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
        The axis to filter on, expressed either as an index (int)
        or axis name (str). By default this is the info axis,
        'index' for Series, 'columns' for DataFrame.

    Returns
    -------
    Dataframe or Series
        Subset of `data`.
    """
    if like and regex:
        raise ValueError("Cannot pass both `like` and `regex`")
    elif like:
        if isinstance(like, str):
            like = [like]
        for exp in like:
            data = data.filter(like=exp, axis=axis)
    elif regex:
        if isinstance(regex, str):
            regex = [regex]
        for exp in like:
            data = data.filter(regex=exp, axis=axis)
    else:
        raise ValueError("Must pass either `like` or `regex` but not both")
    return data


def title(snake_case: str):
    """Format snake case string as title."""
    return snake_case.replace("_", " ").strip().title()


def title_mode(data: pd.DataFrame):
    """Return copy of `data` with strings formatted as titles."""
    result = data.copy()
    result.update(result.select_dtypes("object").applymap(title))
    for label, column in result.select_dtypes("category").items():
        result[label] = column.cat.rename_categories(title)
    if result.columns.dtype == "object":
        result.columns = result.columns.map(title)
    if result.index.dtype == "object":
        result.index = result.index.map(title)
    return result


def cartesian(*arrays: ArrayLike) -> np.ndarray:
    """Returns the Cartesian product of some 1d arrays.

    Returns
    -------
    ndarray
        Cartesian product.
    """
    arrays = list(arrays)
    for i, array in enumerate(arrays):
        array = np.asarray(array)
        arrays[i] = array
        _check_1d(array)
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def broad_corr(frame: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Get correlations between features of one frame with those of another.

    Parameters
    ----------
    frame : DataFrame
        First DataFrame.
    other : DataFrame
        Second DataFrame.

    Returns
    -------
    DataFrame
        Pearson correlations.
    """
    return other.apply(lambda x: frame.corrwith(x))


def swap_index(data: pd.Series) -> pd.Series:
    """Swap index and values.

    Parameters
    ----------
    data : Series
        Series for swapping index and values.

    Returns
    -------
    Series
        Swapped Series.
    """
    return pd.Series(data.index, index=data.values, name=data.name, copy=True)


def explicit_sort(
    data: FrameOrSeries,
    *,
    order: list,
    mode: str = "values",
    inplace: bool = False,
    **kwargs,
) -> FrameOrSeries:
    """Sort DataFrame or Series values in explicitly specified order.

    Parameters
    ----------
    data : FrameOrSeries
        Data structure to sort.
    order : list
        List specifying sort order.
    mode : str, optional
        Whether to sort 'values' (default) or 'index'.
    inplace : bool, optional
        Perform operation in place; False by default.
    Returns
    -------
    FrameOrSeries
        Sorted data structure or None if `inplace` is set.
    """
    order = list(order)
    mode = mode.lower()

    if mode not in {"values", "index"}:
        raise ValueError("`mode` must be 'values' or 'index'")

    # Define vectorized key function
    get_rank = np.vectorize(lambda x: order.index(x))

    # Sort according to mode
    if mode == "values":
        data = data.sort_values(key=get_rank, inplace=inplace, **kwargs)
    else:
        data = data.sort_index(key=get_rank, inplace=inplace, **kwargs)

    # Return copy or None
    return data


def bitgen(seed: Union[None, int, ArrayLike] = None):
    return np.random.default_rng(seed).bit_generator


@singledispatch
def get_func_name(
    func: Union[
        Callable,
        FunctionTransformer,
        Collection[Callable],
        Collection[FunctionTransformer],
    ]
) -> Union[str, Collection[str]]:
    """Get function name(s) from function-like objects.

    Parameters
    ----------
    func : Callable, FunctionTransformer, collection of
        Function-like object(s) to get names of.

    Returns
    -------
    str or collection of
        Function name(s).
    """
    if hasattr(func, "pyfunc"):
        name = get_func_name(func.pyfunc)
    elif hasattr(func, "func"):
        name = get_func_name(func.func)
    elif isinstance(func, Callable):
        name = func.__name__
    else:
        raise TypeError(
            f"Expected Callable or FunctionTransformer but encountered {type(func)}."
        )
    return name


@get_func_name.register
def _(func: FunctionTransformer) -> str:
    return get_func_name(func.func)


@get_func_name.register
def _(func: Series) -> pd.Series:
    return func.map(get_func_name)


@get_func_name.register
def _(func: ndarray) -> ndarray:
    return flat_map(get_func_name, func)


@get_func_name.register
def _(func: list) -> list:
    return [get_func_name(x) for x in func]


@singledispatch
def implode(
    data: FrameOrSeries, column: Union[str, List[str]] = None, allow_dups=False
) -> FrameOrSeries:
    """Retract "exploded" DataFrame or Series into container of nested lists.

    Parameters
    ----------
    data : DataFrame or Series
        Exploded data structure.

    Returns
    -------
    DataFrame or Series (same as input)
        Frame with values retracted into list-likes.
    """
    raise TypeError(f"Expected DataFrame or Series, got {type(data).__name__}.")


@implode.register
def _(data: Series, column: Union[str, List[str]] = None, allow_dups=False) -> Series:
    """Dispatch for Series."""
    if not allow_dups:
        data = (
            data.reset_index()
            .drop_duplicates()
            .set_index(data.index.name or "index")
            .squeeze()
        )
    return data.groupby(data.index).agg(lambda x: x.to_list())


@implode.register
def _(
    data: DataFrame, columns: Union[str, List[str]] = None, allow_dups=False
) -> DataFrame:
    """Dispatch for DataFrame"""
    if columns is None:
        raise ValueError("Must pass `columns` if input is DataFrame.")
    if isinstance(columns, str):
        columns = [columns]
    imploded = {x: implode(data.loc[:, x], allow_dups=allow_dups) for x in columns}
    data = data.loc[~data.index.duplicated()]
    return data.assign(**imploded)


@singledispatch
def expand(
    data: Union[DataFrame, Series], column: str = None, labels: List[str] = None
) -> DataFrame:
    """Expand a column of length-N list-likes into N columns.

    Parameters
    ----------
    data : Series or DataFrame
        Series or DataFrame with column to expand.
    column : str, optional
        Column of length-N list-likes to expand into N columns, by default None.
        Only relevant for DataFrame input.
    labels : list of str, optional
        Labels for new columns (must provide N labels), by default None

    Returns
    -------
    DataFrame
        Expanded frame.
    """
    # This is the fallback dispatch.
    raise TypeError(f"Expected Series or DataFrame, got {type(data)}.")


@expand.register
def _(data: Series, column: str = None, labels: List[str] = None) -> DataFrame:
    """Dispatch for Series. Expands into DataFrame."""
    if not data.map(is_list_like).all():
        raise ValueError("Elements must all be list-like")
    if not data.map(len).nunique() == 1:
        raise ValueError("List-likes must all be same length")
    col_data = list(zip(*data))
    if labels is not None:
        if len(labels) != len(col_data):
            raise ValueError("Number of `labels` must equal number of new columns")
    else:
        labels = range(len(col_data))
        if data.name is not None:
            labels = [f"{data.name}_{x}" for x in labels]
    col_data = dict(zip(labels, col_data))
    return DataFrame(col_data, index=data.index)


@expand.register
def _(data: DataFrame, column: str = None, labels: List[str] = None) -> DataFrame:
    """Dispatch for DataFrame. Returns DataFrame."""
    if data.columns.value_counts()[column] > 1:
        raise ValueError("`column` must be unique in DataFrame")
    if column is None:
        raise ValueError("Must pass `column` if input is DataFrame")
    expanded = expand(data.loc[:, column], labels=labels)
    insert_at = data.columns.get_loc(column)
    data = data.drop(columns=column)
    for i, label in enumerate(expanded.columns):
        data.insert(
            insert_at + i, label, expanded.loc[:, label], allow_duplicates=False
        )
    return data


def flat_map(func: Callable, arr: np.ndarray, **kwargs):
    # Record shape
    shape = arr.shape

    # Make list
    flat = [func(x, **kwargs) for x in arr.flat]

    # Construct flat array
    arr = np.array(flat, dtype=arr.dtype)

    # Reshape in original shape
    return arr.reshape(shape)


@singledispatch
def prune_categories(
    data: NDFrame,
    column: str = None,
    cut=None,
    qcut=None,
    inclusive=True,
    show_report=True,
):
    raise TypeError(f"`data` must be Series or DataFrame, got {type(data).__name__}.")


@prune_categories.register
def _(
    data: Series,
    column: str = None,
    cut=None,
    qcut=None,
    inclusive=True,
    show_report=True,
):
    if column is not None:
        raise UserWarning("Param `column` is irrelevant for Series input.")
    if cut is not None:
        if isinstance(cut, float):
            assert 0.0 <= cut <= 1.0
            counts = data.value_counts(True)
        elif isinstance(cut, int):
            assert 0 <= cut <= data.size
            counts = data.value_counts()
    elif qcut is not None:
        assert 0.0 <= qcut <= 1.0
        counts = data.value_counts()
        cut = counts.quantile(qcut)
    else:
        raise ValueError("Must provide either `cut` or `qcut`.")

    # Slice out categories to keep
    keep = counts.loc[counts >= cut if inclusive else counts > cut]
    keep = set(keep.index)
    data = data.loc[data.isin(keep)].copy()

    # Remove unused categories if necessary
    if is_categorical_dtype(data):
        data = data.cat.remove_unused_categories()

    if show_report:
        if set(counts.index) == keep:
            print("No categories dropped.")
        else:
            report = counts.to_frame("Support")
            status = pd.Series(data="dropped", index=counts.index, name="Status")
            status[keep] = "retained"
            report = pd.merge(status, report, left_index=True, right_index=True)
            print(report)
    return data


@prune_categories.register
def _(
    data: DataFrame,
    column: str = None,
    cut=None,
    qcut=None,
    inclusive=True,
    show_report=True,
):
    if pd.isnull(column):
        raise ValueError("Must specify `column` for DataFrame input.")
    # Slice out cat variable, reset index to integer range
    cats = data.loc[:, column].reset_index(drop=True)
    # Eliminate small cats using Series dispatch
    cats = prune_categories(
        cats,
        column=None,
        cut=cut,
        qcut=qcut,
        inclusive=inclusive,
        show_report=show_report,
    )
    # Slice out surviving rows by integer location
    data = data.iloc[cats.index].copy()
    # Remove unused categories if necessary
    if is_categorical_dtype(cats):
        data[column] = data.loc[:, column].cat.remove_unused_categories()
    return data


def stratified_sample(
    data: DataFrame,
    by: Union[str, Series],
    n=None,
    frac=None,
    replace=False,
    class_weight="balanced",
    random_state=None,
    axis=None,
):
    if isinstance(by, str):
        by = data.loc[:, by]
    elif isinstance(by, Series):
        check_consistent_length(by, data)
        by, data = by.align(data)
    else:
        raise TypeError(f"Expected `by` to be str or Series, got {type(by).__name__}.")
    weights = compute_sample_weight(class_weight, by)
    return data.sample(
        n=n,
        frac=frac,
        weights=weights,
        replace=replace,
        random_state=random_state,
        axis=axis,
    )


def aligned_sample(*arrays, size, replace=False, weights=None, random_state=None):
    check_consistent_length(*arrays)
    n_rows = arrays[0].shape[0]
    rng = np.random.default_rng(random_state)
    if weights is not None and weights.sum() != 1:
        if weights.sum() != 0:
            weights = weights / weights.sum()
        else:
            raise ValueError("Invalid weights: weights sum to 0.")
    if size < 0:
        raise ValueError("Size must be positive int or float.")
    if is_float(size):
        if size > 1 and not replace:
            raise ValueError("If `size` is fraction > 1, `replace` must be True.")
        size = round(n_rows * size)
    elif is_integer(size) and not size <= n_rows:
        raise ValueError("`size` must be <= array length.")
    row_idx = rng.choice(n_rows, size=size, replace=replace, p=weights)
    return tuple([x.take(row_idx, axis=0) for x in arrays])


def high_corr(data: DataFrame, thresh: float = 0.75) -> Series:
    """Get non-reflexive feature correlations at or above `thresh`.

    Parameters
    ----------
    data : DataFrame
        Data for finding high correlations.
    thresh : float, optional
        High correlation threshold, by default 0.75.

    Returns
    -------
    Series
        High correlations.
    """
    corr_df = pd.get_dummies(data).corr()
    mask = np.tril(np.ones_like(corr_df, dtype=np.bool_))
    corr_df = corr_df.mask(mask).stack()
    high = corr_df >= thresh
    return corr_df[high]


def download(url: str, dst: str, chunk_size: int = 10 ** 6):
    """Download a file to disk (with progress bar).

    Parameters
    ----------
    url : str
        Source URL.
    dst : str
        Destination filepath.
    chunk_size : int, optional
        Number of bytes to download per iteration, by default 10 ** 6.

    Returns
    -------
    str
        Output filepath.
    """
    dst = os.path.normpath(dst)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length"))
        with tqdm(total=total, unit="B", unit_scale=True) as pbar:
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    pbar.update(len(chunk))
                    f.write(chunk)
        return dst
