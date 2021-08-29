import itertools
import os
import re
from functools import partial, singledispatch
from multiprocessing.pool import Pool
from operator import itemgetter
from typing import Callable, Iterable, List, Union
from matplotlib import ticker

import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import WRatio as weighted_ratio
from fuzzywuzzy.process import extractOne as extract_one
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.api.types import is_numeric_dtype
from tools import outliers, plotting
from tools._validation import _invalid_value
from tools.language.utils import process_docs

from ..typing import Documents, PatternLike


def _findall(id_, pat, docs, flags):
    findings = (
        docs.str.findall(pat, flags=flags)
        .explode()
        .dropna()
        .to_frame("locate_patterns")
        .assign(pattern=id_)
    )
    return findings


def locate_patterns(
    patterns: List[PatternLike],
    docs: Series,
    exclusive: bool = False,
    flags: re.RegexFlag = 0,
    n_jobs: int = None,
) -> Series:
    """Find all occurrences of one or more regex in a string Series.

    Parameters
    ----------
    patterns: List of regex
        Patterns to search for using `re.findall`.
    docs : Series
        Series of str to find and index patterns in.
    exclusive : bool, optional
        Drop indices that match more than one pattern. False by default.
    flags : RegexFlag, optional
        Flags for regular expressions, by default 0.
    n_jobs : int, optional
        Number of processes to open. Defaults to CPU count if None.

    Returns
    -------
    Series
        Series of matches (str).
    """
    # Gather findings for each pattern
    findall = partial(_findall, docs=docs, flags=flags)
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    if n_jobs == 1:
        findings = []
        for id_, pat in enumerate(patterns):
            df = findall(id_, pat)
            findings.append(df)
    else:
        with Pool(processes=n_jobs) as pool:
            findings = pool.starmap(findall, enumerate(patterns))

    # Merge all findings
    findings = pd.concat(findings, axis=0)

    if exclusive:
        # Drop rows with findings from more than one pattern
        groups = findings.groupby("pattern").groups
        for key, indices in groups.items():
            groups[key] = set(indices)
        discard = set()
        for p1, p2 in itertools.combinations(groups.keys(), 2):
            discard.update(groups[p1] & groups[p2])
        findings.drop(discard, inplace=True)

    # Sort and return
    return findings.drop("pattern", axis=1).squeeze().sort_index()


@singledispatch
def fuzzy_match(
    strings: Iterable[str],
    choices: Iterable[str],
    scorer: Callable[[str, str], int] = weighted_ratio,
    **kwargs,
) -> DataFrame:
    """Fuzzy match each element of `strings` with one of `choices`.

    Parameters
    ----------
    strings : iterable of str
        Strings to find matches for.
    choices : iterable of str
        Strings to choose from.
    scorer : callable ((string, choice) -> int), optional
        Scoring function, by default weighted_ratio.

    Returns
    -------
    DataFrame
        Table of matches and scores.
    """
    # This is the fallback dispatch
    # Try to coerce iterable into Series
    if isinstance(strings, ndarray):
        strings = Series(strings)
    else:
        strings = Series(list(strings))
    return fuzzy_match(strings, choices, scorer=scorer, **kwargs)


@fuzzy_match.register
def _(
    strings: Series,
    choices: Iterable[str],
    scorer: Callable[[str, str], int] = weighted_ratio,
    **kwargs,
) -> DataFrame:
    """Dispatch for Series (retains index)."""
    select_option = partial(
        extract_one,
        choices=choices,
        scorer=scorer,
        **kwargs,
    )
    scores = strings.map(select_option, "ignore")
    strings = strings.to_frame("original")
    strings["match"] = scores.map(itemgetter(0), "ignore")
    strings["score"] = scores.map(itemgetter(1), "ignore")
    return strings


def length_outliers(
    docs: Union[Series, DataFrame],
    method: str = "quantile",
    q_inner: float = None,
    q_lower: float = None,
    q_upper: float = None,
    q_interp: str = "linear",
    iqr_mult: float = 1.5,
    z_thresh: float = 3.0,
) -> Series:
    data = process_docs(docs, len)
    if method == "quantile":
        mask = outliers.quantile_outliers(
            data,
            inner=q_inner,
            lower=q_lower,
            upper=q_upper,
            interp=q_interp,
        )
    elif method == "iqr":
        mask = outliers.tukey_outliers(data, mult=iqr_mult)
    elif method == "z-score":
        mask = outliers.z_outliers(data, thresh=z_thresh)
    else:
        _invalid_value("method", method, ("quantile", "iqr", "z-score"))
    length_info(outliers.trim(data, mask, False), compute_len=False,)
    return mask

def length_info(docs: Union[Series, DataFrame], compute_len=True):
    if not isinstance(docs, (Series, DataFrame)):
        raise TypeError(f"Expected Series or DataFrame, got {type(docs).__name__}.")
    if isinstance(docs, Series):
        docs = docs.to_frame()
    if compute_len:
        data = process_docs(docs.select_dtypes("object"), len)
    else:
        data = docs
    report = data.agg(["min", "median", "max"]).add_prefix("len_")
    print(report.to_string(float_format="{:,.0f}".format))

def trim_length_outliers(
    docs: Union[Series, DataFrame],
    method: str = "quantile",
    q_inner: float = None,
    q_lower: float = None,
    q_upper: float = None,
    q_interp: str = "linear",
    iqr_mult: float = 1.5,
    z_thresh: float = 3.0,
    subset=None,
    show_report=True,
):
    get_outliers = partial(
        length_outliers,
        method=method,
        q_inner=q_inner,
        q_lower=q_lower,
        q_upper=q_upper,
        q_interp=q_interp,
        iqr_mult=iqr_mult,
        z_thresh=z_thresh,
    )
    if subset is not None:
        if not isinstance(docs, DataFrame):
            raise TypeError("`docs` must be DataFrame if `subset` is specified.")
        mask = DataFrame(
            np.zeros_like(docs, dtype=np.bool_),
            index=docs.index,
            columns=docs.columns,
        )
        mask.loc[:, subset] = get_outliers(docs.loc[:, subset])
    else:
        mask = get_outliers(docs)
    if show_report:
        print("\n")
    return outliers.trim(docs, mask, show_report)


def length_dist(data: DataFrame, subset=None, tick_prec=0, **kwargs):
    if isinstance(data, Series):
        data = data.to_frame(data.name or "Unnamed")
    subset = subset or data.columns
    if isinstance(subset, str):
        subset = [subset]
    n_chars = data.loc[:, subset]
    n_chars = n_chars.applymap(len, "ignore")
    fig = plotting.multi_dist(data=n_chars, **kwargs)
    axs = fig.get_axes()
    for col, ax in zip(subset, axs):
        ax.set(
            xlabel="Character Count",
            ylabel="Document Count",
            title=f"Length of '{col}'",
        )
        ax.xaxis.set_major_formatter(plotting.big_number_formatter(tick_prec))
        ax.yaxis.set_major_formatter(plotting.big_number_formatter(tick_prec))
    fig.tight_layout()
    return fig
