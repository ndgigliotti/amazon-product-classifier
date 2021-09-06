import glob
import os
import re
import tempfile
from multiprocessing.pool import Pool
from operator import itemgetter
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
from IPython.core.display import HTML
from IPython.display import display
from numpy import ndarray
from numpy.random import RandomState
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.io import json
from sklearn.base import BaseEstimator, is_classifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import compute_sample_weight, deprecated
from tools import utils
from tools._validation import _check_overwrite

SEARCH_ESTS = (
    GridSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
    HalvingRandomSearchCV,
)
ParamSpaceSpec = Union[Dict[str, List], List[Dict[str, List]]]
SearchEstimator = Union[
    GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
]
JOBLIB_EXT = re.compile(r"(\.joblib.*)$", flags=re.I)


def _to_joblib(obj: object, dst: str, compress=False, test: bool = False) -> str:
    """Serialize an object via Joblib.

    Parameters
    ----------
    obj : object
        Object to serialize and save to disk.
    dst : str
        Path of the file in which it is to be stored. The compression method
        corresponding to one of the supported filename extensions
        ('.z', '.gz', '.bz2', '.xz' or '.lzma') will be used automatically.
    test : bool, optional
        Saves object to a temporary file to see if any errors are raised
        during serialization. The file is immediately removed. By default False.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and write times.
        Using a value of 3 is often a good compromise. See the notes for more details.
        If compress is True, the compression level used is 3. If compress is a 2-tuple,
        the first element must correspond to a string between supported compressors
        (e.g 'zlib', 'gzip', 'bz2', 'lzma' 'xz'), the second element must be an integer
        from 0 to 9, corresponding to the compression level. Defaults to False.

    Returns
    -------
    str
        Filepath of object.
    """

    if test:
        # Pickle object to tempfile
        with tempfile.TemporaryFile() as f:
            # Deleted when closed
            joblib.dump(obj, f, compress=compress)
            dst = "success"
    else:
        # Pickle object to `dst`
        dst = os.path.normpath(dst)

        if os.path.dirname(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.basename(dst):
            raise ValueError(f"Invalid file path: {dst}")

        joblib.dump(obj, dst, compress=compress)
    return dst


def _prep_param_space(
    param_space: ParamSpaceSpec, add_prefix: str = None
) -> ParamSpaceSpec:
    """Prepare a parameter space specification for the search estimator.

    Parameters
    ----------
    param_space : dict, list of dict
        Parameter space specification. Similar types will be coerced to dict or list of dict.
    add_prefix : str, optional
        Prefix to add to all parameter names, by default None.

    Returns
    -------
    dict or list of dict
        Preprocessed parameter space.
    """
    if add_prefix is None:
        add_prefix = ""
    # Coerce Sequence to list of dict and add prefix
    if isinstance(param_space, Sequence):
        param_space = list(param_space)
        for i, sub_space in enumerate(param_space):
            param_space[i] = {f"{add_prefix}{k}": v for k, v in sub_space.items()}
    # Coerce Series to dict and add prefix
    elif isinstance(param_space, pd.Series):
        param_space = param_space.add_prefix(add_prefix).to_dict()
    # Coerce Mapping to dict and add prefix
    elif isinstance(param_space, Mapping):
        param_space = dict(param_space)
        param_space = {f"{add_prefix}{k}": v for k, v in param_space.items()}
    # Error if unexpected type
    else:
        raise TypeError(
            f"Expected mapping, Series, or list thereof, got {type(param_space)}."
        )
    return param_space


def sweep(
    estimator: Union[BaseEstimator, Pipeline],
    param_space: ParamSpaceSpec,
    *,
    X: Union[DataFrame, Series, ndarray],
    y: Union[Series, ndarray],
    dst: str = None,
    cv_dst: str = None,
    compress: Union[int, bool, Tuple[str, int]] = False,
    scoring: Union[str, Callable, List, Tuple, Dict] = None,
    n_jobs: int = None,
    n_iter: int = 10,
    n_samples: Union[int, float] = None,
    refit: bool = False,
    cv: int = None,
    kind: str = "grid",
    add_prefix: str = None,
    verbose: int = 1,
    pre_dispatch: str = "2*n_jobs",
    error_score: float = np.nan,
    return_train_score: bool = False,
    random_state: Union[int, RandomState] = None,
    factor: int = 3,
    resource: str = "n_samples",
    max_resources: Union[int, str] = "auto",
    min_resources: Union[int, str] = "exhaust",
    n_candidates: Union[int, str] = "exhaust",
    aggressive_elimination: bool = False,
    **kwargs,
) -> str:
    """Fit and serialize any Scikit-Learn search estimator.

    Fit and save a `GridSearchCV`, `HalvingGridSearchCV`, `RandomizedSearchCV`,
    or `HalvingRandomSearchCV` object. Immediately saving the search estimator
    via Joblib helps prevent losing the results. See the Scikit-Learn documentation
    on the aforementioned search estimators for more details on their parameters.

    Parameters
    ----------
    estimator : estimator or Pipeline
        Estimator or pipeline ending with estimator.
    param_space : dict, list of dict
        Specification of the parameter search space.
    X : Union[DataFrame, Series, ndarray]
        Independent variables.
    y : Union[Series, ndarray]
        Target variable.
    dst : str
        Output filepath for pickled search estimator. The compression method
        corresponding to one of the supported filename extensions ('.z', '.gz',
        '.bz2', '.xz' or '.lzma') will be used automatically. Defaults to None.
    cv_dst: str, optional
        Output filepath for pickled `cv_results_`. Can be set to 'auto' if `dst`
        is provided, in which case the filepath will be derived from `dst`. Defaults
        to None.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for pickles. 0 or False is no compression.
        Higher value means more compression, but also slower read and write times.
        Using a value of 3 is often a good compromise. See the notes for more details.
        If compress is True, the compression level used is 3. If compress is a 2-tuple,
        the first element must correspond to a string between supported compressors
        (e.g 'zlib', 'gzip', 'bz2', 'lzma' 'xz'), the second element must be an integer
        from 0 to 9, corresponding to the compression level. Defaults to False.
    scoring : Union[str, Callable, List, Tuple, Dict], optional
        Metric name(s) or callable(s) to be passed to search estimator.
    n_jobs : int, optional
        Number of tasks to run in parallel. Defaults to 1 if not specified.
        Pass -1 to use all available CPU cores.
    n_iter : int, optional
        Number of iterations for randomized search, by default 10.
        Irrelevant for non-randomized searches.
    n_samples : int or float, optional
        If an int, the number of samples to use in search. If a float,
        the fraction of total samples to use. If None, use full data.
        Sampling is performed randomly without replacement before search.
        Defaults to None.
    refit : bool, optional
        Whether to refit the estimator with the best parameters from the
        search. False by default.
    cv : int, optional
        Number of cross validation folds, or cross validator object.
        Defaults to 5 if not specified.
    kind : str, optional
        String specifying search type:
            * 'grid' - GridSearchCV
            * 'hgrid' - HalvingGridSearchCV
            * 'rand' - RandomizedSearchCV
            * 'hrand' - HalvingRandomSearchCV
    verbose : int, optional
        Print out details about the search, by default 1.
    add_prefix : str, optional
        Prefix to add to all parameter names.
    pre_dispatch : str, optional
        Controls the number of jobs that get dispatched during parallel
        execution, by default "2*n_jobs".
    error_score : float, optional
        Score if an error occurs in estimator fitting, by default np.nan.
    return_train_score : bool, optional
        Whether to include training scores in `cv_results_`, by default False.
    random_state : int or RandomState, optional
        Seed for random number generator, or RandomState, by default None.
    factor : int, optional
        Proportion of candidates that are selected for each subsequent iteration,
        by default 3. Only relevant for halving searches.
    resource : str, optional
        Defines the resource that increases with each iteration, 'n_samples' by default.
        Only relevant for halving searches.
    max_resources : Union[int, str], optional
        The maximum amount of resource that any candidate is allowed to use
        for a given iteration, by default 'auto'. Only relevant for halving searches.
    min_resources : Union[int, str], optional
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Can be integer, 'smallest' or 'exhaust' (default).
        Only relevant for halving searches.
    n_candidates : Union[int, str], optional
        The number of candidate parameters to sample, at the first
        iteration. Using 'exhaust' will sample enough candidates so that the
        last iteration uses as many resources as possible, based on
        `min_resources`, `max_resources` and `factor`. In this case,
        `min_resources` cannot be 'exhaust'. Only relevant for 'hrand'.
    aggressive_elimination : bool, optional
        Replay the first iteration to weed out candidates until enough are eliminated
        such that only `factor` candidates are evaluated in the final iteration.
        False by default. Only relevant for halving searches.

    Returns
    -------
    str
        Filename of pickled search estimator.

    """
    if dst is not None:
        if not JOBLIB_EXT.search(dst):
            dst = f"{dst}.joblib"
        dst = os.path.normpath(dst)
        _check_overwrite(dst)
    if cv_dst is not None:
        if cv_dst == "auto" and dst is not None:
            cv_dst = JOBLIB_EXT.sub("_cv.joblib", dst)
        elif not JOBLIB_EXT.search(cv_dst):
            cv_dst = f"{cv_dst}.joblib"
        cv_dst = os.path.normpath(cv_dst)
        _check_overwrite(cv_dst)

    # Select search class
    kinds = dict(
        grid=GridSearchCV,
        hgrid=HalvingGridSearchCV,
        rand=RandomizedSearchCV,
        hrand=HalvingRandomSearchCV,
    )
    try:
        cls = kinds[kind.lower()]
    except KeyError:
        raise ValueError("Valid kinds are 'grid', 'hgrid', 'rand', and 'hrand'.")

    # Coerce to dict or list of dict and add prefix
    param_space = _prep_param_space(param_space, add_prefix=add_prefix)

    # Filter out the relevant parameters
    relevant = pd.Series(locals()).drop("kwargs")
    relevant["param_grid"] = param_space
    relevant["param_distributions"] = param_space
    relevant = relevant.loc[utils.get_param_names(cls.__init__)]
    relevant.update(kwargs)

    search = cls(**relevant)

    # Test pickling before fitting
    if dst is not None:
        _to_joblib(search, dst, compress=compress, test=True)
    if cv_dst is not None:
        _to_joblib(param_space, cv_dst, compress=compress, test=True)

    # Sample the data
    if n_samples is not None:
        if is_classifier(estimator):
            weights = compute_sample_weight("balanced", y)
            weights /= weights.sum()
        else:
            weights = None
        X, y = utils.aligned_sample(
            X,
            y,
            size=n_samples,
            weights=weights,
            random_state=random_state,
        )

    search.fit(X, y)

    # Pickle search estimator and CV results
    out = []
    if dst is not None:
        _to_joblib(search, dst, compress=compress)
        out.append(dst)
    if cv_dst is not None:
        cv_dst = _to_joblib(search.cv_results_, cv_dst, compress=compress)
        out.append(cv_dst)
    if out:
        display(out)

    return search


def prune_cv(
    cv_results,
    *,
    drop_splits: bool = True,
    short_names: bool = True,
    drop_dicts: bool = False,
    stats: Sequence[str] = ("mean_fit_time", "mean_test_score", "rank_test_score"),
) -> DataFrame:
    # Construct DataFrame
    df = pd.DataFrame(cv_results)

    # Identify param columns and stat columns
    par_cols = df.columns[df.columns.str.startswith("param_")].to_list()
    par_cols.sort()
    if not drop_dicts:
        par_cols += ["params"]
    stat_cols = df.columns[~df.columns.isin(par_cols)].to_list()

    # Put param columns on the left and stat columns on the right
    df = utils.explicit_sort(df, order=(par_cols + stat_cols), mode="index", axis=1)

    # Drop stats not specified in `stats` argument
    if stats is not None:
        df.drop(set(stat_cols) - set(stats), axis=1, inplace=True)

    # Prune columns of individual splits
    if drop_splits:
        splits = df.filter(regex=r"split[0-9]+_").columns
        df.drop(columns=splits, inplace=True)

    # Sort by rank score and fit time
    if "rank_test_score" in df.columns and "mean_fit_time" in df.columns:
        df.sort_values(["rank_test_score", "mean_fit_time"], inplace=True)
    elif "rank_test_score" in df.columns:
        df.sort_values("rank_test_score")
    # Reset index after sorting
    df.index = pd.RangeIndex(0, df.shape[0], name=df.index.name)

    # Cut out pipeline prefixes and the word 'test'
    if short_names:
        df.columns = df.columns.str.split("__").map(itemgetter(-1))
        df.columns = df.columns.str.replace("test_score", "score", regex=False)
        if df.index.name is not None:
            df.index.name = df.index.name.replace("test_score", "score")
    df = df.applymap(_func_xformers_to_str)
    return df


def load_results(
    path: str,
    *,
    drop_splits: bool = True,
    short_names: bool = True,
    drop_dicts: bool = False,
    stats: Sequence[str] = ("mean_fit_time", "mean_test_score", "rank_test_score"),
) -> DataFrame:
    """Load stripped-down version of search results from pickle.

    Retrieves the `cv_results_` from a serialized, fitted, search estimator
    and optionally trims it down for quick readability.

    Parameters
    ----------
    path : str
        Filename of serialized search estimator.
    drop_splits : bool, optional
        Drop the columns of individual cross validation splits. By default True.
    short_names : bool, optional
        Strip pipeline prefixes and extra words like 'test' from column labels.
        By default True.
    drop_dicts : bool, optional
        Drop parameter dictionaries to make the DataFrame prettier. By default True.
    stats : Sequence[str], optional
        Stats to include in the report, by default 'mean_test_score'
        and 'rank_test_score'. Pass `None` to for all the available stats.

    Returns
    -------
    DataFrame
        Table of cross validation results.
    """
    cv_results = load(path).cv_results_
    return prune_cv(
        cv_results,
        drop_splits=drop_splits,
        short_names=short_names,
        drop_dicts=drop_dicts,
        stats=stats,
    )


def _func_xformers_to_str(x):
    """Convert FunctionTransformer to pretty string, otherwise do nothing."""
    if isinstance(x, FunctionTransformer):
        name = x.func.__name__
        if x.kw_args is not None:
            kwargs = ", ".join([f"{k}={repr(v)}" for k, v in x.kw_args.items()])
        else:
            kwargs = ""
        x = f"{name}({kwargs})"
    return x


@deprecated("Use `joblib.load` instead.")
def load_best_params(path: str) -> dict:
    """Return best parameters from serialized search estimator.

    Parameters
    ----------
    path : str
        Path to serialized search estimator.
        Adds '.joblib' extension if not given.

    Returns
    -------
    dict
        Dict of best parameters.
    """
    if ".joblib" not in path:
        path = f"{path}.joblib"
    search = joblib.load(os.path.normpath(path))
    return search.best_params_


@deprecated("Use `joblib.load` instead.")
def load(path: str) -> SearchEstimator:
    """Load serialized search estimator.

    Parameters
    ----------
    path : str
        Path to serialized search estimator.
        Adds '.joblib' extension if not given.

    Returns
    -------
    dict
        Dict of best parameters.
    """
    return joblib.load(os.path.normpath(path))


def _to_json(df, dst):
    df.to_json(dst)


def batch_export_cv(dir_path, prune=False, n_jobs=None):
    dir_path = os.path.normpath(dir_path) + "/*.joblib"
    with Pool(n_jobs) as pool:
        paths = [x for x in glob.glob(dir_path)]
        ests = pool.map(joblib.load, paths)
        ests = [(e, p) for e, p in zip(ests, paths) if hasattr(e, "cv_results_")]
        ests, paths = zip(*ests)
        if prune:
            dfs = [prune_cv(e.cv_results_) for e in ests]
        else:
            dfs = [pd.DataFrame(e.cv_results_) for e in ests]
        paths = [JOBLIB_EXT.sub(".json", x) for x in paths]
        pool.starmap(_to_json, zip(dfs, paths))
        return paths


def space_size(param_space: ParamSpaceSpec, n_folds=5) -> Series:
    """Return number of parameters, number of combos, number of fits.

    Parameters
    ----------
    param_space : dict or list of dict
        Parameter space specification.
    n_folds : int, optional
        Number of cross-validation folds for for calculating
        number of fits by exhaustive search. By default 5.

    Returns
    -------
    Series
        Description of parameter space size.
    """
    if isinstance(param_space, Sequence):
        n_combos = 0
        params = set()
        for sub_space in param_space:
            params = params.union(sub_space.keys())
            lists = [np.arange(len(x)) for x in sub_space.values()]
            combos = utils.cartesian(*lists)
            n_combos += combos.shape[0]
        n_params = len(params)
    elif isinstance(param_space, pd.Series):
        lists = [np.arange(len(x)) for x in param_space.to_list()]
        combos = utils.cartesian(*lists)
        n_combos, n_params = combos.shape
    elif isinstance(param_space, Mapping):
        lists = [np.arange(len(x)) for x in param_space.values()]
        combos = utils.cartesian(*lists)
        n_combos, n_params = combos.shape
    else:
        raise TypeError(f"Expected dict or list of dicts, got {type(param_space)}.")
    return pd.Series(
        {
            "n_params": n_params,
            "n_combos": n_combos,
            "n_folds": n_folds,
            "n_fits": n_combos * n_folds,
        }
    )


def get_starter_grid(cls):
    return {k: [v] for k, v in utils.get_defaults(cls).items()}
