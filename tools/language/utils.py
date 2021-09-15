from functools import partial, singledispatch
from os.path import normpath
from typing import Any, Callable, Collection, List, Mapping, NoReturn, Union

import joblib
import nltk
import numpy as np
import scipy as sp
import pandas as pd
from IPython.core.display import Markdown, display
from numpy import ndarray
from pandas._typing import AnyArrayLike
from pandas.core.dtypes.missing import isna, notna
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from tools._validation import _check_1d, _validate_strings, _validate_tokens, _check_tokdocs
from tools.typing import CallableOnStr, SeedLike, TaggedTokens, Strings, TokenDocs
from tools.utils import swap_index


@singledispatch
def process_strings(
    strings: Strings, func: CallableOnStr, n_jobs: int = None, **kwargs
) -> Any:
    """Apply `func` to a string or iterable of strings (elementwise).

    Most string filtering/processing functions in the language module
    are polymorphic, capable of handling either a single string or an
    iterable of strings. Whenever possible, they rely on this generic
    function to apply a callable to string(s). This allows them to
    behave polymorphically and take advantage of multiprocessing while
    having a simple implementation.

    This is a single dispatch generic function, meaning that it consists
    of multiple specialized sub-functions which each handle a different
    argument type. When called, the dispatcher checks the type of the first
    positional argument and then dispatches the sub-function registered
    for that type. In other words, when the function is called, the call
    is routed to the appropriate sub-function based on the type of the first
    positional argument. If no sub-function is registered for a given type,
    the correct dispatch is determined by the type's method resolution order.
    The function definition decorated with `@singledispatch` is registered for
    the `object` type, meaning that it's the dispatcher's last resort.

    The most important sub-function is the list dispatch, where
    multiprocessing is implemented via Joblib. Every other sub-function
    (besides str) routes data there for multiprocessing.

    Parameters
    ----------
    strings : str, iterable of str
        String(s) to map `func` over. Null values are ignored.
    func : Callable
        Callable for processing `strings`.
    n_jobs: int
        The maximum number of concurrently running jobs. If -1 all CPUs are used.
        If 1 or None is given, no parallel computing code is used at all. For n_jobs
        below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
        one are used. Defaults to None.
    **kwargs
        Keyword arguments for `func`.

    Returns
    -------
    Any
        Processed string(s), same container type as input.
    """
    # This is the fallback dispatch
    _validate_strings(strings)

    # Send to list dispatch
    return process_strings(list(strings), func, n_jobs=n_jobs, **kwargs)


@process_strings.register
def _(strings: list, func: CallableOnStr, n_jobs: int = None, **kwargs) -> list:
    """Dispatch for list."""
    workers = joblib.Parallel(n_jobs=n_jobs, prefer="processes")
    func = joblib.delayed(partial(func, **kwargs))
    ident = joblib.delayed(lambda x: x)
    return workers(func(x) if notna(x) else ident(x) for x in strings)


@process_strings.register
def _(strings: set, func: CallableOnStr, n_jobs: int = None, **kwargs) -> set:
    """Dispatch for Set."""
    strings = process_strings(list(strings), func, n_jobs=n_jobs, **kwargs)
    return set(strings)


@process_strings.register
def _(strings: ndarray, func: CallableOnStr, n_jobs: int = None, **kwargs) -> ndarray:
    """Dispatch for ndarray. Applies `func` elementwise."""
    orig_shape = strings.shape
    strings = process_strings(strings.flatten().tolist(), func, n_jobs=n_jobs, **kwargs)
    return np.array(strings, dtype="O", copy=False).reshape(orig_shape)


@process_strings.register
def _(strings: Series, func: CallableOnStr, n_jobs: int = None, **kwargs) -> Series:
    """Dispatch for Series."""
    name = strings.name
    index = strings.index
    strings = process_strings(strings.to_list(), func, n_jobs=n_jobs, **kwargs)
    return Series(strings, index=index, name=name)


@process_strings.register
def _(
    strings: DataFrame, func: CallableOnStr, n_jobs: int = None, **kwargs
) -> DataFrame:
    """Dispatch for DataFrame. Applies `func` elementwise."""
    columns = strings.columns
    index = strings.index
    strings = process_strings(strings.to_numpy(), func, n_jobs=n_jobs, **kwargs)
    return DataFrame(strings, index=index, columns=columns, copy=False)


@process_strings.register
def _(strings: str, func: CallableOnStr, n_jobs: int = None, **kwargs) -> Any:
    """Dispatch for single string."""
    return func(strings, **kwargs)


def process_tokenized(docs: Series, func: Callable, n_jobs=None, **kwargs):
    assert pd.api.types.is_list_like(docs.iloc[0])
    index = docs.index
    name = docs.name
    workers = joblib.Parallel(n_jobs=n_jobs, prefer="processes")
    func = joblib.delayed(partial(func, **kwargs))
    docs = workers(func(x) for x in docs)
    return Series(docs, index=index, name=name)


@singledispatch
def process_tokens(
    tokdocs: TokenDocs, func: Callable, n_jobs: int = None, **kwargs
) -> TokenDocs:
    # This is the fallback dispatch
    _check_tokdocs(tokdocs)

    # Send to list dispatch
    return process_tokens(list(tokdocs), func, n_jobs=n_jobs, **kwargs)


@process_tokens.register
def _(tokens: list, func: Callable, n_jobs: int = None, **kwargs) -> list:
    """Dispatch for list."""
    in_struct = _check_tokdocs(tokens)
    if in_struct is Collection[str]:
        tokens = func(tokens, **kwargs)
    else:
        workers = joblib.Parallel(n_jobs=n_jobs, prefer="processes")
        func = joblib.delayed(partial(func, **kwargs))
        tokens = workers(func(x) for x in tokens)
    try:
        out_struct = _check_tokdocs(tokens)
        assert out_struct is in_struct
    except (TypeError, AssertionError):
        raise ValueError("Output of `func` must have same structure as input.")
    return tokens


@process_tokens.register
def _(tokens: ndarray, func: Callable, n_jobs: int = None, **kwargs) -> ndarray:
    """Dispatch for ndarray."""
    _check_1d(tokens)
    tokens = process_tokens(tokens.tolist(), func, n_jobs=n_jobs, **kwargs)
    dtype = str if isinstance(tokens[0], str) else "O"
    return np.array(tokens, dtype=dtype)


@process_tokens.register
def _(tokens: Series, func: Callable, n_jobs: int = None, **kwargs) -> Series:
    """Dispatch for Series."""
    name = tokens.name
    index = tokens.index
    tokens = process_tokens(tokens.to_list(), func, n_jobs=n_jobs, **kwargs)
    dtype = "string" if isinstance(tokens[0], str) else "object"
    return Series(tokens, index=index, name=name, dtype=dtype)


def chain_processors(strings: Strings, funcs: List[Callable], n_jobs=None) -> Any:
    """Apply a pipeline of processing functions to strings.

    Parameters
    ----------
    strings : str or iterable of str
        String(s) to process.
    funcs : list of callable
        Callables to apply elementwise to strings. The first callable must
        take a single str argument.

    Returns
    -------
    Any
        Result of processing. Probably a str, iterable of str, or nested structure.
    """
    # Define pipeline function for singular input.
    # This allows use of `_process` for any chain of function
    # transformations which initially takes a single str argument.
    # The functions can return lists of tuples of str, or whatever,
    # as long as the first function takes a str argument.
    def process_singular(string):
        for func in funcs:
            string = func(string)
        return string

    _validate_strings(strings)

    # Make `process_singular` polymorphic
    return process_strings(strings, process_singular, n_jobs=n_jobs)


def make_preprocessor(funcs: List[Callable]) -> partial:
    """Create a pipeline callable which applies a chain of processors to docs.

    The resulting generic pipeline function will accept one argument
    of type str or iterable of str.

    Parameters
    ----------
    funcs : list of callable
        Callables to apply elementwise to strings. The first callable must
        take a single str argument.

    Returns
    -------
    partial object
        Generic pipeline callable.
    """
    return partial(chain_processors, funcs=funcs)


def to_token_array(token_docs: Series):
    """Experimental. Converts a Series of lists of tokens into an ndarray."""
    lengths = token_docs.str.len().to_numpy()
    max_len = lengths.max()
    pad_widths = max_len - lengths
    token_arr = []
    append = token_arr.append
    pad = np.pad
    for pad_width, tokens in zip(pad_widths, token_docs):
        tokens = pad(tokens, (0, pad_width), mode="empty").astype(str)
        append(tokens)
    return np.vstack(token_arr)


def extract_tags(
    tag_toks: TaggedTokens, as_string: bool = False
) -> Union[List[str], str]:
    _, tags = zip(*tag_toks)
    return " ".join(tags) if as_string else list(tags)


def frame_doc_vecs(
    doc_vecs: csr_matrix,
    vocab: Mapping[str, int],
    doc_index: Union[List, AnyArrayLike] = None,
) -> DataFrame:
    """Convert sparse document vectors into a DataFrame with feature labels.

    Designed for use with Scikit-Learn's CountVectorizer or TfidfVectorizer.

    Parameters
    ----------
    doc_vecs : csr_matrix
        Sparse matrix from Scikit-Learn TfidfVectorizer or similar.
    vocab : mapping (str -> int)
        Mapping from terms to feature indices.
    doc_index : list or array-like, optional
        Index for new DataFrame, defaults to a standard RangeIndex.

    Returns
    -------
    DataFrame
        Document vectors with feature labels.
    """
    vocab = swap_index(Series(vocab)).sort_index()
    if doc_index is None:
        doc_index = pd.RangeIndex(0, doc_vecs.shape[0])
    return DataFrame(doc_vecs.todense(), columns=vocab.to_numpy(), index=doc_index)


def readable_sample(
    data: Series, n: int = 10, random_state: SeedLike = None
) -> NoReturn:
    """Display readable sample of text from `data`.

    Parameters
    ----------
    data : Series of str
        Series of strings to sample.
    n : int, optional
        Sample size, by default 10
    random_state : int, array-like, BitGenerator or RandomState, optional
        Seed for pseudorandom number generator, by default None.
    """
    if isinstance(data, DataFrame):
        raise ValueError(f"Expected Series, got {type(data)}")
    if n > data.size:
        n = data.size
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def tagset_info(tagset="upenn_tagset"):
    path = normpath(f"help/tagsets/{tagset}.pickle")
    info = DataFrame(nltk.data.load(path), index=["tag_definition", "tag_examples"])
    return info.T


def groupby_tag(tag_toks: TaggedTokens):
    tag_toks = DataFrame(tag_toks, columns=["token", "tag"])
    return tag_toks.groupby("tag")
