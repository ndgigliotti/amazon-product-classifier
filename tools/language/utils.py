from functools import partial, singledispatch
from os.path import normpath
from typing import Any, Callable, List, Mapping, NoReturn, Union

import nltk
import numpy as np
import pandas as pd
from IPython.core.display import Markdown, display
from numpy import ndarray
from pandas._typing import AnyArrayLike
from pandas.core.dtypes.missing import isna, notna
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from tools._validation import _check_1d, _validate_docs
from tools.typing import CallableOnStr, Documents, SeedLike, TaggedTokenSeq
from tools.utils import swap_index


@singledispatch
def process_docs(docs: Documents, func: CallableOnStr, **kwargs) -> Any:
    """Apply `func` to a string or iterable of strings (elementwise).

    Most string filtering/processing functions in the language module
    are polymorphic, capable of handling either a single string (single
    document), or an iterable of strings (corpus of documents). Whenever
    possible, they rely on this generic function to apply a callable to
    documents(s). This allows them to behave polymorphically while having
    a simple implementation.

    This is a single dispatch generic function, meaning that it consists
    of multiple specialized sub-functions which each handle a different
    argument type. When called, the dispatcher checks the type of the first
    positional argument and then dispatches the sub-function registered
    for that type. In other words, when the function is called, the call
    is routed to the appropriate sub-function based on the type of the first
    positional argument. If no sub-function is registered for a given type,
    the correct dispatch is determined by the type's method resolution order.
    The function definition decorated with `@singledispatch` is registered for
    the `object` type, meaning that it is the dispatcher's last resort.

    Parameters
    ----------
    docs : str, iterable of str
        Document(s) to map `func` over.
    func : Callable
        Callable for processing `docs`.
    **kwargs
        Keyword arguments for `func`.

    Returns
    -------
    Any
        Processed string(s), same container type as input.
    """
    # This is the fallback dispatch

    # Return generator
    return (func(x, **kwargs) if notna(x) else x for x in docs)


@process_docs.register
def _(docs: list, func: CallableOnStr, **kwargs) -> list:
    """Dispatch for list."""
    return [func(x, **kwargs) if notna(x) else x for x in docs]


@process_docs.register
def _(docs: set, func: CallableOnStr, **kwargs) -> set:
    """Dispatch for Set."""
    return {func(x, **kwargs) if notna(x) else x for x in docs}


@process_docs.register
def _(docs: Series, func: CallableOnStr, **kwargs) -> Series:
    """Dispatch for Series."""
    return docs.map(partial(func, **kwargs), na_action="ignore")


@process_docs.register
def _(docs: DataFrame, func: CallableOnStr, **kwargs) -> DataFrame:
    """Dispatch for DataFrame."""
    return docs.applymap(partial(func, **kwargs), na_action="ignore")


@process_docs.register
def _(docs: ndarray, func: CallableOnStr, **kwargs) -> ndarray:
    """Dispatch for 1darray."""
    _check_1d(docs)
    docs = docs.copy()
    null = isna(docs)
    docs[~null] = np.array([func(x, **kwargs) for x in docs[~null]])
    return docs


@process_docs.register
def _(docs: str, func: CallableOnStr, **kwargs) -> Any:
    """Dispatch for single string."""
    return func(docs, **kwargs)


def chain_processors(docs: Documents, funcs: List[Callable]) -> Any:
    """Apply a pipeline of processing functions to docs.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    funcs : list of callable
        Callables to apply elementwise to docs. The first callable must
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
    def process_singular(doc):
        for func in funcs:
            doc = func(doc)
        return doc

    # Make sure we have all our docs in a row
    _validate_docs(docs)

    # Make `process_singular` polymorphic
    return process_docs(docs, process_singular)


def make_preprocessor(funcs: List[Callable]) -> partial:
    """Create a pipeline callable which applies a chain of processors to docs.

    The resulting generic pipeline function will accept one argument
    of type str or iterable of str.

    Parameters
    ----------
    funcs : list of callable
        Callables to apply elementwise to docs. The first callable must
        take a single str argument.

    Returns
    -------
    partial object
        Generic pipeline callable.
    """
    return partial(chain_processors, funcs=funcs)


def extract_tags(
    tag_toks: TaggedTokenSeq, as_string: bool = False
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
