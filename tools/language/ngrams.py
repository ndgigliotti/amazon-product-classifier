from functools import partial, singledispatch
from types import MappingProxyType
from typing import Collection, Union

import nltk
import pandas as pd
from numpy import ndarray
from pandas.core.series import Series

from .._validation import _validate_docs
from ..typing import CallableOnStr, Documents, Tokenizer
from .processors.tokens import fetch_stopwords, filter_stopwords
from .settings import DEFAULT_TOKENIZER

NGRAM_FINDERS = MappingProxyType(
    {
        2: nltk.BigramCollocationFinder,
        3: nltk.TrigramCollocationFinder,
        4: nltk.QuadgramCollocationFinder,
    }
)
"""Mapping for selecting ngram-finder."""

NGRAM_MEASURES = MappingProxyType(
    {
        2: nltk.BigramAssocMeasures,
        3: nltk.TrigramAssocMeasures,
        4: nltk.QuadgramAssocMeasures,
    }
)
"""Mapping for selecting ngram scoring object."""


@singledispatch
def scored_ngrams(
    docs: Documents,
    n: int = 2,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Union[str, Collection[str]] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    """Get Series of collocations and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    measure : str, optional
        Scoring metric to use. Valid options include:
        'raw_freq', 'pmi', 'mi_like', 'likelihood_ratio',
        'jaccard', 'poisson_stirling', 'chi_sq', 'student_t'.
        See nltk.BigramAssocMeasures, nltk.TrigramAssocMeasures,
        and nltk.QuadgramAssocMeasures for additional size-specific
        options.
    tokenizer : callable, optional
        Callable for tokenizing docs.
    preprocessor : callable, optional
        Callable for preprocessing docs before tokenization, by default None.
    stopwords : str or collection of str, optional
        Name of known stopwords set or collection of stopwords to remove from docs.
        By default None.
    min_freq : int, optional
        Drop ngrams below this frequency, by default 0.
    fuse_tuples : bool, optional
        Join ngram tuples with `sep`, by default True.
    sep : str, optional
        Separator to use for joining ngram tuples, by default " ".
        Only relevant if `fuze_tuples=True`.

    Returns
    -------
    Series
        Series {ngrams -> scores}.
    """
    _validate_docs(docs)
    # Coerce docs to list
    # if isinstance(docs, (ndarray, Series)):
    #     docs = docs.squeeze().tolist()
    # else:
    #     docs = list(docs)

    # Get collocation finder and measures
    if not isinstance(n, int):
        raise TypeError(f"Expected `n` to be int, got {type(n)}.")
    if 1 < n < 5:
        n = int(n)
        finder = NGRAM_FINDERS[n]
        measures = NGRAM_MEASURES[n]()
    else:
        raise ValueError(f"Valid `n` values are 2, 3, and 4. Got {n}.")

    if preprocessor is not None:
        # Apply preprocessing
        docs = map(preprocessor, docs)
    # Tokenize
    docs = map(tokenizer, docs)
    if stopwords is not None:
        # Fetch stopwords if passed str
        if isinstance(stopwords, str):
            stopwords = fetch_stopwords(stopwords)
        # Remove stopwords
        docs = map(partial(filter_stopwords, stopwords=stopwords), docs)

    # Find and score collocations
    ngrams = finder.from_documents(docs)
    ngrams.apply_freq_filter(min_freq)
    ngram_score = ngrams.score_ngrams(getattr(measures, measure))

    # Put the results in a DataFrame, squeeze into Series
    kind = {2: "bigram", 3: "trigram", 4: "quadgram"}[n]
    ngram_score = pd.DataFrame(ngram_score, columns=[kind, "score"])
    if fuse_tuples:
        # Join ngram tuples
        ngram_score[kind] = ngram_score[kind].str.join(sep)
    return ngram_score.set_index(kind).squeeze()


@scored_ngrams.register
def _(
    docs: str,
    n: int = 2,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    """Dispatch for single str."""
    # Process as singleton
    ngram_score = scored_ngrams(
        [docs],
        n=n,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return ngram_score


def scored_bigrams(
    docs: str,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    """Get Series of bigrams and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    measure : str, optional
        Scoring metric to use. Valid options include:
        'raw_freq', 'pmi', 'mi_like', 'likelihood_ratio',
        'jaccard', 'poisson_stirling', 'chi_sq', 'student_t'.
        See nltk.BigramAssocMeasures for additional size-specific
        options.
    tokenizer : callable, optional
        Callable for tokenizing docs.
    preprocessor : callable, optional
        Callable for preprocessing docs before tokenization, by default None.
    stopwords : collection of str, optional
        Stopwords to remove from docs, by default None.
    min_freq : int, optional
        Drop ngrams below this frequency, by default 0.
    fuse_tuples : bool, optional
        Join ngram tuples with `sep`, by default True.
    sep : str, optional
        Separator to use for joining ngram tuples, by default " ".
        Only relevant if `fuze_tuples=True`.

    Returns
    -------
    Series
        Series {ngrams -> scores}.
    """
    bigram_score = scored_ngrams(
        docs,
        n=2,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return bigram_score


def scored_trigrams(
    docs: str,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    """Get Series of trigrams and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    measure : str, optional
        Scoring metric to use. Valid options include:
        'raw_freq', 'pmi', 'mi_like', 'likelihood_ratio',
        'jaccard', 'poisson_stirling', 'chi_sq', 'student_t'.
        See nltk.TrigramAssocMeasures for additional size-specific
        options.
    tokenizer : callable, optional
        Callable for tokenizing docs.
    preprocessor : callable, optional
        Callable for preprocessing docs before tokenization, by default None.
    stopwords : collection of str, optional
        Stopwords to remove from docs, by default None.
    min_freq : int, optional
        Drop ngrams below this frequency, by default 0.
    fuse_tuples : bool, optional
        Join ngram tuples with `sep`, by default True.
    sep : str, optional
        Separator to use for joining ngram tuples, by default " ".
        Only relevant if `fuze_tuples=True`.

    Returns
    -------
    Series
        Series {ngrams -> scores}.
    """
    trigram_score = scored_ngrams(
        docs,
        n=3,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return trigram_score


def scored_quadgrams(
    docs: str,
    measure: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = True,
    sep: str = " ",
) -> Series:
    """Get Series of quadgrams and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    measure : str, optional
        Scoring metric to use. Valid options include:
        'raw_freq', 'pmi', 'mi_like', 'likelihood_ratio',
        'jaccard', 'poisson_stirling', 'chi_sq', 'student_t'.
        See nltk.QuadgramAssocMeasures for additional size-specific
        options.
    tokenizer : callable, optional
        Callable for tokenizing docs.
    preprocessor : callable, optional
        Callable for preprocessing docs before tokenization, by default None.
    stopwords : collection of str, optional
        Stopwords to remove from docs, by default None.
    min_freq : int, optional
        Drop ngrams below this frequency, by default 0.
    fuse_tuples : bool, optional
        Join ngram tuples with `sep`, by default True.
    sep : str, optional
        Separator to use for joining ngram tuples, by default " ".
        Only relevant if `fuze_tuples=True`.

    Returns
    -------
    Series
        Series {ngrams -> scores}.
    """
    quadgram_score = scored_ngrams(
        docs,
        n=4,
        measure=measure,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return quadgram_score
