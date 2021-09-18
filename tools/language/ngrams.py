from functools import partial, singledispatch
from types import MappingProxyType
from typing import Collection, Union

import joblib
import nltk
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from tools.language.utils import chain_processors, process_strings
from tqdm.notebook import tqdm

from .._validation import _validate_strings
from ..typing import CallableOnStr, Documents, Tokenizer
from .processors.tokens import fetch_stopwords, remove_stopwords
from .settings import DEFAULT_TOKENIZER

NGRAM_FINDERS = MappingProxyType(
    {
        2: nltk.BigramCollocationFinder,
        3: nltk.TrigramCollocationFinder,
        4: nltk.QuadgramCollocationFinder,
    }
)
"""Mapping for selecting ngram-finder."""

NGRAM_METRICS = MappingProxyType(
    {
        2: nltk.BigramAssocMeasures,
        3: nltk.TrigramAssocMeasures,
        4: nltk.QuadgramAssocMeasures,
    }
)
"""Mapping for selecting ngram scoring object."""


def stratified_ngrams(
    data: DataFrame,
    *,
    text: str,
    cat: Union[str, Series],
    n: int = 2,
    metric: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Union[str, Collection[str]] = None,
    min_freq: int = 0,
    select_best: float = None,
    fuse_tuples: bool = False,
    sep: str = " ",
    n_jobs=None,
):
    get_ngrams = partial(
        scored_ngrams,
        n=n,
        metric=metric,
        stopwords=stopwords,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    get_ngrams = joblib.delayed(get_ngrams)
    workers = joblib.Parallel(n_jobs=n_jobs, prefer="processes")

    # Get aligned labels and group frames, ignoring empty
    labels, groups = zip(
        *[(lab, grp) for lab, grp in data.groupby(cat) if not grp.empty]
    )
    # Search for ngrams with optional multiprocessing
    cat_ngrams = workers(
        get_ngrams(grp.loc[:, text]) for grp in tqdm(groups, desc="scored_ngrams")
    )

    # Turn each scored ngram Series into a DataFrame
    cat_ngrams = [
        ng.reset_index().assign(**{cat: lab})
        for lab, ng in zip(labels, cat_ngrams)
        if not ng.empty
    ]

    # Select top scores in each category
    if select_best is not None:
        for i, group in enumerate(cat_ngrams):
            cut = group.score.quantile(1 - select_best)
            cat_ngrams[i] = group.loc[group.score >= cut]

    # Stack frames vertically and renumber
    return pd.concat(cat_ngrams).reset_index(drop=True)


@singledispatch
def scored_ngrams(
    docs: Documents,
    n: int = 2,
    metric: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Union[str, Collection[str]] = None,
    min_freq: int = 0,
    fuse_tuples: bool = False,
    sep: str = " ",
) -> Series:
    """Get Series of collocations and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    metric : str, optional
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
    _validate_strings(docs)
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
        measures = NGRAM_METRICS[n]()
    else:
        raise ValueError(f"Valid `n` values are 2, 3, and 4. Got {n}.")
    pre_pipe = []
    if preprocessor is not None:
        # Apply preprocessing
        pre_pipe.append(preprocessor)
    # Tokenize
    pre_pipe.append(tokenizer)
    if stopwords is not None:
        # Fetch stopwords if passed str
        if isinstance(stopwords, str):
            stopwords = fetch_stopwords(stopwords)
        # Remove stopwords
        pre_pipe.append(partial(remove_stopwords, stopwords=stopwords))

    docs = chain_processors(docs, pre_pipe)

    # Find and score collocations
    ngrams = finder.from_documents(docs)
    ngrams.apply_freq_filter(min_freq)
    ngram_score = ngrams.score_ngrams(getattr(measures, metric))

    # Put the results in a DataFrame, squeeze into Series
    kind = {2: "bigram", 3: "trigram", 4: "quadgram"}[n]
    ngram_score = pd.DataFrame(ngram_score, columns=[kind, "score"])
    if fuse_tuples:
        # Join ngram tuples
        ngram_score[kind] = ngram_score[kind].str.join(sep)
    ngram_score.set_index(kind, inplace=True)
    if ngram_score.shape[0] > 1:
        ngram_score = ngram_score.squeeze()
    return ngram_score


@scored_ngrams.register
def _(
    docs: str,
    n: int = 2,
    metric: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = False,
    sep: str = " ",
) -> Series:
    """Dispatch for single str."""
    # Process as singleton
    ngram_score = scored_ngrams(
        [docs],
        n=n,
        metric=metric,
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
    metric: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = False,
    sep: str = " ",
) -> Series:
    """Get Series of bigrams and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    metric : str, optional
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
        metric=metric,
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
    metric: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = False,
    sep: str = " ",
) -> Series:
    """Get Series of trigrams and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    metric : str, optional
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
        metric=metric,
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
    metric: str = "pmi",
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    preprocessor: CallableOnStr = None,
    stopwords: Collection[str] = None,
    min_freq: int = 0,
    fuse_tuples: bool = False,
    sep: str = " ",
) -> Series:
    """Get Series of quadgrams and scores.

    Parameters
    ----------
    docs : str or iterable of str
        Documents to scan for ngrams.
    n : int, optional
        Size of collocations, by default 2.
    metric : str, optional
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
        metric=metric,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        stopwords=stopwords,
        min_freq=min_freq,
        fuse_tuples=fuse_tuples,
        sep=sep,
    )
    return quadgram_score
