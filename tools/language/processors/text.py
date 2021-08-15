import re
import string
import html
from collections import defaultdict
from functools import lru_cache, partial, singledispatch
from types import MappingProxyType
from typing import Any, Collection, Union
from deprecation import deprecated
import gensim.parsing.preprocessing as gensim_pp
import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk.sentiment.util import mark_negation as nltk_mark_neg
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import casual as nltk_casual
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from numpy import ndarray
from pandas.core.series import Series
from sacremoses.tokenize import MosesTokenizer
from sklearn.feature_extraction import text as skl_text

from ..._validation import _check_1d, _validate_docs
from ...typing import CallableOnStr, Documents, TaggedTokenSeq, Tokenizer, TokenSeq
from ..settings import CACHE_SIZE, DEFAULT_SEP, DEFAULT_TOKENIZER
from .tokens import moses_detokenize, wordnet_lemmatize

SENT_DELIM = frozenset(".!?")


tb_tokenizer = TreebankWordTokenizer()
"""Treebank tokenizer. Useful for tokenize -> process -> detokenize."""


@singledispatch
def _process(docs: Documents, func: CallableOnStr, **kwargs) -> Any:
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

    # Return iterable
    return map(partial(func, **kwargs), docs)


@_process.register
def _(docs: list, func: CallableOnStr, **kwargs) -> list:
    """Dispatch for list."""
    return [func(x, **kwargs) for x in docs]


@_process.register
def _(docs: set, func: CallableOnStr, **kwargs) -> set:
    """Dispatch for Set."""
    return {func(x, **kwargs) for x in docs}


@_process.register
def _(docs: Series, func: CallableOnStr, **kwargs) -> Series:
    """Dispatch for Series."""
    return docs.map(partial(func, **kwargs))


@_process.register
def _(docs: ndarray, func: CallableOnStr, **kwargs) -> ndarray:
    """Dispatch for 1darray."""
    _check_1d(docs)
    return np.array([func(x, **kwargs) for x in docs])


@_process.register
def _(docs: str, func: CallableOnStr, **kwargs) -> Any:
    """Dispatch for single string."""
    return func(docs, **kwargs)


def lowercase(docs: Documents) -> Documents:
    """Convenience function to make letters lowercase.

    Just a named, polymorphic, wrapper around str.lower.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to make lowercase.

    Returns
    -------
    str or iterable of str
        Lowercase document(s).
    """

    def lower(x):
        return x.lower()

    return _process(docs, lower)


@deprecated(details="Use `tokens.filter_length` instead.")
def strip_short(docs: Documents, minsize: int = 3) -> Documents:
    """Remove words with less than `minsize` characters.

    Polymorphic wrapper for gensim.parsing.preprocessing.strip_short.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    minsize: int, optional
        Minimum word length in characters; defaults to 3.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gensim_pp.strip_short, minsize=minsize)


def strip_multiwhite(docs: Documents) -> Documents:
    """Replace stretches of whitespace with a single space.

    Polymorphic wrapper for gensim.parsing.preprocessing.strip_multiple_whitespaces.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gensim_pp.strip_multiple_whitespaces)


def strip_numeric(docs: Documents) -> Documents:
    """Remove numeric characters.

    Polymorphic wrapper for gensim.parsing.preprocessing.strip_numeric.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gensim_pp.strip_numeric)


def strip_non_alphanum(docs: Documents) -> Documents:
    """Remove all non-alphanumeric characters.

    Polymorphic wrapper for gensim.parsing.preprocessing.strip_non_alphanum.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gensim_pp.strip_non_alphanum)


def split_alphanum(docs: Documents) -> Documents:
    """Split up the letters and numerals in alphanumeric sequences.

    Polymorphic wrapper for gensim.parsing.preprocessing.split_alphanum.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gensim_pp.split_alphanum)


def limit_repeats(docs: Documents) -> Documents:
    """Limit strings of repeating characters (e.g. 'aaaaa') to length 3.

    Polymorphic wrapper for nltk.tokenize.casual.reduce_lengthening. This is
    the function used by TweetTokenizer if `reduce_len=True`.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, nltk_casual.reduce_lengthening)


def strip_html_tags(docs: Documents) -> Documents:
    """Remove HTML tags.

    Polymorphic wrapper for gensim.parsing.preprocessing.strip_tags.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    return _process(docs, gensim_pp.strip_tags)


def decode_html_entities(docs: Documents) -> Documents:
    return _process(docs, html.unescape)


@deprecated(details="Use `tokens.porter_stem` instead.")
@singledispatch
def stem_text(docs: Documents, lowercase: bool = False) -> Documents:
    """Apply Porter stemming to text.

    Keeps cache to reuse previously computed results. This improves
    performance if the function is called repeatedly as a step in a
    preprocessing pipeline.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    # This is the dispatch for non-str types.
    _validate_docs(docs)
    return _process(docs, stem_text, lowercase=lowercase)


@stem_text.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str, lowercase: bool = False):
    """Dispatch for str. Keeps cache to reuse previous results."""
    stem = nltk.PorterStemmer()
    tokens = moses_tokenize(docs)
    tokens = [stem.stem(x, lowercase) for x in tokens]
    return moses_detokenize(tokens)


def strip_twitter_handles(docs: Documents) -> Documents:
    """Remove Twitter @mentions or other similar handles.

    Polymorphic wrapper for nltk.tokenize.casual.remove_handles.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    _validate_docs(docs)
    return _process(docs, nltk_casual.remove_handles)


def uni2ascii(docs: Documents) -> Documents:
    """Transliterate Unicode symbols into ASCII or drop.

    Polymorphic wrapper for sklearn.feature_extraction.text.strip_accents_ascii.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    _validate_docs(docs)
    return _process(docs, skl_text.strip_accents_ascii)


def deaccent(docs: Documents) -> Documents:
    """Transliterate accentuated unicode symbols into their simple counterpart.

    Polymorphic wrapper for sklearn.feature_extraction.text.strip_accents_unicode.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    _validate_docs(docs)
    return _process(docs, skl_text.strip_accents_unicode)


def strip_punct(
    docs: Documents,
    repl: str = " ",
    punct: str = string.punctuation,
    exclude: str = "",
) -> Documents:
    """Strip punctuation, optionally excluding some characters.

    Extension of gensim.parsing.preprocessing.strip_punctuation.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    repl : str, optional
        Replacement character, by default " ".
    punct : str, optional
        String of punctuation symbols, by default `string.punctuation`.
    exclude : str, optional
        String of symbols to exclude, empty by default.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    _validate_docs(docs)
    charset = set(punct).union(exclude)
    if not charset.issubset(string.punctuation):
        invalid = "".join(charset.difference(string.punctuation))
        raise ValueError(f"Invalid punctuation symbols: '{invalid}'")
    if exclude:
        exclude = re.escape(exclude)
        punct = re.sub(fr"[{exclude}]", "", punct)
    re_punct = re.compile(fr"[{re.escape(punct)}]")

    def sub(string):
        return re_punct.sub(repl, string)

    return _process(docs, sub)


@deprecated(details="Use `tokens.filter_stopwords` instead.")
@singledispatch
def strip_stopwords(
    docs: Documents, stopwords: Collection[str] = gensim_pp.STOPWORDS
) -> Documents:
    """Remove stopwords from document(s).

    Parameters
    ----------
    docs : Documents
        Documents for stopword removal.
    stopwords : collection of str, optional
        Set of stopwords to remove. Defaults to Gensim stopwords.

    Returns
    -------
    str or iterable of str
        Documents with stopwords removed.
    """
    _validate_docs(docs)
    return _process(docs, strip_stopwords, stopwords=stopwords)


@strip_stopwords.register
def _(docs: str, stopwords: Collection[str] = gensim_pp.STOPWORDS):
    """Dispatch for str."""
    stopwords = set(stopwords)
    tokens = [x for x in moses_tokenize(docs) if x not in stopwords]
    return moses_detokenize(tokens)


def space_tokenize(docs: Documents) -> Union[TokenSeq, Collection[TokenSeq]]:
    """Convenience function to tokenize by whitespace.

    Uses regex to split on any whitespace character.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize.

    Returns
    -------
    list of str or iterable of lists of str
        Tokenized document(s).
    """
    # Make sure docs are good
    _validate_docs(docs)

    re_white = re.compile(r"\s+")
    return _process(docs, re_white.split)


def moses_tokenize(docs: Documents, lang="en") -> Union[TokenSeq, Collection[TokenSeq]]:
    # Make sure docs are good
    _validate_docs(docs)

    tokenizer = MosesTokenizer(lang=lang)
    return _process(docs, tokenizer.tokenize)


@singledispatch
def tokenize_tag(
    docs: Documents,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    tagset: str = None,
    lang: str = "eng",
    fuse_tuples: bool = False,
    sep: str = DEFAULT_SEP,
    as_tokens: bool = True,
) -> Union[
    Documents,
    TaggedTokenSeq,
    TokenSeq,
    Collection[TaggedTokenSeq],
    Collection[TokenSeq],
]:
    """Tokenize and POS-tag documents.

    Keeps cache to reuse previously computed results. This improves
    performance if the function is called repeatedly as a step in a
    preprocessing pipeline.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize and tag.
    tokenizer: callable, optional
        Callable to apply to documents.
    tagset: str, optional
        Name of NLTK tagset to use.
    fuse_tuples: bool, optional
        Join tuples (word, tag) into single strings using
        separator `sep`.
    sep: str, optional
        Separator string for joining (word, tag) tuples. Only
        relevant if `fuse_tuples=True`.

    Returns
    -------
    list of tuples of str, list of str, collection of list of tuples of str,
    or collection of list of str
        Tokenized and tagged document(s).
    """
    # This is the dispatch for non-str types.

    # Check the docs
    _validate_docs(docs)

    # Process using dispatch for singular str
    docs = _process(
        docs,
        tokenize_tag,
        tokenizer=tokenizer,
        tagset=tagset,
        lang=lang,
        fuse_tuples=fuse_tuples,
        sep=sep,
        as_tokens=as_tokens,
    )
    return docs


@tokenize_tag.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(
    docs: str,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    tagset: str = None,
    lang: str = "eng",
    fuse_tuples: bool = False,
    sep: str = DEFAULT_SEP,
    as_tokens: bool = True,
) -> Union[str, TokenSeq, TaggedTokenSeq]:
    """Dispatch for str. Keeps cache to reuse previous results."""
    # Tuples must be fused if returning a str
    if not as_tokens:
        fuse_tuples = True

    # Tokenize and tag
    docs = tokenizer(docs)
    docs = nltk.pos_tag(docs, lang=lang, tagset=tagset)

    if fuse_tuples:
        # Fuse tuples
        docs = [nltk.tuple2str(x, sep) for x in docs]
    return docs if as_tokens else " ".join(docs)


@deprecated(details="Use `tokens.pos_tag` instead.")
@singledispatch
def mark_pos(docs: Documents, tagset: str = None, sep: str = DEFAULT_SEP) -> Documents:
    """Mark POS in documents with suffix.

    Keeps cache to reuse previously computed results. This improves
    performance if the function is called repeatedly as a step in a
    preprocessing pipeline.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize and tag.
    tagset: str, optional
        Name of NLTK tagset to use.
    sep: str, optional
        Separator string for joining (word, tag) tuples.

    Returns
    -------
    str or collection of str
        POS marked document(s).
    """
    # Check the docs
    _validate_docs(docs)

    # Process using str dispatch
    return _process(docs, mark_pos, tagset=tagset, sep=sep)


@mark_pos.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str, tagset: str = None, sep: str = DEFAULT_SEP) -> Documents:
    """Dispatch for str. Keeps cache to reuse previous results."""
    # Get tokens with POS suffixes
    tokens = tokenize_tag(
        docs,
        tokenizer=moses_tokenize,
        tagset=tagset,
        fuse_tuples=True,
        sep=sep,
    )
    # Detokenize and return
    return moses_detokenize(tokens)


@deprecated(details="Use `tokens.mark_negation` instead.")
@singledispatch
def mark_negation_text(
    docs: Documents, double_neg_flip: bool = False, sep: str = DEFAULT_SEP
) -> Documents:
    """Mark words '_NEG' which fall between a negating word and punctuation mark.

    Polymorphic wrapper for nltk.sentiment.util.mark_negation. Keeps cache to reuse
    previously computed results. This improves performance if the function is called
    repeatedly as a step in a preprocessing pipeline.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to mark negation in.
    double_neg_flip : bool, optional
        Double negation does not count as negation, false by default.
    sep : str, optional
        Separator for 'NEG' suffix.

    Returns
    -------
    str or iterable of str
        Same as input type, with negation marked.
    """
    # Check the docs
    _validate_docs(docs)

    # Process using str dispatch
    return _process(docs, mark_negation_text, double_neg_flip=double_neg_flip, sep=sep)


@mark_negation_text.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str, double_neg_flip: bool = False, sep: str = DEFAULT_SEP) -> str:
    """Dispatch for str. Keeps cache to reuse previous results."""
    # Tokenize with Treebank
    docs = moses_tokenize(docs)

    # Apply nltk.sentiment.util.mark_negation
    docs = nltk_mark_neg(docs, double_neg_flip=double_neg_flip)

    # Subsitute underscore for `sep`
    re_neg = re.compile(r"_NEG$")
    for i, word in enumerate(docs):
        docs[i] = re_neg.sub(f"{sep}NEG", word)

    # Detokenize and return
    return moses_detokenize(docs)


@deprecated(details="Use `tokens.wordnet_lemmatize` instead.")
@singledispatch
def lemmatize_text(docs: Documents) -> Documents:
    """Lemmatize document(s) using POS-tagging and WordNet lemmatization.

    Tag parts of speech and feed tagged unigrams into WordNet Lemmatizer.
    Keeps cache to reuse previously computed results. This improves performance
    if the function is called repeatedly as a step in a preprocessing pipeline.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to lemmatize.
    tokenizer : callable (str -> list of str), optional
        Callable for tokenizing document(s).
    as_tokens : bool, optional
        Return document(s) as list(s) of tokens, by default False.

    Returns
    -------
    str, collection of str
    """
    # This is the fallback dispatch
    # Make sure docs are good
    _validate_docs(docs)

    # Process using the str dispatch
    return _process(docs, lemmatize_text)


@lemmatize_text.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(docs: str) -> str:
    """Dispatch for str. Keeps cache to reuse previous results."""
    # Tokenize and tag POS
    tokens = tuple(moses_tokenize(docs))
    tokens = wordnet_lemmatize(tokens)

    # Detokenize and return
    return moses_detokenize(tokens)
