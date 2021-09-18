from collections import defaultdict
import html
import re
import string
from functools import lru_cache, partial, singledispatch
from typing import Collection, Iterable, Union
import unicodedata
from pandas import Series, DataFrame
import gensim.parsing.preprocessing as gensim_pp
import nltk
from numpy import ndarray
from nltk.corpus.reader import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from tools import plotting
import joblib
import numpy as np
from IPython.core.display import HTML
from nltk.tokenize import casual as nltk_casual
from pandas.core.frame import DataFrame
from sacremoses.tokenize import MosesTokenizer
from sklearn.feature_extraction import text as skl_text
from sklearn.utils import deprecated
from tools._validation import _validate_strings
from tools.language.processors.tokens import porter_stem, wordnet_lemmatize
from tools.language.settings import CACHE_SIZE, DEFAULT_SEP, DEFAULT_TOKENIZER
from tools.language.utils import chain_processors, process_strings, groupby_tag
from tools.typing import (
    Documents,
    PatternLike,
    TaggedTokens,
    Tokenizer,
    Tokens,
    TokenDocs,
)

SENT_DELIM = frozenset(".!?")
SPACE = re.compile(r"(\s+)")
NON_SPACE = re.compile(r"(\S+)")
END_SPACE = re.compile(r"^(\s+)|(\s+)$")
NUMERIC = re.compile(r"(\d+)")
WORD = re.compile(r"(\w+)")
HTML_TAG = re.compile(r"<([^>]+)>")
HTML_PROP = re.compile(r"\w+=[\"\'][\w\s-]+[\"\']")
HTML_PART_TAG = re.compile(r"</?\w+\b|/?\w+>")
PUNCT = re.compile(fr"([{re.escape(string.punctuation)}])")
SKL_TOKEN = re.compile(r"\b\w\w+\b")


@singledispatch
def lowercase(docs: Documents) -> Documents:
    """Convenience function to make letters lowercase.

    Just a polymorphic, wrapper around str.lower.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to make lowercase.

    Returns
    -------
    str or iterable of str
        Lowercase document(s).
    """
    return lowercase(list(docs))


@lowercase.register
def _(docs: str):
    return docs.lower()


@lowercase.register
def _(docs: list):
    return [x.lower() for x in docs]


@lowercase.register
def _(docs: Series):
    return docs.str.lower()


@lowercase.register
def _(docs: DataFrame):
    return docs.apply(lambda x: x.str.lower())


@lowercase.register
def _(docs: ndarray):
    return np.char.lower(docs.astype(str))


def strip_extra_periods(docs: Documents, n_jobs=None) -> Documents:
    periods = re.compile(r"(\.[\s\.]+)")
    strip = partial(periods.sub, ". ")
    return process_strings(docs, strip, n_jobs=n_jobs, bar_desc="strip_extra_periods")


def strip_extra_space(docs: Documents, n_jobs=None) -> Documents:
    pipeline = [NON_SPACE.findall, " ".join]
    return chain_processors(docs, pipeline, n_jobs=n_jobs, bar_desc="strip_extra_space")


def strip_end_space(docs: Documents, n_jobs=None) -> Documents:
    strip = partial(END_SPACE.sub, "")
    return process_strings(docs, strip, n_jobs=n_jobs, bar_desc="strip_end_space")


def strip_gap_space(docs: Documents, n_jobs=None) -> Documents:
    """Replace stretches of whitespace with a single space.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    strip = partial(SPACE.sub, " ")
    return process_strings(docs, strip, n_jobs=n_jobs, bar_desc="strip_gap_space")


def strip_numeric(docs: Documents, n_jobs=None) -> Documents:
    """Remove numeric characters.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    strip = partial(NUMERIC.sub, "")
    return process_strings(docs, strip, n_jobs=n_jobs, bar_desc="strip_numeric")


def strip_non_word(docs: Documents, n_jobs=None) -> Documents:
    pipeline = [WORD.findall, " ".join]
    return chain_processors(docs, pipeline, n_jobs=n_jobs, bar_desc="strip_non_word")


def limit_repeats(docs: Documents, cut=3, repl=None, n_jobs=None) -> Documents:
    """Cut strings of repeating characters (e.g. 'aaaaa') to length `cut`.

    Derived from nltk.tokenize.casual.reduce_lengthening.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.
    cut : int
        Cutoff length.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """
    if repl is None:
        repl = cut
    repeating = re.compile(fr"(.)\1{{{cut},}}")
    shorten = partial(repeating.sub, r"\1" * repl)
    return process_strings(docs, shorten, n_jobs=n_jobs, bar_desc="limit_repeats")


def strip_html_tags(docs: Documents, n_jobs=None) -> Documents:
    """Remove HTML tags.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to process.

    Returns
    -------
    str or iterable of str
        Processed document(s).
    """

    pipe = [
        partial(HTML_TAG.sub, " "),
        partial(HTML_PART_TAG.sub, " "),
        partial(HTML_PROP.sub, " "),
        NON_SPACE.findall,
        " ".join,
    ]
    return chain_processors(docs, pipe, n_jobs=n_jobs, bar_desc="strip_html_tags")


def decode_html_entities(docs: Documents, n_jobs=None) -> Documents:
    return process_strings(
        docs, html.unescape, n_jobs=n_jobs, bar_desc="decode_html_entities"
    )


def strip_twitter_handles(docs: Documents, n_jobs=None) -> Documents:
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
    return process_strings(
        docs,
        nltk_casual.remove_handles,
        n_jobs=n_jobs,
        bar_desc="strip_twitter_handles",
    )


def force_ascii(docs: Documents, n_jobs=None) -> Documents:
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
    return process_strings(
        docs, skl_text.strip_accents_ascii, n_jobs=n_jobs, bar_desc="force_ascii"
    )


def deaccent(docs: Documents, n_jobs=None) -> Documents:
    """Transliterate accentuated unicode symbols into their simple counterparts.

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
    return process_strings(
        docs, skl_text.strip_accents_unicode, n_jobs=n_jobs, bar_desc="deaccent"
    )


def strip_punct(
    docs: Documents,
    repl: str = " ",
    punct: str = string.punctuation,
    exclude: str = "",
    n_jobs=None,
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
    charset = set(punct).union(exclude)
    if not charset.issubset(string.punctuation):
        invalid = "".join(charset.difference(string.punctuation))
        raise ValueError(f"Invalid punctuation symbols: '{invalid}'")
    if exclude:
        exclude = re.escape(exclude)
        punct = re.sub(fr"[{exclude}]", "", punct)
    re_punct = re.compile(fr"[{re.escape(punct)}]")

    sub = partial(re_punct.sub, repl)
    return process_strings(docs, sub, n_jobs=n_jobs, bar_desc="strip_punct")


def regex_tokenize(
    docs: Documents,
    pattern: PatternLike,
    flags: re.RegexFlag = 0,
    n_jobs: int = None,
    bar_desc: str = "regex_tokenize",
) -> TokenDocs:
    if isinstance(pattern, str):
        pattern = re.compile(pattern, flags=flags)
    return process_strings(docs, pattern.findall, n_jobs=n_jobs, bar_desc=bar_desc)


def space_tokenize(docs: Documents, n_jobs=None) -> TokenDocs:
    """Convenience function to tokenize by whitespace.

    Uses regex to find sequences of non-whitespace characters.

    Parameters
    ----------
    docs : str or iterable of str
        Document(s) to tokenize.

    Returns
    -------
    list of str or iterable of lists of str
        Tokenized document(s).
    """
    return regex_tokenize(docs, NON_SPACE, n_jobs=n_jobs, bar_desc="space_tokenize")


def moses_tokenize(docs: Documents, lang="en", n_jobs=None) -> TokenDocs:
    return process_strings(
        docs,
        MosesTokenizer(lang=lang).tokenize,
        n_jobs=n_jobs,
        bar_desc="moses_tokenize",
    )


def skl_tokenize(docs: Documents, n_jobs=None) -> TokenDocs:
    return regex_tokenize(docs, SKL_TOKEN, n_jobs=n_jobs, bar_desc="skl_tokenize")


@singledispatch
def tokenize_tag(
    docs: Documents,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    tokenize_sents: bool = True,
    tagset: str = None,
    lang: str = "eng",
    fuse_tuples: bool = False,
    sep: str = DEFAULT_SEP,
    as_tokens: bool = True,
    n_jobs=None,
) -> Union[
    Documents,
    TaggedTokens,
    Tokens,
    Collection[TaggedTokens],
    Collection[Tokens],
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
    # Process using dispatch for singular str
    docs = process_strings(
        docs,
        tokenize_tag,
        tokenizer=tokenizer,
        tokenize_sents=tokenize_sents,
        tagset=tagset,
        lang=lang,
        fuse_tuples=fuse_tuples,
        sep=sep,
        as_tokens=as_tokens,
        n_jobs=n_jobs,
    )
    return docs


@tokenize_tag.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(
    docs: str,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
    tokenize_sents: bool = True,
    tagset: str = None,
    lang: str = "eng",
    fuse_tuples: bool = False,
    sep: str = DEFAULT_SEP,
    as_tokens: bool = True,
    n_jobs=None,
) -> Union[str, Tokens, TaggedTokens]:
    """Dispatch for str. Keeps cache to reuse previous results."""
    # Tuples must be fused if returning a str
    if not as_tokens:
        fuse_tuples = True

    # There is really only one doc
    if tokenize_sents:
        sents = nltk.sent_tokenize(docs)
    else:
        sents = [docs]

    tag_toks = []

    for sent in sents:
        sent = tokenizer(sent)
        sent = nltk.pos_tag(sent, lang=lang, tagset=tagset)
        tag_toks += sent

    if fuse_tuples:
        # Fuse tuples
        tag_toks = [nltk.tuple2str(x, sep) for x in tag_toks]
    return tag_toks if as_tokens else " ".join(tag_toks)