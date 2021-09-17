import array
from collections import defaultdict, Counter
import re
import string
from functools import lru_cache, partial
import warnings
from tools import utils
from types import MappingProxyType
import joblib
import scipy as sp
import nltk
from sklearn.utils import _IS_32BIT
from tools.language.processors.text import PUNCT
from typing import Callable
import copy
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.keyedvectors import KeyedVectors
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas.core.series import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    _VectorizerMixin,
    strip_accents_ascii,
    strip_accents_unicode,
    HashingVectorizer,
    _make_int_array,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, normalize
from sklearn.utils.validation import check_is_fitted

from .. import language as lang
from .._validation import _invalid_value, _validate_raw_docs
from ..typing import CallableOnStr

# Load VADER once
vader = SentimentIntensityAnalyzer()


@lru_cache(maxsize=1_000_000, typed=False)
def _vader_polarity_scores(text: str) -> Series:
    """Return VADER polarity scores for `text`.

    Keeps cache to reuse previous results.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    Series
        Sentiment polarity scores.
    """
    return Series(vader.polarity_scores(text))


class VaderVectorizer(BaseEstimator, TransformerMixin):
    """Extracts VADER polarity scores from short documents.

    Parameters
    ----------
    decode_html_entities: bool, optional
        Decode HTML entities such as '&mdash;' or '&lt;' or '&gt;' into symbols,
        e.g. '—', '<', '>'. True by default.
    trinarize : bool, optional
        Convert vector elements to ternary sign indicators -1.0, 0.0, and 1.0. By default False.
    category : bool, optional
        Include the positive, neutral, and negative scores in vectors, by default True.
    compound : bool, optional
        Include the compound score in vectors, by default True.
    preprocessor : CallableOnStr, optional
        Callable for preprocessing text before VADER analysis, by default None.
    norm : str, optional
        Normalization to apply, by default "l2".
    sparse : bool, optional
        Output a sparse matrix, by default True.
    """

    def __init__(
        self,
        decode_html_entities=True,
        trinarize=False,
        category=True,
        compound=True,
        preprocessor: CallableOnStr = None,
        norm=None,
        sparse=True,
    ):
        self.decode_html_entities = decode_html_entities
        self.trinarize = trinarize
        self.category = category
        self.compound = compound
        self.preprocessor = preprocessor
        self.norm = norm
        self.sparse = sparse

    def build_postprocessor(self):
        """Construct postprocessing pipeline based on parameters."""
        pipe = Pipeline([("sign", None), ("norm", None), ("csr", None)])
        if self.trinarize:
            pipe.set_params(sign=FunctionTransformer(np.sign))
        if self.norm is not None:
            pipe.set_params(norm=Normalizer(norm=self.norm))
        if self.sparse:
            pipe.set_params(csr=FunctionTransformer(csr_matrix))
        return pipe

    def _validate_params(self):
        """Validate some parameters."""
        if not (self.category or self.compound):
            raise ValueError("Either `category` or `compound` must be True.")
        if self.preprocessor is not None:
            if not isinstance(self.preprocessor, Callable):
                raise TypeError(
                    f"Expected `preprocessor` to be callable, got {type(self.preprocessor)}"
                )

    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names_

    def fit(self, X, y=None):
        """Does nothing except validate parameters and save feature names."""
        self._validate_params()
        _validate_raw_docs(X)
        self.feature_names_ = []
        if self.category:
            self.feature_names_ += ["neg", "neu", "pos"]
        if self.compound:
            self.feature_names_.append("comp")
        return self

    def transform(self, X):
        """Extracts the polarity scores and applies postprocessing."""
        # Input and param validation
        self._validate_params()
        _validate_raw_docs(X)

        # Apply preprocessing
        docs = X
        if self.decode_html_entities:
            docs = [lang.decode_html_entities(x) for x in docs]
        if self.preprocessor is not None:
            docs = self.preprocessor(docs)

        # Perform VADER analysis
        vecs = pd.DataFrame([_vader_polarity_scores(x) for x in docs])
        if self.compound and not self.category:
            vecs = vecs.loc[:, ["comp"]]
        if self.category and not self.compound:
            vecs = vecs.loc[:, ["neg", "neu", "pos"]]
        self.feature_names_ = vecs.columns.to_list()

        # Apply postprocessing and return
        postprocessor = self.build_postprocessor()
        return postprocessor.fit_transform(vecs.to_numpy())


class VectorizerMixin(_VectorizerMixin):
    def _word_ngrams(self, tokens, stop_words=None, sep="_"):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            join = sep.join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(join(original_tokens[i : i + n]))
        return tokens

    def build_preprocessor(self):
        if self.preprocessor is not None:
            return self.preprocessor

        pipe = []

        # Make case insensitive
        if self.lowercase:
            pipe.append(lang.lowercase)

        # Decode HTML entities like '&mdash;' to '—'
        if self.decode_html_entities:
            pipe.append(lang.decode_html_entities)

        # Strip accents
        if not self.strip_accents:
            pass
        elif callable(self.strip_accents):
            pipe.append(self.strip_accents)
        elif self.strip_accents == "ascii":
            pipe.append(strip_accents_ascii)
        elif self.strip_accents == "unicode":
            pipe.append(strip_accents_unicode)
        else:
            _invalid_value("strip_accents", self.strip_accents)

        # Strip HTML tags
        if self.strip_html_tags:
            pipe.append(lang.strip_html_tags)

        # Strip numerals
        if self.strip_numeric:
            pipe.append(lang.strip_numeric)

        # Strip Twitter @handles
        if self.strip_twitter_handles:
            pipe.append(lang.strip_twitter_handles)

        # Pad numeric sequences with space, i.e. 'paris64' -> 'paris 64'
        if self.pad_numeric:
            pipe.append(lang.pad_numeric)

        # Strip punctuation
        if self.strip_punct:
            if isinstance(self.strip_punct, str):
                pipe.append(partial(lang.strip_punct, punct=self.strip_punct))
            else:
                pipe.append(lang.strip_punct)

        # Strip all non-word characters (non-alphanumeric)
        if self.strip_non_word:
            pipe.append(lang.strip_non_word)

        # Strip extra whitespaces, tabs, and linebreaks
        if self.strip_extra_space:
            pipe.append(lang.strip_extra_space)

        # Wrap `pipe` into single callable
        return lang.make_preprocessor(pipe)

    def build_analyzer(self):
        """Return the complete text preprocessing pipeline as a callable.

        Handles decoding, character filtration, tokenization, word filtration,
        marking, and n-gram generation. Alternatively, returns a callable
        wrapping the custom analyzer passed via the `analyzer` parameter.

        Returns
        -------
        analyzer: callable
            A function to handle decoding, character filtration, tokenization,
            word filtration, n-gram generation, and marking.
        """
        pipe = [self.decode]

        if callable(self.analyzer):
            pipe.append(self.analyzer)
        elif self.analyzer == "char":
            pipe += [self.build_preprocessor(), self._char_ngrams]
        elif self.analyzer == "char_wb":
            pipe += [self.build_preprocessor(), self._char_wb_ngrams]
        elif self.analyzer == "word":
            preprocessor = self.build_preprocessor()
            tokenizer = self.build_tokenizer()
            pipe += [preprocessor, tokenizer]

            if self.uniq_char_thresh is not None:
                pipe.append(
                    partial(lang.uniq_char_thresh, thresh=self.uniq_char_thresh)
                )

            # Stem or lemmatize
            if callable(self.stemmer):
                pipe.append(self.stemmer)
            elif self.stemmer == "porter":
                pipe.append(lang.porter_stem)
            elif self.stemmer == "wordnet":
                pipe.append(lang.wordnet_lemmatize)

            # Remove stopwords
            if self.stop_words is not None:
                stop_words = self.get_stop_words()
                self._check_stop_words_consistency(stop_words, preprocessor, tokenizer)
                pipe.append(partial(lang.remove_stopwords, stopwords=stop_words))

            # Generate n-grams
            pipe.append(self._word_ngrams)

        else:
            raise _invalid_value("analyzer", self.analyzer, ("word", "char", "char_wb"))

        # Wrap `pipe` into single callable
        return lang.make_preprocessor(pipe)

    def get_stop_words(self):
        """Build or fetch the effective stop words set.

        Returns
        -------
        stop_words: frozenset or None
                A set of stop words.
        """
        # Exit if None
        if self.stop_words is None:
            return None
        # Process string input
        elif isinstance(self.stop_words, str):
            result = lang.fetch_stopwords(self.stop_words)
        # Assume collection if not str or None
        else:
            result = frozenset(self.stop_words)
        if self.process_stop_words:
            preprocessor = self.build_preprocessor()
            result = [preprocessor(w) for w in result]
            if self.stemmer == "porter":
                result = lang.porter_stem(result)
            elif self.stemmer == "wordnet":
                result = lang.wordnet_lemmatize(result)
            result = frozenset(result)
        return result

    def _validate_params(self):
        super()._validate_params()
        # Check `input`
        valid_input = {"filename", "file", "content"}
        if self.input not in valid_input:
            _invalid_value("input", self.input, valid_input)
        # Check `decode_error`
        valid_decode = {"strict", "ignore", "replace"}
        if self.decode_error not in valid_decode:
            _invalid_value("decode_error", self.decode_error, valid_decode)
        # Check `strip_accents`
        valid_accent = {"ascii", "unicode", None}
        if self.strip_accents not in valid_accent:
            if not callable(self.strip_accents):
                _invalid_value("strip_accents", self.strip_accents, valid_accent)
        # Check `strip_punct`
        if isinstance(self.strip_punct, str):
            if not set(self.strip_punct).issubset(string.punctuation):
                _invalid_value(
                    "strip_punct", self.strip_punct, f"subset of '{string.punctuation}'"
                )
        # Check `stemmer`
        valid_stemmer = {"porter", "wordnet", None}
        if self.stemmer not in valid_stemmer:
            if not callable(self.stemmer):
                _invalid_value("stemmer", self.stemmer, valid_stemmer)


class FreqVectorizer(TfidfVectorizer, VectorizerMixin):
    """Convert a collection of raw documents to a matrix of word-frequency features.

    Extends Scikit-Learn's `TfidfVectorizer` with advanced preprocessing options.
    These include numerous filters, stemming/lemmatization, and markers such as PoS tags.
    Some preprocessing options are applied before tokenization, and some, which require
    tokens, are applied during the tokenization step.

    There are now a wider selection of built-in stopwords sets, and these include the NLTK
    sets for many different languages. Complex stopwords queries are now also supported.


    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be a sequence of items that
        can be of type string or byte.

    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode'}
        Remove accents and perform other character normalization
        during the preprocessing step.
        * 'ascii' is a fast method that only works on characters that have
            an direct ASCII mapping.
        * 'unicode' is a slightly slower method that works on any characters.
        *  None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    decode_html_entities : bool, ** NEW **
        Decode HTML entities such as '&mdash;' or '&lt;' or '&gt;' into symbols,
        e.g. '—', '<', '>'. True by default.

    lowercase : bool
        Convert all characters to lowercase before tokenizing. True by default.

    strip_extra_space: bool, ** NEW **
        Strip extra whitespaces (including tabs and newlines). False by default.

    strip_numeric: bool, ** NEW **
        Strip numerals [0-9] from text. False by default.

    pad_numeric: bool, ** NEW **
        Add space between alphabetic and numeric characters which appear together
        in a word-like sequence. For example, 'spiderman2' would become 'spiderman 2'.
        False by default.

    strip_non_word: bool, ** NEW **
        Strip all non-alphanumeric characters (except underscore). False by default.

    strip_punct: bool or str of punctuation symbols
        If True, strip all punctuation. If passed a string of punctuation symbols, strip
        only those symbols. False by default.

    strip_twitter_handles: bool, ** NEW **
        Strip Twitter @mentions. False by default.

    strip_html_tags: bool, ** NEW **
        Strip HTML tags such as '<p>' or '<div>'. False by default.

    limit_repeats: bool, ** NEW **
        Limit strings of repeating characters, e.g. 'zzzzzzzzzzz', to length 3.

    length_filter: tuple (int, int), ** NEW **
        Drop tokens which are outside the prescribed character length range.
        Range is inclusive. Defaults to (None, None).

    stemmer: {'porter', 'wordnet'}, ** NEW **
        Stemming or lemmatization algorithm to use. Both implement caching in order to
        reuse previous computations. Valid options:
        * 'porter' - Porter stemming algorithm (faster).
        * 'wordnet' - Lemmatization using Wordnet (slower).
        * None - Do not stem tokens (default).

    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer is not callable``.

    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    analyzer : callable, default=None
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    stop_words : str, list, ** IMPROVED **
        If a string, it is passed to `tools.language.fetch_stopwords` and
        the appropriate stopword list is returned. Valid strings:
        * 'skl_english' - Scikit-Learn's English stopwords.
        * 'nltk_LANGUAGE' - Any NLTK stopwords set, where the fileid (language) follows the underscore.
            For example: 'nltk_english', 'nltk_french', 'nltk_spanish'.
        * 'gensim_english' - Gensim's English stopwords set.
        * Supports complex queries using set operators, e.g. '(nltk_french & nltk_spanish) | skl_english'.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern : str, default=r"(?u)\\b\\w\\w+\\b"
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        At most one capturing group is permitted.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams. Defaults to (1, 1).
        Only applies if ``analyzer is not callable``.

    max_df : float or int
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words). Defaults to 1.0.
        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float or int
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature. Defaults to 1.
        If float in range of [0.0, 1.0], the parameter represents a proportion
        of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        None by default.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. None by default.

    binary : bool
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs).
        False by default.

    dtype : dtype
        Type of the matrix returned by fit_transform() or transform().
        'float64' by default.

    norm : {'l2', 'l1', 'max'}
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied. None by default.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`.

    use_idf : bool
        Enable inverse-document-frequency reweighting. False by default.

    smooth_idf : bool
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions. True by default.

    sublinear_tf : bool
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        False by default.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    fixed_vocabulary_: bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user

    idf_ : array of shape (n_features,)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.
    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        decode_html_entities=True,
        lowercase=True,
        strip_extra_space=False,
        strip_numeric=False,
        pad_numeric=False,
        strip_non_word=False,
        strip_punct=False,
        strip_twitter_handles=False,
        strip_html_tags=False,
        limit_repeats=False,
        uniq_char_thresh=None,
        stemmer=None,
        preprocessor=None,
        tokenizer=None,
        token_pattern=r"\b\w\w+\b",
        analyzer="word",
        stop_words=None,
        process_stop_words=True,
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm=None,
        use_idf=False,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

        self.decode_html_entities = decode_html_entities
        self.strip_extra_space = strip_extra_space
        self.strip_numeric = strip_numeric
        self.pad_numeric = pad_numeric
        self.strip_non_word = strip_non_word
        self.strip_punct = strip_punct
        self.strip_twitter_handles = strip_twitter_handles
        self.strip_html_tags = strip_html_tags
        self.limit_repeats = limit_repeats
        self.stemmer = stemmer
        self.uniq_char_thresh = uniq_char_thresh
        self.process_stop_words = process_stop_words

    def get_keywords(self, document, top_n=None):
        check_is_fitted(self, "vocabulary_")
        vec = self.transform([document])
        vocab = utils.swap_index(Series(self.vocabulary_))
        keywords = Series(vec.data, index=vocab.loc[vec.indices], name="keywords")
        if top_n is None:
            top_n = len(keywords)
        return keywords.nlargest(top_n)

    @classmethod
    def from_sklearn(cls, vectorizer, transfer_fit=True):
        allowed_types = (CountVectorizer, TfidfVectorizer, TfidfTransformer)
        if not isinstance(vectorizer, allowed_types):
            raise TypeError(
                f"Expected {[x.__name__ for x in allowed_types]}, got {type(vectorizer).__name__}."
            )
        freq_vec = cls(**vectorizer.get_params())
        if transfer_fit:
            if hasattr(vectorizer, "vocabulary_"):
                freq_vec.vocabulary_ = copy.copy(vectorizer.vocabulary_)
            if hasattr(vectorizer, "fixed_vocabulary_"):
                freq_vec.fixed_vocabulary_ = vectorizer.fixed_vocabulary_
            if hasattr(vectorizer, "stop_words_"):
                freq_vec.stop_words_ = copy.copy(vectorizer.stop_words_)
            if hasattr(vectorizer, "idf_"):
                freq_vec.idf_ = vectorizer.idf_.copy()
        return freq_vec


class Doc2Vectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    """Doc2Vec Vectorizer for Scikit Learn API with the standard preprocessing.

    Largely derived from gensim.sklearn_api.D2VTransformer. This class exists
    because the Gensim transformer is no longer being maintained and lacks
    preprocessing functionality.

    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        decode_html_entities=True,
        lowercase=True,
        strip_extra_space=False,
        strip_numeric=False,
        pad_numeric=False,
        strip_non_word=False,
        strip_punct=False,
        strip_twitter_handles=False,
        strip_html_tags=False,
        limit_repeats=False,
        stemmer=None,
        preprocessor=None,
        tokenizer=None,
        token_pattern=r"(?u)\b\w\w+\b",
        analyzer="word",
        stop_words=None,
        process_stop_words=True,
        ngram_range=(1, 1),
        norm="l2",
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        comment=None,
        trim_rule=None,
        vector_size=100,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        hs=0,
        negative=5,
        cbow_mean=1,
        hashfxn=hash,
        epochs=10,
        sorted_vocab=1,
        batch_words=10000,
    ):
        # Related to pre/post-processing
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.decode_html_entities = decode_html_entities
        self.strip_extra_space = strip_extra_space
        self.strip_numeric = strip_numeric
        self.pad_numeric = pad_numeric
        self.strip_non_word = strip_non_word
        self.strip_punct = strip_punct
        self.strip_twitter_handles = strip_twitter_handles
        self.strip_html_tags = strip_html_tags
        self.limit_repeats = limit_repeats
        self.stemmer = stemmer
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.process_stop_words = process_stop_words
        self.ngram_range = ngram_range
        self.norm = norm

        # Related to gensim.models.Doc2Vec
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.comment = comment
        self.trim_rule = trim_rule

        # Related to gensim.models.Word2Vec
        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        # Parameter validation
        _validate_raw_docs(X)
        self._warn_for_unused_params()
        self._validate_params()
        analyzer = self.build_analyzer()
        docs = [analyzer(doc) for doc in X]
        tagged_docs = [TaggedDocument(words, [i]) for i, words in enumerate(docs)]
        self.d2v_model_ = Doc2Vec(
            documents=tagged_docs,
            dm_mean=self.dm_mean,
            dm=self.dm,
            dbow_words=self.dbow_words,
            dm_concat=self.dm_concat,
            dm_tag_count=self.dm_tag_count,
            comment=self.comment,
            trim_rule=self.trim_rule,
            vector_size=self.vector_size,
            alpha=self.alpha,
            window=self.window,
            min_count=self.min_count,
            max_vocab_size=self.max_vocab_size,
            sample=self.sample,
            seed=self.seed,
            workers=self.workers,
            min_alpha=self.min_alpha,
            hs=self.hs,
            negative=self.negative,
            cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn,
            epochs=self.epochs,
            sorted_vocab=self.sorted_vocab,
            batch_words=self.batch_words,
        )
        return self

    def get_vectors(self):
        check_is_fitted(self, "d2v_model_")
        # vecs = [self.d2v_model_.dv[i] for i in self.d2v_model_.dv.index_to_key]
        vecs = self.d2v_model_.dv.vectors
        if self.norm is not None:
            vecs = normalize(vecs, norm=self.norm, copy=False)
        return vecs

    def transform(self, X):
        check_is_fitted(self, "d2v_model_")
        _validate_raw_docs(X)
        analyzer = self.build_analyzer()
        docs = [analyzer(doc) for doc in X]
        vecs = [self.d2v_model_.infer_vector(doc) for doc in docs]
        vecs = np.reshape(np.array(vecs), (len(docs), self.d2v_model_.vector_size))
        if self.norm is not None:
            vecs = normalize(vecs, norm=self.norm, copy=False)
        return vecs

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).get_vectors()


class AverageVectorizer(FreqVectorizer):
    def __init__(
        self,
        word_vecs,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        decode_html_entities=True,
        lowercase=True,
        strip_extra_space=False,
        strip_numeric=False,
        pad_numeric=False,
        strip_non_word=False,
        strip_punct=False,
        strip_twitter_handles=False,
        strip_html_tags=False,
        limit_repeats=False,
        uniq_char_thresh=None,
        stemmer=None,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        process_stop_words=True,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        binary=True,
        dtype=np.float64,
        use_idf=False,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        if not isinstance(word_vecs, KeyedVectors):
            raise TypeError(
                f"Expected `word_vecs` to be KeyedVectors; got {type(word_vecs).__name__}."
            )
        # Mapping {id -> 1darray}
        self.word_vecs = word_vecs
        self.vector_size = word_vecs.vector_size
        vocabulary = word_vecs.key_to_index
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            norm=None,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            decode_html_entities=decode_html_entities,
            strip_extra_space=strip_extra_space,
            strip_numeric=strip_numeric,
            pad_numeric=pad_numeric,
            strip_non_word=strip_non_word,
            strip_punct=strip_punct,
            strip_twitter_handles=strip_twitter_handles,
            strip_html_tags=strip_html_tags,
            limit_repeats=limit_repeats,
            stemmer=stemmer,
            uniq_char_thresh=uniq_char_thresh,
            process_stop_words=process_stop_words,
        )

    def fit(self, X, y=None):
        # Parameter validation
        _validate_raw_docs(X)
        self._check_params()
        self._warn_for_unused_params()
        return self

    def transform(self, X):
        self.fit(X)
        freq_matrix = super().fit_transform(X)
        embedding = self.word_vecs.get_normed_vectors()
        doc_vecs = []
        append_vec = doc_vecs.append
        average = np.average
        for row in freq_matrix:
            term_idx = row.nonzero()[1]
            vec = average(embedding[term_idx], weights=row.data, axis=0)
            append_vec(vec)
        doc_vecs = np.vstack(doc_vecs)
        return doc_vecs

    def fit_transform(self, X, y=None):
        return self.transform(X)