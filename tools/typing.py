from re import Pattern
from typing import Any, Callable, Collection, Iterable, List, Sequence, Tuple, TypeVar, Union

from numpy import ndarray
from numpy.random import BitGenerator, RandomState
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse.base import spmatrix
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# RandomState-related types
SeedLike = Union[int, ndarray, BitGenerator, RandomState]
LegacySeedLike = Union[int, ndarray, RandomState]

# String or compiled regex
PatternLike = Union[str, Pattern]

# Series or NDArray
SeriesOrArray = Union[Series, ndarray]
FrameOrSeries = Union[Series, DataFrame]
ArrayLike = Union[DataFrame, Series, ndarray, spmatrix]
# Estimator or Pipeline
EstimatorLike = Union[BaseEstimator, Pipeline]

# One or more strings
Documents = Union[str, Iterable[str]]
Strings = Documents

# Collection of word tokens
Tokens = Collection[str]
TokenTuple = Tuple[str]

# One or more token sequences
TokenDocs = Union[Tokens, Collection[Tokens]]

# List of tokens with POS tags
TaggedTokens = Collection[Tuple[str, str]]
TaggedTokenTuple = Tuple[Tuple[str, str]]

# One or more tagged token sequences
TaggedTokenDocs = Union[TaggedTokens, Collection[TaggedTokens]]

# Function which takes a string
CallableOnStr = Callable[[str], Any]

# Function which tokenizes a string
Tokenizer = Callable[[str], Tokens]
