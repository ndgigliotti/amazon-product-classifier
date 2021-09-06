import os
from typing import Collection, Iterable, Sequence, Union
import warnings

from numpy import ndarray
from pandas.core.generic import NDFrame

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from pandas.core.dtypes.missing import isna, notna

from .typing import ArrayLike, Documents, Strings, TokenSeq, TokenTuple


def _validate_orient(orient: str):
    """Check that `orient` is 'h' or 'v'."""
    if orient.lower() not in {"h", "v"}:
        raise ValueError(f"`orient` must be 'h' or 'v', not {orient}")


def _validate_sort(sort: str):
    """Check that `sort` is 'asc' or 'desc'."""
    if sort is None:
        pass
    elif sort.lower() not in {"asc", "desc"}:
        raise ValueError("`sort` must be 'asc', 'desc', or None")


def _validate_train_test_split(X_train, X_test, y_train, y_test):
    """Check that data shapes are consistent with proper split."""
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    if X_train.ndim > 1:
        assert X_train.shape[1] == X_test.shape[1]
    if y_train.ndim > 1:
        assert y_train.shape[1] == y_test.shape[1]


def _check_1dlike(data: ArrayLike):
    """Check that data is shape (n_samples,) or (n_samples, 1)."""
    if not (hasattr(data, "shape") and hasattr(data, "ndim")):
        raise TypeError(f"Expected array-like, got {type(data)}")
    msg = "Data must be shape (n_samples,) or (n_samples, 1)."
    if data.ndim == 2 and data.shape[1] > 1:
        raise ValueError(msg)
    elif data.ndim > 2:
        raise ValueError(msg)


def _check_1d(data: ArrayLike):
    """Check that data is 1-dimensional."""
    if not (hasattr(data, "ndim")):
        raise TypeError(f"Expected array-like, got {type(data)}")
    elif data.ndim > 1:
        raise ValueError(
            f"Expected data to be 1-dimensional, but shape is {data.shape}."
        )


def _validate_transformer(obj: TransformerMixin):
    """Check that `obj` is Scikit Learn transformer or pipeline."""
    trans = isinstance(obj, TransformerMixin)
    pipe = isinstance(obj, Pipeline)
    if not trans or pipe:
        raise TypeError(
            f"Expected Scikit Learn transformer or pipeline, got {type(obj)}."
        )


def _validate_raw_docs(X: Iterable[str]):
    """Used for text vectorizers. Makes sure X is iterable over raw documents."""
    if not isinstance(X, Iterable) or isinstance(X, str):
        raise TypeError(
            f"Expected iterable over raw documents, {type(X)} object received."
        )
    if hasattr(X, "ndim") and X.ndim > 1:
        raise ValueError(
            f"Expected iterable over raw documents, received {X.ndim}-d {type(X)}."
        )
    for doc in X:
        if not isinstance(doc, str):
            raise TypeError(f"Expected iterable of only str; encountered {type(doc)}.")


def _validate_strings(strings: Strings):
    """Check that `strings` is str or other iterable."""
    # If str, say no more
    if isinstance(strings, str):
        return

    if not isinstance(strings, Iterable):
        raise TypeError(
            f"Expected str or iterable of str; {type(strings)} object received."
        )
            


def _check_overwrite(filename, action="warn"):
    if filename is not None:
        filename = os.path.normpath(filename)
        basename = os.path.basename(filename)
    if os.path.exists(filename):
        if action == "raise":
            raise FileExistsError(f"'{basename}' already exists.")
        elif action == "warn":
            warnings.warn(f"'{basename}' already exists and will be overwritten.")
        else:
            _invalid_value("action", action, ("warn", "raise"))


def _validate_tokens(tokens: TokenSeq, check_str=False):
    if not isinstance(tokens, Sequence):
        raise TypeError(f"Expected sequence of str, got {type(tokens)}.")
    if check_str:
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(
                    f"Expected sequence of str; encountered {type(token)} when iterating."
                )


def _invalid_value(param_name, value, valid_options=None):
    if valid_options is not None:
        raise ValueError(
            f"Invalid value {value} for `{param_name}`. Valid options: {valid_options}"
        )
    else:
        raise ValueError(f"Invalid value for `{param_name}`: {value}")
