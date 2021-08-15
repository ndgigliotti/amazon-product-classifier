from functools import singledispatchmethod
from typing import Union

from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._validation import _validate_transformer
from ..typing import ArrayLike


class PandasWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None):
        super().__init__()
        self.transformer = transformer
        if not self.passthrough:
            _validate_transformer(self.transformer)

    @property
    def passthrough(self):
        return self.transformer is None or self.transformer == "passthrough"

    def unwrap(self):
        return self.transformer

    def _reconst_frame(self, X: Union[ndarray, spmatrix]):
        if isinstance(X, spmatrix):
            X = X.todense()
        X = DataFrame(X, self.index_, self.columns_)
        for column, dtype in self.dtypes_.items():
            X[column] = X[column].astype(dtype)
        return X

    @singledispatchmethod
    def fit(self, X: ArrayLike, y: ArrayLike = None, **fit_params):
        if not self.passthrough:
            self.transformer.fit(X, y, **fit_params)
        return self

    @fit.register
    def _(self, X: DataFrame, y: Series = None, **fit_params):
        self.index_ = X.index
        self.columns_ = X.columns
        self.dtypes_ = X.dtypes
        if not self.passthrough:
            if y is not None:
                y = y.to_numpy()
            self.transformer.fit(X.to_numpy(), y, **fit_params)
        return self

    @fit.register
    def _(self, X: Series, y: Series = None, **fit_params):
        if not self.passthrough:
            self.fit(X.to_frame(), y, **fit_params)
        return self

    @singledispatchmethod
    def transform(self, X: ArrayLike):
        check_is_fitted(self)
        if not self.passthrough:
            X = self.transformer.transform(X)
        return X

    @transform.register
    def _(self, X: DataFrame):
        check_is_fitted(self)
        if not self.passthrough:
            init_shape = X.shape
            X = self.transformer.transform(X.to_numpy())
            if X.shape != init_shape:
                raise RuntimeError("Transformation must preserve shape")
            X = self._reconst_frame(X)
        return X

    @transform.register
    def _(self, X: Series):
        return self.transform(X.to_frame()).squeeze()

    @singledispatchmethod
    def inverse_transform(self, X: ArrayLike):
        check_is_fitted(self)
        if not self.passthrough:
            X = self.transformer.inverse_transform(X)
        return X

    @inverse_transform.register
    def _(self, X: DataFrame):
        check_is_fitted(self)
        if not self.passthrough:
            init_shape = X.shape
            X = self.transformer.inverse_transform(X.to_numpy())
            if X.shape != init_shape:
                raise RuntimeError("Inverse transformation must preserve shape")
            X = self._reconst_frame(X)
        return X

    @inverse_transform.register
    def _(self, X: Series):
        return self.inverse_transform(X.to_frame()).squeeze()
