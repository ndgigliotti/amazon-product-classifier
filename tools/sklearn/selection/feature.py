from typing import List, Union

import pandas as pd
from feature_engine.selection import (
    SmartCorrelatedSelection as SmartCorrelatedSelectionFE,
)
from IPython.display import HTML, display
from sklearn.utils.validation import check_is_fitted

Variables = Union[None, int, str, List[Union[str, int]]]


class SmartCorrelatedSelection(SmartCorrelatedSelectionFE):
    """Wrapper for feature_engine.selection.SmartCorrelatedSelection."""

    def __init__(
        self,
        variables: Variables = None,
        method: str = "pearson",
        threshold: float = 0.8,
        missing_values: str = "ignore",
        selection_method: str = "missing_values",
        estimator=None,
        scoring: str = "roc_auc",
        cv: int = 3,
        verbose: bool = False,
    ):
        super().__init__(
            variables=variables,
            method=method,
            threshold=threshold,
            missing_values=missing_values,
            selection_method=selection_method,
            estimator=estimator,
            scoring=scoring,
            cv=cv,
        )
        self.verbose = verbose

    @property
    def selected_features_(self):
        check_is_fitted(self)
        corr_superset = set().union(*self.correlated_feature_sets_)
        return list(corr_superset.difference(self.features_to_drop_))

    def show_report(self):
        check_is_fitted(self)
        name = self.__class__.__name__
        info = [
            pd.Series(self.selected_features_, name="Selected"),
            pd.Series(self.features_to_drop_, name="Rejected"),
        ]
        info = pd.concat(info, axis=1)
        name = self.__class__.__name__
        info = info.T.to_html(na_rep="", header=False, max_cols=6, notebook=True)
        info = f"<h4>{name}</h4>{info}"
        display(HTML(info))

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        super().fit(X, y=y)
        if self.verbose:
            self.show_report()
        return self
