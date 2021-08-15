from functools import partial
from typing import Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.figure import Figure
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.io.formats.style import Styler
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.metrics import classification_report as sk_report
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

from ..._validation import _validate_train_test_split
from ...plotting.utils import smart_subplots
from ...typing import EstimatorLike, SeedLike, SeriesOrArray
from ...utils import pandas_heatmap


def _get_estimator_name(estimator: EstimatorLike) -> str:
    """Returns estimator class name.

    If a Pipeline is passed, returns the class name of the final estimator.

    Parameters
    ----------
    estimator : Estimator or Pipeline
        Estimator to get class name for.

    Returns
    -------
    str
        Class name.
    """
    if isinstance(estimator, Pipeline):
        name = estimator[-1].__class__.__name__
    else:
        name = estimator.__class__.__name__
    return name


def high_correlations(data: DataFrame, thresh: float = 0.75) -> Series:
    """Get non-trivial feature correlations at or above `thresh`.

    Parameters
    ----------
    data : DataFrame
        Data for finding high correlations.
    thresh : float, optional
        High correlation threshold, by default 0.75.

    Returns
    -------
    Series
        High correlations.
    """
    corr_df = pd.get_dummies(data).corr()
    mask = np.tril(np.ones_like(corr_df, dtype=np.bool_))
    corr_df = corr_df.mask(mask).stack()
    high = corr_df >= thresh
    return corr_df[high]


def classification_report(
    y_test: SeriesOrArray,
    y_pred: ndarray,
    zero_division: str = "warn",
    heatmap: bool = False,
) -> Union[DataFrame, Styler]:
    """Return diagnostic report for classification, optionally styled as a heatmap.

    Parameters
    ----------
    y_test : Series or ndarray of shape (n_samples,)
        Target test set.
    y_pred :  ndarray of shape (n_samples,)
        Values predicted from predictor test set.
    zero_division : str, optional
        Value to return for division by zero: 0, 1, or 'warn'.
    heatmap : bool, optional
        Style report as a heatmap, by default False.

    Returns
    -------
    DataFrame or Styler (if `heatmap = True`)
        Diagnostic report table.
    """
    # Coerce to array
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Get unique labels
    labels = np.unique(np.hstack([y_test.flatten(), y_pred.flatten()]))

    # Get Scikit-Learn report
    report = DataFrame(
        sk_report(y_test, y_pred, output_dict=True, zero_division=zero_division)
    )

    # Set column order
    order = report.columns.to_list()[: labels.size] + [
        "macro avg",
        "weighted avg",
        "accuracy",
    ]
    report = report.loc[:, order]

    # Normalize support
    support = report.loc["support"].iloc[: labels.size]
    support /= report.loc["support", "macro avg"]
    report.loc["support"] = support

    # Calculate balanced accuracy
    report["bal accuracy"] = balanced_accuracy_score(y_test, y_pred)

    # Mask extra cells
    mask = np.array([[0, 1, 1, 1], [0, 1, 1, 1]]).T.astype(np.bool_)
    report[["accuracy", "bal accuracy"]] = report.filter(like="accuracy", axis=1).mask(
        mask
    )
    if heatmap:
        report = pandas_heatmap(report, subset=labels, axis=1, vmin=0, vmax=1)
    return report


def compare_scores(
    est_1: EstimatorLike,
    est_2: EstimatorLike,
    X_test: Union[DataFrame, Series, ndarray],
    y_test: Union[Series, ndarray],
    prec: int = 3,
    heatmap: bool = True,
) -> Union[DataFrame, Styler]:
    """Compare the classification reports of two fitted estimators.

    Parameters
    ----------
    est_1 : EstimatorLike
        Fitted classifier or pipeline ending with classifier.
    est_2 : EstimatorLike
        Fitted classifier or pipeline ending with classifier.
    X_test : DataFrame, Series, or ndarray
        Independent variables test set.
    y_test : Series or ndarray
        Target variable test set.
    prec : int, optional
        Decimal places to show for floats, by default 3.
    heatmap : bool, optional
        Style result as a heatmap, by default True.

    Returns
    -------
    Styler or DataFrame
        Comparison table showing score differences.
    """
    # Get classification reports
    scores_1 = classification_report(y_test, est_1.predict(X_test), precision=prec)
    scores_2 = classification_report(y_test, est_2.predict(X_test), precision=prec)

    # Create comparison table
    result = scores_1.compare(scores_2, keep_equal=True, keep_shape=True)
    name_1 = _get_estimator_name(est_1)
    name_2 = _get_estimator_name(est_2)
    result.rename(columns=dict(self=name_1, other=name_2), inplace=True)
    result = result.T
    return pandas_heatmap(result) if heatmap else result


def classification_plots(
    estimator: Union[BaseEstimator, Pipeline],
    X_test: Union[DataFrame, ndarray],
    y_test: Union[Series, ndarray],
    pos_label: Union[bool, int, float, str] = None,
    average: str = "macro",
    size: Tuple[float, float] = (5, 5),
) -> Figure:
    """Plot confusion matrix, ROC curve, and precision-recall curve.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Fitted classification estimator or pipeline with fitted
        final estimator to evaluate.
    X_test : DataFrame or ndarray of shape (n_samples, n_features)
        Predictor test set.
    y_test : Series or ndarray of shape (n_samples,)
        target test set.
    average : str, optional
        Method of averaging: 'micro', 'macro' (default), 'weighted', 'samples'.
    size: tuple (float, float), optional
        Size of each subplot; (5, 5) by default.

    Returns
    -------
    Figure
        Figure containing three subplots.
    """

    # Validate shapes
    assert X_test.shape[0] == y_test.shape[0]
    # if X_test.ndim == 1:
    #     X_test = np.expand_dims(X_test, axis=1)
    labels = estimator.classes_
    if pos_label is None:
        # Default positive class
        pos_label = labels[-1]
    if labels.size > 2:
        # One subplot for multi-class
        fig, ax1 = plt.subplots(figsize=size)
    else:
        # Three subplots for binary
        fig, (ax1, ax2, ax3) = smart_subplots(nplots=3, ncols=3, size=size)
        plot_roc_curve(
            estimator,
            X_test,
            y_test,
            pos_label=pos_label,
            ax=ax2,
        )
        plot_precision_recall_curve(
            estimator, X_test, y_test, pos_label=pos_label, ax=ax3
        )

        # Draw dummy lines for comparison
        if is_numeric_dtype(y_test):
            baseline_style = dict(lw=2, linestyle=":", color="r", alpha=1)
            ax2.plot([0, 1], [0, 1], **baseline_style)
            ax3.plot([0, 1], [y_test.mean()] * 2, **baseline_style)
            ax3.plot([0, 0], [y_test.mean()], **baseline_style)

        # Get predictions for calculating scores
        try:            
            try:
                y_score = estimator.predict_proba(X_test)[:, pos_label]
            except IndexError:
                y_score = estimator.predict_proba(X_test)[:, -1]
        except AttributeError:
            y_score = estimator.decision_function(X_test)
        auc_score = roc_auc_score(y_test, y_score, average=average, labels=labels).round(2)
        ap_score = average_precision_score(y_test, y_score, average=average, pos_label=pos_label).round(2)

        ax2.set_title(f"Receiver Operating Characteristic Curve: AUC = {auc_score}")
        ax3.set_title(f"Precision-Recall Curve: AP = {ap_score}")
        ax2.get_legend().set_visible(False)
        ax3.get_legend().set_visible(False)

    # For both binary and multi-class
    plot_confusion_matrix(
        estimator,
        X_test,
        y_test,
        cmap="Blues",
        normalize="true",
        colorbar=False,
        ax=ax1,
    )

    ax1.set_title("Normalized Confusion Matrix")

    # Hide disruptive grid
    ax1.grid(False)
    fig.tight_layout()

    return fig


def plot_double_confusion_matrices(
    estimator: Union[BaseEstimator, Pipeline],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    cmap: str = "Blues",
    colorbar: bool = False,
    size: Tuple[float, float] = (5, 5),
    **kwargs,
) -> plt.Figure:
    """Plot normalized and raw confusion matrices side by side.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Fitted classification estimator or pipeline with fitted
        final estimator to evaluate.
    X_test : DataFrame or ndarray of shape (n_samples, n_features)
        Predictor test set.
    y_test : Series or ndarray of shape (n_samples,)
        target test set.
    cmap : str, optional
        Matplotlib colormap for the matrices, by default "Blues".
    colorbar : bool, optional
        Show colorbars, by default False.
    size: tuple (float, float), optional
        Size of each subplot; (5, 5) by default.
    **kwargs:
        Keyword arguments passed to `sklearn.metrics.plot_confusion_matrix`
        for both plots.

    Returns
    -------
    Figure
        Two confusion matrices.
    """
    fig, (ax1, ax2) = smart_subplots(nplots=2, size=size, ncols=2, sharey=True)

    # Generic partial function for both plots
    plot_matrix = partial(
        plot_confusion_matrix,
        estimator=estimator,
        X=X_test,
        y_true=y_test,
        cmap=cmap,
        colorbar=colorbar,
        **kwargs,
    )

    # Plot normalized matrix
    plot_matrix(ax=ax1, normalize="true")

    # Plot raw matrix
    plot_matrix(ax=ax2)

    # Set titles
    ax1.set(title="Normalized Confusion Matrix")
    ax2.set(title="Raw Confusion Matrix")

    fig.tight_layout()
    return fig


def standard_report(
    estimator: Union[BaseEstimator, Pipeline],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    pos_label: Union[bool, int, float, str] = None,
    zero_division: str = "warn",
    size: Tuple[float, float] = (4, 4),
    heatmap: bool = True,
) -> None:
    """Display standard report of diagnostic metrics and plots for classification.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted classification estimator for evaluation.
    X_test : DataFrame or ndarray of shape (n_samples, n_features)
        Predictor test set.
    y_test : Series or ndarray of shape (n_samples,)
        Target test set.
    zero_division : str, optional
        Value to return for division by zero: 0, 1, or 'warn'.
    """
    table = classification_report(
        y_test, estimator.predict(X_test), zero_division=zero_division, heatmap=heatmap
    )
    classification_plots(
        estimator,
        X_test,
        y_test,
        pos_label=pos_label,
        size=size,
    )
    display(table)


def test_fit(
    estimator: Union[BaseEstimator, Pipeline],
    X_train: Union[DataFrame, ndarray],
    X_test: Union[DataFrame, ndarray],
    y_train: Union[Series, ndarray],
    y_test: Union[Series, ndarray],
    resplit: bool = False,
    random_state: SeedLike = None,
    pos_label: Union[bool, int, float, str] = None,
    zero_division: str = "warn",
    size: Tuple[float, float] = (4, 4),
):
    """Train and test, then show standard report.

    Recommended: create a functools.partial object after
    your train-test-split and plug in the data.

    Parameters
    ----------
    estimator : Estimator or Pipeline
        Classification estimator or pipeline ending with estimator.
    X_train : DataFrame or ndarray
        Independent variables training set.
    X_test : DataFrame or ndarray
        Independent variables test set.
    y_train : Series or ndarray
        Target variable training set.
    y_test : Series or ndarray
        Target variable test set.
    resplit: bool, optional
        Do a fresh split before training, defaults to False.
    random_state: int, ndarray, or RandomState, optional
        Seed or RandomState for resplit, defaults to None.
    pos_label : bool, int, float, or str, optional
        Label of positive class, by default None.
    zero_division : str, optional
        Action for zero-division, by default "warn".
    size : tuple of floats, optional
        Size of each diagnostic plot, by default (4, 4).
    """
    # Check data shapes
    _validate_train_test_split(X_train, X_test, y_train, y_test)

    if resplit:
        # Concatenate, shuffle, and resplit. Should
        # be developed as a separate generic function.
        if X_train.ndim > 1:
            X = pd.concat([DataFrame(X_train), DataFrame(X_test)], axis=0)
        else:
            X = pd.concat([Series(X_train), Series(X_test)], axis=0)
        y = pd.concat([Series(y_train), Series(y_test)], axis=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            shuffle=True,
            stratify=y,
            random_state=random_state,
        )

    estimator.fit(X_train, y_train)

    standard_report(
        estimator,
        X_test,
        y_test,
        pos_label=pos_label,
        zero_division=zero_division,
        size=size,
    )