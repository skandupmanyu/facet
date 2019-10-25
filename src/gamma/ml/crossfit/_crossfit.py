"""
Core implementation of :mod:`gamma.ml.crossfit`
"""
import logging
from abc import ABC
from typing import *

import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.sklearndf import BaseEstimatorDF, BaseLearnerDF, ClassifierDF, RegressorDF

log = logging.getLogger(__name__)

__all__ = ["BaseCrossfit", "LearnerCrossfit", "RegressorCrossfit", "ClassifierCrossfit"]

T = TypeVar("T")
T_EstimatorDF = TypeVar("T_EstimatorDF", bound=BaseEstimatorDF)
T_LearnerDF = TypeVar("T_LearnerDF", bound=BaseLearnerDF)
T_ClassifierDF = TypeVar("T_ClassifierDF", bound=ClassifierDF)
T_RegressorDF = TypeVar("T_RegressorDF", bound=RegressorDF)


class BaseCrossfit(ParallelizableMixin, ABC, Generic[T_EstimatorDF]):
    """
    :class:~gamma.sklearn all splits of a given cross-validation
    strategy, based on a pipeline.

    :param base_estimator: predictive pipeline to be fitted
    :param cv: the cross validator generating the train splits
    :param n_jobs: number of jobs to use in parallel; \
        if `None`, use joblib default (default: `None`).
    :param shared_memory: if `True` use threads in the parallel runs. If `False` \
        use multiprocessing (default: `False`).
    :param pre_dispatch: number of batches to pre-dispatch; \
        if `None`, use joblib default (default: `None`).
    :param verbose: verbosity level used in the parallel computation; \
        if `None`, use joblib default (default: `None`).
    """

    __slots__ = [
        "base_estimator",
        "cv",
        "n_jobs",
        "shared_memory",
        "verbose",
        "_model_by_split",
    ]

    def __init__(
        self,
        base_estimator: T_EstimatorDF,
        cv: BaseCrossValidator,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.base_estimator = base_estimator
        self.cv = cv

        self._model_by_split: Optional[List[T_EstimatorDF]] = None
        self._training_sample: Optional[Sample] = None

    def fit(self: T, sample: Sample, **fit_params) -> T:
        """
        Fit the base estimator to the full sample, and fit a clone of the base
        estimator to each of the train splits generated by the cross-validator
        :param sample: the sample to fit the estimators to
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the base estimator
        :return: `self`
        """
        self_typed: BaseCrossfit = self  # support better type hinting in PyCharm
        base_estimator = self_typed.base_estimator

        features = sample.features
        target = sample.target

        base_estimator.fit(X=sample.features, y=sample.target, **fit_params)

        train_splits, test_splits = tuple(zip(*self_typed.cv.split(features, target)))

        with self_typed._parallel() as parallel:
            self._model_by_split: List[T_EstimatorDF] = parallel(
                self_typed._delayed(BaseCrossfit._fit_model_for_split)(
                    base_estimator.clone(),
                    features.iloc[train_indices],
                    target.iloc[train_indices],
                    **fit_params,
                )
                for train_indices in train_splits
            )

        self_typed._training_sample = sample

        return self

    @property
    def is_fitted(self) -> bool:
        """`True` if the delegate estimator is fitted, else `False`"""
        return self._training_sample is not None

    def get_n_splits(self) -> int:
        """
        Number of splits used for this crossfit.
        """
        self._ensure_fitted()
        return len(self._model_by_split)

    def splits(self) -> Generator[Tuple[Sequence[int], Sequence[int]], None, None]:
        self._ensure_fitted()
        return self.cv.split(
            self._training_sample.features, self._training_sample.target
        )

    def models(self) -> Iterator[T_EstimatorDF]:
        """Iterator of all models fitted on the cross-validation train splits."""
        self._ensure_fitted()
        return iter(self._model_by_split)

    @property
    def training_sample(self) -> Sample:
        """The sample used to train this crossfit."""
        self._ensure_fitted()
        return self._training_sample

    def _ensure_fitted(self) -> None:
        if self._training_sample is None:
            raise RuntimeError(f"{type(self).__name__} expected to be fitted")

    # noinspection PyPep8Naming
    @staticmethod
    def _fit_model_for_split(
        estimator: T_EstimatorDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        **fit_params,
    ) -> T_EstimatorDF:
        """
        Fit a pipeline using a sample.

        :param estimator:  the :class:`gamma.ml.ModelPipelineDF` to fit
        :param train_sample: data used to fit the pipeline
        :return: fitted pipeline for the split
        """
        return estimator.fit(X=X, y=y, **fit_params)


class LearnerCrossfit(BaseCrossfit[T_LearnerDF], ABC, Generic[T_LearnerDF]):
    """
    Generate cross-validated prediction for each observation in a sample, based on
    multiple fits of a learner across a collection of cross-validation splits

    :param base_estimator: predictive pipeline to be fitted
    :param cv: the cross validator generating the train splits
    :param n_jobs: number of jobs to _rank_learners in parallel (default: 1)
    :param shared_memory: if ``True`` use threading in the parallel runs. If `False`, \
      use multiprocessing
    :param verbose: verbosity level used in the parallel computation
    """

    COL_SPLIT_ID = "split_id"
    COL_TARGET = "target"

    def __init__(
        self,
        base_estimator: T_LearnerDF,
        cv: BaseCrossValidator,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        super().__init__(
            base_estimator=base_estimator,
            cv=cv,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

    def _predictions_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.Series, pd.DataFrame], None, None]:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``COL_SPLIT_ID``),
        and with columns ``COL_PREDICTION` (the predicted value for the
        given observation and split), and ``COL_TARGET`` (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the crossfit per observation and test split
        """

        # todo: move this method to Simulator class -- too specific!

        for split_id, (model, (_, test_indices)) in enumerate(
            zip(self.models(), self.splits())
        ):
            test_features = sample.features.iloc[test_indices, :]
            yield model.predict(X=test_features)


class RegressorCrossfit(LearnerCrossfit[T_RegressorDF], Generic[T_RegressorDF]):
    pass


class ClassifierCrossfit(LearnerCrossfit[T_ClassifierDF], Generic[T_ClassifierDF]):
    __slots__ = ["_probabilities_for_all_samples", "_log_probabilities_for_all_samples"]

    COL_PROBA = "proba_class_0"

    __PROBA = "proba"
    __LOG_PROBA = "log_proba"
    __DECISION_FUNCTION = "decision_function"

    def _probabilities_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame]], None, None]:
        yield from self._classification_oob(
            sample=sample, method=lambda model, x: model.predict_proba(x)
        )

    def _log_probabilities_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame]], None, None]:
        yield from self._classification_oob(
            sample=sample, method=lambda model, x: model.predict_log_proba(x)
        )

    def _decision_function(
        self, sample: Sample
    ) -> Generator[Union[pd.Series, pd.DataFrame], None, None]:
        yield from self._classification_oob(
            sample=sample, method=lambda model, x: model._decision_function(x)
        )

    def _classification_oob(
        self,
        sample: Sample,
        method: Callable[
            [T_ClassifierDF, pd.DataFrame],
            Union[pd.DataFrame, List[pd.DataFrame], pd.Series],
        ],
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame], pd.Series], None, None]:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``COL_SPLIT_ID``),
        and with columns ``COL_PREDICTION` (the predicted value for the
        given observation and split), and ``COL_TARGET`` (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the crossfit per observation and test split
        """

        # todo: move this method to Simulator class -- too specific!

        for split_id, (model, (_, test_indices)) in enumerate(
            zip(self.models(), self.splits())
        ):
            test_features = sample.features.iloc[test_indices, :]
            yield method(model, test_features)
