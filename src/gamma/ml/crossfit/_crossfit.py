"""
Core implementation of :mod:`gamma.ml.crossfit`
"""
import logging
from abc import ABC
from typing import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from gamma.common.fit import FittableMixin
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.sklearndf import BaseLearnerDF, ClassifierDF, RegressorDF

log = logging.getLogger(__name__)

__all__ = ["LearnerCrossfit"]

T_Self = TypeVar("T", bound="LearnerCrossfit")
T_LearnerDF = TypeVar("T_LearnerDF", bound=BaseLearnerDF)
T_ClassifierDF = TypeVar("T_ClassifierDF", bound=ClassifierDF)
T_RegressorDF = TypeVar("T_RegressorDF", bound=RegressorDF)


class LearnerCrossfit(
    FittableMixin[Sample], ParallelizableMixin, ABC, Generic[T_LearnerDF]
):
    """
    Fits a learner to all train splits of a given cross-validation strategy.
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
        base_learner: T_LearnerDF,
        cv: BaseCrossValidator,
        *,
        shuffle_features: Optional[bool] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        f"""
        :param base_learner: predictive pipeline to be fitted
        :param cv: the cross validator generating the train splits
        :param shuffle_features: if `True`, shuffle column order of features for every \
            crossfit (default: `False`)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        {ParallelizableMixin.__init__.__doc__}
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.base_estimator = base_learner  #: the learner being trained
        self.cv = cv  #: the cross validator
        self.shuffle_features = False if shuffle_features is None else shuffle_features
        self.random_state = random_state

        self._model_by_split: Optional[List[T_LearnerDF]] = None
        self._training_sample: Optional[Sample] = None
        self._feature_order_by_split: Optional[List[np.ndarray]] = None

        if shuffle_features is None:
            log.warning(
                "default value of shuffle_features will change to False to True in "
                "release 1.2.0; we recommend that you set this parameter explicitly."
            )

    def fit(self: T_Self, sample: Sample, **fit_params) -> T_Self:
        """
        Fit the base estimator to the full sample, and fit a clone of the base
        estimator to each of the train splits generated by the cross-validator
        :param sample: the sample to fit the estimators to
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the base estimator
        :return: `self`
        """
        self_typed: LearnerCrossfit = self  # support better type hinting in PyCharm
        base_estimator = self_typed.base_estimator

        features: pd.DataFrame = sample.features
        feature_columns = features.columns
        n_features = len(feature_columns)
        target = sample.target
        shuffle_features = self_typed.shuffle_features
        if shuffle_features:
            # we are shuffling features, so we create an infinite iterator
            # that creates a new random permutation of feature indices on each
            # iteration
            random_state = check_random_state(self_typed.random_state)
            feature_sequence = iter(lambda: random_state.permutation(n_features), [])
        else:
            # we are not shuffling features, hence we create an infinite iterator of
            # always the same slice that preserves the existing feature sequence
            original_feature_sequence = slice(None)
            feature_sequence = iter(lambda: original_feature_sequence, slice(0))

        base_estimator.fit(X=features, y=target, **fit_params)

        with self_typed._parallel() as parallel:
            self._model_by_split: List[T_LearnerDF] = parallel(
                self_typed._delayed(LearnerCrossfit._fit_model_for_split)(
                    base_estimator.clone(),
                    features.iloc[train_indices, feature_sequence],
                    target.iloc[train_indices],
                    **fit_params,
                )
                for feature_sequence, (train_indices, _) in zip(
                    feature_sequence, self_typed.cv.split(features, target)
                )
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

    def splits(self) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        """
        :return: an iterator of all train/test splits used by this crossfit
        """
        self._ensure_fitted()
        return self.cv.split(
            X=self._training_sample.features, y=self._training_sample.target
        )

    def models(self) -> Iterator[T_LearnerDF]:
        """Iterator of all models fitted on the cross-validation train splits."""
        self._ensure_fitted()
        return iter(self._model_by_split)

    @property
    def training_sample(self) -> Sample:
        """The sample used to train this crossfit."""
        self._ensure_fitted()
        return self._training_sample

    # noinspection PyPep8Naming
    @staticmethod
    def _fit_model_for_split(
        estimator: T_LearnerDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        **fit_params,
    ) -> T_LearnerDF:
        """
        Fit a pipeline using a sample.

        :param estimator:  the :class:`gamma.ml.ModelPipelineDF` to fit
        :param train_sample: data used to fit the pipeline
        :return: fitted pipeline for the split
        """
        return estimator.fit(X=X, y=y, **fit_params)
