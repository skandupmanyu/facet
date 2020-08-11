"""
Factories for SHAP explainers from the ``shap`` package
"""
import functools
import logging
from abc import ABCMeta, abstractmethod
from distutils import version
from typing import *

import pandas as pd
import shap
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from pytools.api import AllTracker, inheritdoc, validate_type
from sklearndf import ClassifierDF, LearnerDF, RegressorDF

_EARLIEST_SUPPORTED_VERSION = version.LooseVersion("0.34")

log = logging.getLogger(__name__)

if version.LooseVersion(shap.__version__) < _EARLIEST_SUPPORTED_VERSION:
    raise RuntimeError(
        f"shap package version {shap.__version__} is not supported; "
        f"please upgrade to version {_EARLIEST_SUPPORTED_VERSION} or later"
    )

__all__ = ["ExplainerFactory", "TreeExplainerFactory", "KernelExplainerFactory"]
__tracker = AllTracker(globals())


class ExplainerFactory(metaclass=ABCMeta):
    """
    A factory for constructing :class:`~shap.Explainer` objects.
    """

    @property
    @abstractmethod
    def explains_raw_output(self) -> bool:
        """
        ``True`` if explainers made by this factory explain raw model output,
        ``False`` otherwise
        """

    @property
    @abstractmethod
    def supports_shap_interaction_values(self) -> bool:
        """
        ``True`` if explainers made by this factory allow for calculating SHAP
        interaction values, ``False`` otherwise.
        """

    @abstractmethod
    def make_explainer(self, model: LearnerDF, data: pd.DataFrame) -> Explainer:
        """
        Construct a new :class:`~shap.Explainer` to compute shap values.

        :param model: learner for which to compute shap values
        :param data: background dataset (optional)
        :return: the new explainer object
        """

    @staticmethod
    def _remove_null_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v is not None}


@inheritdoc(match="[see superclass]")
class TreeExplainerFactory(ExplainerFactory):
    """
    A factory constructing class:`~shap.TreeExplainer` objects.
    """

    def __init__(
        self,
        model_output: Optional[str] = None,
        feature_perturbation: Optional[str] = None,
        use_background_dataset: bool = True,
    ) -> None:
        """
        :param model_output: (optional) override the default model output parameter
        :param feature_perturbation: (optional) override the default \
            feature_perturbation parameter
        :param use_background_dataset: if ``False``, don't pass the background \
            dataset on to the tree explainer even if a background dataset is passed \
            to :meth:`.make_explainer`
        """
        super().__init__()
        validate_type(
            model_output, expected_type=str, optional=True, name="arg model_output"
        )
        validate_type(
            feature_perturbation,
            expected_type=str,
            optional=True,
            name="arg feature_perturbation",
        )
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.use_background_dataset = use_background_dataset

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return self.model_output in [None, "raw"]

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return self.feature_perturbation == "tree_path_dependent"

    def make_explainer(
        self, model: LearnerDF, data: Optional[pd.DataFrame] = None
    ) -> Explainer:
        """
        Construct a new :class:`~shap.TreeExplainer` to compute shap values.

        :param model: learner for which to compute shap values
        :param data: background dataset (optional)
        :return: the new explainer object
        """

        explainer = shap.TreeExplainer(
            model=model.native_estimator,
            data=data if self.use_background_dataset else None,
            **self._remove_null_kwargs(
                dict(
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                )
            ),
        )

        # set check_additivity=False; see github.gamma.bcg.com/BCG/gamma-ml/issues/68
        explainer.shap_values = functools.partial(
            explainer.shap_values, check_additivity=False
        )

        return explainer


# @inheritdoc(match="[see superclass]")
class KernelExplainerFactory(ExplainerFactory):
    """
    A factory constructing class:`~shap.KernelExplainer` objects.
    """

    def __init__(
        self,
        link: Optional[str] = None,
        l1_reg: Optional[str] = "num_features(10)",
        data_size_limit: Optional[int] = 100,
    ) -> None:
        """
        :param link: (optional) override the default link parameter
        :param l1_reg: (optional) override the default l1_reg parameter of method \
            :meth:`~shap.KernelExplainer.shap_values`; pass ``None`` to use the \
            default value used by :meth:`~shap.KernelExplainer.shap_values`
        :param data_size_limit: (optional) maximum number of observations to use as \
            the background data set; larger data sets will be down-sampled using \
            method :meth:`~shap.kmeans`. \
            Pass ``None`` to prevent down-sampling the background data set.
        """
        super().__init__()
        validate_type(link, expected_type=str, optional=True, name="arg link")
        self.link = link
        self.l1_reg = l1_reg if l1_reg is not None else "num_features(10)"
        self.data_size_limit = data_size_limit

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return self.link in [None, "identity"]

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return False

    def make_explainer(self, model: LearnerDF, data: pd.DataFrame) -> Explainer:
        """
        Construct a new :class:`~shap.KernelExplainer` to compute shap values.

        :param model: learner for which to compute shap values
        :param data: background dataset
        :return: the new explainer object
        """

        model_root_estimator: BaseEstimator = model.native_estimator

        try:
            if isinstance(model, RegressorDF):
                # noinspection PyUnresolvedReferences
                model = model_root_estimator.predict
            elif isinstance(model, ClassifierDF):
                # noinspection PyUnresolvedReferences
                model = model_root_estimator.predict_proba
            else:
                model = None
        except AttributeError as cause:
            raise TypeError(
                f"arg model does not support default prediction method: {cause}"
            ) from cause

        if not model:
            raise TypeError(
                "arg model is neither a regressor nor a classifier: "
                f"{type(model).__name__}"
            )

        data_size_limit = self.data_size_limit
        if data_size_limit is not None and len(data) > data_size_limit:
            data = shap.kmeans(data, data_size_limit, round_values=True)

        explainer = shap.KernelExplainer(
            model=model, data=data, **self._remove_null_kwargs(dict(link=self.link))
        )

        if self.l1_reg is not None:
            explainer.shap_values = functools.partial(
                explainer.shap_values, l1_reg=self.l1_reg
            )

        return explainer


__tracker.validate()
