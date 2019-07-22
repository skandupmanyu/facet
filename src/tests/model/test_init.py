import pytest
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from tests.model import make_simple_transformer
from yieldengine.model import ModelPipelineDF
from yieldengine.sklearndf.regression import LGBMRegressorDF


def test_model() -> None:

    model = ModelPipelineDF(
        predictor=LGBMRegressorDF(), preprocessing=make_simple_transformer()
    )

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ModelPipelineDF(predictor=LGBMRegressor(), preprocessing=OneHotEncoder())
