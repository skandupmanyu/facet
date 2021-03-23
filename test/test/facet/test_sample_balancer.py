""" Test facet.data.SampleBalancer"""
import logging

import pytest

from facet.data import Sample, SampleBalancer
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def set_numpy_seed() -> None:
    np.random.seed(42)
    yield


@pytest.fixture(scope="module")
def binary_target() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c1": np.floor(np.random.random(1000) * 1000),
            "c2": np.floor(np.random.random(1000) * 1000),
            "c3": np.floor(np.random.random(1000) * 1000),
            "target": ([0] * 900) + ([1] * 100),
        }
    )


@pytest.fixture()
def continuous_target() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c1": np.floor(np.random.random(110) * 1000),
            "c2": np.floor(np.random.random(110) * 1000),
            "c3": np.floor(np.random.random(110) * 1000),
            "target": (
                [0.0] * 100 + [1.15, 1.1, 1.2, 1.3, 1.1, 1.1, 5.1, 5.2, 5.2, 4.9]
            ),
        }
    )


@pytest.fixture(scope="module")
def multiclass_target() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c1": np.floor(np.random.random(1000) * 1000),
            "c2": np.floor(np.random.random(1000) * 1000),
            "c3": np.floor(np.random.random(1000) * 1000),
            "target": (["a"] * 700) + (["b"] * 200) + (["c"] * 100),
        }
    )


def test_argument_validation(multiclass_target: pd.DataFrame,) -> None:

    faulty_args = (
        {"target_class_ratio": 1.01, "bins": 10},  # ratio too high
        {"target_class_ratio": 0.0, "bins": 10},  # ratio too low
        {"target_class_ratio": -0.4, "bins": 10},  # ratio negative
        {
            "target_class_ratio": {"x": 0.1},
            "bins": 10,
        },  # for dict, requires bins='labels'
        {"target_class_ratio": 0.5, "bins": 0},  # bins too low
        {"target_class_ratio": 0.5, "bins": 1},  # bins too low
        {"target_class_ratio": 0.5, "bins": 1.0},  # bins faulty type
    )

    for kwargs in faulty_args:
        with pytest.raises(expected_exception=ValueError):
            SampleBalancer(**kwargs)

    # test validation of passed label dictionary
    with pytest.raises(
        expected_exception=ValueError,
        match="Keys in 'target_class_ratio' dict do not "
        "match with minority class labels",
    ):
        s = Sample(observations=multiclass_target, target_name="target")
        SampleBalancer(
            bins="labels", target_class_ratio={"a": 0.2, "b": 0.4, "faulty": 10}
        ).balance(s)


def test_undersample_with_binary_labels(binary_target) -> None:
    test_sample = Sample(observations=binary_target, target_name="target")

    test_balancer = SampleBalancer(
        target_class_ratio=0.4, bins="labels", undersample=True
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    assert value_counts[1] / value_counts[0] == pytest.approx(0.4, abs=0.02)

    log.info(value_counts)


def test_oversample_with_binary_labels(binary_target) -> None:

    test_sample = Sample(observations=binary_target, target_name="target")

    test_balancer = SampleBalancer(
        target_class_ratio=0.4, bins="labels", undersample=False
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    assert value_counts[1] / value_counts[0] == pytest.approx(0.4, abs=0.02)

    log.info(value_counts)


def test_undersample_with_multilabel(multiclass_target) -> None:
    test_sample = Sample(observations=multiclass_target, target_name="target")

    test_balancer = SampleBalancer(
        target_class_ratio=0.4, bins="labels", undersample=True
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()

    # most frequent minority class will reach 0.8 freq. of majority:
    assert value_counts[1] / value_counts[0] == pytest.approx(0.8, abs=0.05)

    # least frequent minority class will reach 0.4 freq. of majority:
    assert value_counts[2] / value_counts[0] == pytest.approx(0.4, abs=0.05)
    log.info(value_counts)


def test_oversample_with_multilabel(multiclass_target) -> None:

    test_sample = Sample(observations=multiclass_target, target_name="target")

    test_balancer = SampleBalancer(
        target_class_ratio=0.4, bins="labels", undersample=False
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    assert value_counts[1] / value_counts[0] == pytest.approx(0.4, abs=0.02)

    log.info(value_counts)


def test_undersample_with_multilabel_and_ratios(multiclass_target) -> None:
    test_sample = Sample(observations=multiclass_target, target_name="target")

    test_balancer = SampleBalancer(
        target_class_ratio={"b": 0.5, "c": 0.2}, bins="labels", undersample=True
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()

    assert value_counts[1] / value_counts[0] == pytest.approx(0.5, abs=0.05)
    assert value_counts[2] / value_counts[0] == pytest.approx(0.2, abs=0.06)
    log.info(value_counts)


def test_oversample_with_multilabel_and_ratios(multiclass_target) -> None:

    test_sample = Sample(observations=multiclass_target, target_name="target")

    test_balancer = SampleBalancer(
        target_class_ratio={"b": 0.5, "c": 0.2}, bins="labels", undersample=False
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    assert value_counts[1] / value_counts[0] == pytest.approx(0.5, abs=0.02)
    assert value_counts[2] / value_counts[0] == pytest.approx(0.2, abs=0.02)

    log.info(value_counts)


def test_undersample_with_continuous_target(continuous_target) -> None:

    test_sample = Sample(observations=continuous_target, target_name="target")

    test_balancer = SampleBalancer(target_class_ratio=0.4, bins=3, undersample=True)

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()

    # target 0.0 is in majority (10X frequency) ahead of balancing –
    #  to verify results, manually partition and count:
    count_zero = value_counts[0.0].sum()
    count_to_4 = value_counts[
        (0 < value_counts.index) & (4.5 < value_counts.index)
    ].sum()
    count_from_4 = value_counts[(value_counts.index >= 4.5)].sum()
    assert count_to_4 / count_zero == pytest.approx(0.5, abs=0.1)
    assert count_from_4 / count_zero == pytest.approx(0.5, abs=0.1)
    log.info(value_counts)


def test_upsample_with_continuous_target(continuous_target: pd.DataFrame) -> None:

    test_sample = Sample(observations=continuous_target, target_name="target")

    test_balancer = SampleBalancer(target_class_ratio=0.4, bins=3, undersample=False)

    balanced = test_balancer.balance(test_sample)

    value_counts = balanced.target.value_counts()

    # target 0.0 is in majority ahead of balancing –
    #  to verify results, manually partition and count:
    count_zero = value_counts[0.0].sum()
    count_to_4 = value_counts[
        (0 < value_counts.index) & (4.5 < value_counts.index)
    ].sum()
    count_from_4 = value_counts[(value_counts.index >= 4.5)].sum()

    assert count_to_4 / count_zero == pytest.approx(0.4, abs=0.02)
    assert count_from_4 / count_zero == pytest.approx(0.4, abs=0.02)
    log.info(value_counts)


def test_no_change(binary_target) -> None:
    test_sample = Sample(observations=binary_target, target_name="target")

    test_args = (
        {"target_class_ratio": 0.05, "bins": "labels",},
        {"target_class_ratio": {1: 0.1111}, "bins": "labels",},
    )

    for kwargs in test_args:
        for undersample in (True, False):
            # the ratio of minority to majority is already at 11%, the following
            # should not do anything:
            test_balancer = SampleBalancer(**kwargs, undersample=undersample)

            balanced = test_balancer.balance(test_sample)
            assert test_sample == balanced
            value_counts = balanced.target.value_counts()
            assert value_counts[1] / value_counts[0] == pytest.approx(0.11, abs=0.01)

            log.info(value_counts)
