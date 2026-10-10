import warnings

import evaluate
import pytest


def _metric():
    return evaluate.load("metrics/r_squared")


def test_r_squared_matches_documented_example():
    score = _metric().compute(predictions=[1, 2, 3, 4], references=[0.9, 2.1, 3.2, 3.8])
    assert score == 0.98


def test_constant_references_do_not_return_nonfinite():
    metric = _metric()
    # Imperfect predictions of a constant target used to return -inf.
    assert metric.compute(predictions=[1, 2, 3, 4], references=[5, 5, 5, 5]) == 0.0
    # Exact predictions used to return nan from 0/0.
    assert metric.compute(predictions=[5, 5, 5, 5], references=[5, 5, 5, 5]) == 1.0


def test_zero_division_parameter_is_accepted():
    metric = _metric()
    assert metric.compute(predictions=[1, 2, 3, 4], references=[5, 5, 5, 5], zero_division=1) == 1.0
    assert metric.compute(predictions=[1, 2, 3, 4], references=[5, 5, 5, 5], zero_division=0) == 0.0
    with pytest.warns(RuntimeWarning, match="zero variance"):
        assert metric.compute(predictions=[1, 2, 3, 4], references=[5, 5, 5, 5], zero_division="warn") == 0.0
    # Exact fit stays 1.0 even when zero_division asks to substitute 0.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert metric.compute(predictions=[5, 5, 5, 5], references=[5, 5, 5, 5], zero_division="warn") == 1.0


def test_zero_division_rejects_unknown_values():
    with pytest.raises(ValueError, match="zero_division"):
        _metric().compute(predictions=[1, 2], references=[1, 2], zero_division="ignore")
