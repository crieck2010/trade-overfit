import math

import pytest

from trade_overfit._stats import (
    excess_kurtosis,
    mean,
    median,
    normal_cdf,
    norm_ppf,
    sample_stdev,
    sample_variance,
    skewness,
)


def test_normal_cdf_known_values():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
    assert normal_cdf(3.0) == pytest.approx(0.99865, abs=1e-4)


def test_norm_ppf_known_values():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.975) == pytest.approx(1.96, abs=1e-4)
    assert norm_ppf(0.025) == pytest.approx(-1.96, abs=1e-4)


def test_norm_ppf_cdf_roundtrip():
    for p in (0.01, 0.1, 0.3, 0.7, 0.9, 0.99, 0.999):
        assert normal_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_rejects_bad_p():
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            norm_ppf(bad)


def test_skewness_symmetric_is_zero():
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert skewness(xs) == pytest.approx(0.0, abs=1e-12)


def test_skewness_needs_data():
    assert skewness([1.0, 2.0]) is None
    assert skewness([1.0, 1.0, 1.0, 1.0]) is None  # zero variance


def test_excess_kurtosis_needs_data():
    assert excess_kurtosis([1.0, 2.0, 3.0]) is None
    assert excess_kurtosis([1.0, 1.0, 1.0, 1.0]) is None


def test_median_and_variance():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert median([]) is None
    assert mean([]) is None
    assert sample_variance([1.0]) is None
    assert sample_variance([1.0, 3.0]) == pytest.approx(2.0)
    assert sample_stdev([1.0]) is None
