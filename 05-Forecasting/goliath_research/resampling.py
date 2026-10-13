"""Paired bootstrap and permutation helpers for the Forecasting lessons.

Bootstrap functions resample whole observations, so each x stays with its y.
Permutation functions shuffle one variable and leave the other fixed, which
builds a null distribution for no association.

The default is 4,000 resamples and seed 2026. A seed makes the tables in the
lessons repeatable. More resamples change the Monte Carlo error, not the
sample size of the original data.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SAMPLES = 4_000
DEFAULT_SEED = 2026


def _as_pair(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of observations")
    if x.size < 3:
        raise ValueError("at least 3 paired observations are required")
    return x, y


def pearson_r(x, y) -> float:
    """Pearson correlation of two paired variables. NaN if either is constant."""
    x, y = _as_pair(x, y)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _row_pearson(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Pearson r for each row of two (resamples, n) arrays."""
    x_centered = xs - xs.mean(axis=1, keepdims=True)
    y_centered = ys - ys.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=1) * np.sum(y_centered * y_centered, axis=1)
    )
    correlations = np.full(xs.shape[0], np.nan)
    np.divide(numerator, denominator, out=correlations, where=denominator > 0)
    return correlations


def bootstrap_pearson(x, y, num_samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Bootstrap distribution of Pearson's r, resampling paired rows."""
    x, y = _as_pair(x, y)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(num_samples, x.size))
    return _row_pearson(x[indices], y[indices])


def permutation_pearson(x, y, num_samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Null distribution of Pearson's r from random shuffles of y."""
    x, y = _as_pair(x, y)
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((num_samples, y.size)), axis=1)
    x_repeated = np.broadcast_to(x, (num_samples, x.size))
    return _row_pearson(x_repeated, y[order])


def permutation_p_value(observed: float, null_stats) -> float:
    """Two-sided permutation p-value with the add-one correction.

    The correction counts the observed statistic with the null statistics, so
    the result cannot be zero. With 4,000 shuffles the smallest possible value
    is 1/4001.
    """
    null_stats = np.asarray(null_stats, dtype=float)
    null_stats = null_stats[np.isfinite(null_stats)]
    if null_stats.size == 0 or not np.isfinite(observed):
        return float("nan")
    as_extreme = np.count_nonzero(np.abs(null_stats) >= abs(float(observed)))
    return float((as_extreme + 1) / (null_stats.size + 1))


def percentile_interval(samples, confidence: float = 95.0) -> tuple[float, float]:
    """Percentile interval from a bootstrap sample. Non-finite values are dropped."""
    if not 0 < confidence < 100:
        raise ValueError("confidence must be between 0 and 100")
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("no finite bootstrap statistics")
    tail = (100 - confidence) / 2
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def ols_slope_intercept(x, y) -> tuple[float, float]:
    """Ordinary least-squares slope and intercept for one predictor."""
    x, y = _as_pair(x, y)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered * x_centered))
    if denominator == 0:
        return float("nan"), float("nan")
    slope = float(np.sum(x_centered * (y - y.mean())) / denominator)
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def bootstrap_slope_intercept(
    x, y, num_samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap slopes and intercepts from paired resamples of a simple regression."""
    x, y = _as_pair(x, y)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(num_samples, x.size))
    x_sample = x[indices]
    y_sample = y[indices]
    x_mean = x_sample.mean(axis=1, keepdims=True)
    y_mean = y_sample.mean(axis=1, keepdims=True)
    x_centered = x_sample - x_mean
    denominator = np.sum(x_centered * x_centered, axis=1)
    numerator = np.sum(x_centered * (y_sample - y_mean), axis=1)
    slopes = np.full(num_samples, np.nan)
    np.divide(numerator, denominator, out=slopes, where=denominator > 0)
    intercepts = y_mean.ravel() - slopes * x_mean.ravel()
    return slopes, intercepts
