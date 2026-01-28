"""Knowledge distillation support for binary classification."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Tuple, Type

import numpy as np


@dataclass(frozen=True)
class DistillationConfig:
    """
    Configuration for knowledge distillation.

    Controls how a student node learns from a teacher's soft targets
    using a blended KL-divergence + cross-entropy loss.

    Attributes:
        temperature: Softens probability distributions before KL computation.
            Higher values produce softer distributions. Must be > 0.
        alpha: Blending weight between soft and hard losses.
            Loss = alpha * KL_soft + (1 - alpha) * CE_hard.
            Must be in [0, 1].
    """

    temperature: float = 3.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if not (0 <= self.alpha <= 1):
            raise ValueError(
                f"alpha must be in [0, 1], got {self.alpha}"
            )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def build_distillation_objective(
    soft_targets: np.ndarray,
    config: DistillationConfig,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Build a custom objective for knowledge distillation.

    Returns a closure compatible with XGBoost/LightGBM custom objectives.
    The loss blends KL-divergence (soft targets) with cross-entropy (hard labels).

    Args:
        soft_targets: 1D array of teacher's positive-class probabilities,
            clipped to [1e-7, 1-1e-7].
        config: Distillation configuration.

    Returns:
        Callable (y_true, y_pred) -> (gradient, hessian) where y_pred
        is the raw logit (before sigmoid).
    """
    T = config.temperature
    alpha = config.alpha

    # Clip and compute teacher logits once
    q_t_raw = np.clip(soft_targets, 1e-7, 1 - 1e-7)
    # Teacher logits at temperature T: z_t/T such that sigmoid(z_t/T) = q_t_raw
    # => z_t/T = log(q_t_raw / (1 - q_t_raw))
    teacher_logits_scaled = np.log(q_t_raw / (1.0 - q_t_raw))

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient and hessian for distillation loss."""
        # Student probabilities at temperature T
        q_s = _sigmoid(y_pred / T)
        # Teacher probabilities at temperature T (already computed from clipped targets)
        q_t = _sigmoid(teacher_logits_scaled)

        # Student probabilities at temperature 1 (for hard-label CE)
        p_s = _sigmoid(y_pred)

        # Gradient: alpha * T * (q_s - q_t) + (1 - alpha) * (p_s - y_true)
        grad = alpha * T * (q_s - q_t) + (1.0 - alpha) * (p_s - y_true)

        # Hessian: alpha * q_s*(1-q_s) + (1-alpha) * p_s*(1-p_s)
        hess = alpha * q_s * (1.0 - q_s) + (1.0 - alpha) * p_s * (1.0 - p_s)
        hess = np.maximum(hess, 1e-7)

        return grad, hess

    return objective


def validate_distillation_estimator(estimator_class: Type) -> None:
    """
    Validate that an estimator class supports custom objectives.

    Checks whether the estimator's constructor accepts an ``objective``
    parameter, which is required for injecting the distillation loss.

    Args:
        estimator_class: The estimator class to validate.

    Raises:
        ValueError: If the estimator does not support custom objectives.
    """
    sig = inspect.signature(estimator_class.__init__)
    if "objective" not in sig.parameters:
        raise ValueError(
            f"Estimator {estimator_class.__name__} does not support custom "
            f"objectives (no 'objective' parameter in __init__). "
            f"Distillation requires XGBoost, LightGBM, or similar."
        )
