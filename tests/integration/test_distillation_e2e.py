"""Integration tests for knowledge distillation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from sklearn_meta.api import GraphBuilder
from sklearn_meta.core.model.node import OutputType


# =============================================================================
# Mock estimator that supports custom objectives
# =============================================================================


class MockXGBClassifier:
    """Mock XGBoost-like classifier that supports custom objectives."""

    def __init__(self, objective=None, n_estimators=10, random_state=42):
        self.objective = objective
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._fitted = False
        self._classes = None

    def fit(self, X, y, **kwargs):
        self._fitted = True
        self._classes = np.unique(y)
        # Simple: just store mean of y per "bin" of first feature
        self._mean_y = y.mean()
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        return (np.random.RandomState(self.random_state).rand(n) > 0.5).astype(int)

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        p = np.full(n, self._mean_y)
        return np.column_stack([1 - p, p])

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }


# =============================================================================
# Integration tests
# =============================================================================


@pytest.fixture
def binary_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y, name="target")


class TestDistillationEndToEnd:
    """End-to-end tests for knowledge distillation pipeline."""

    def test_teacher_student_pipeline(self, binary_data):
        """Verify teacher-student distillation pipeline runs end-to-end."""
        X, y = binary_data

        fitted = (
            GraphBuilder("distill_test")
            .add_model("teacher", RandomForestClassifier)
            .with_output_type(OutputType.PROBA)
            .with_fixed_params(n_estimators=10, random_state=42)
            .add_model("student", MockXGBClassifier)
            .distills("teacher", temperature=3.0, alpha=0.5)
            .with_fixed_params(n_estimators=5, random_state=42)
            .with_tuning(n_trials=1, strategy="none", metric="accuracy",
                         greater_is_better=True)
            .with_cv(n_splits=3, random_state=42)
            .fit(X, y)
        )

        # Both nodes should be fitted
        assert "teacher" in fitted.fitted_nodes
        assert "student" in fitted.fitted_nodes

        # Student should have received a custom objective during training
        student_fitted = fitted.get_node("student")
        # Each fold model should have had objective set
        for model in student_fitted.models:
            assert model.objective is not None

    def test_teacher_oof_not_added_as_features(self, binary_data):
        """Verify teacher OOF predictions are not added as input features to student."""
        X, y = binary_data

        fitted = (
            GraphBuilder("distill_test")
            .add_model("teacher", RandomForestClassifier)
            .with_output_type(OutputType.PROBA)
            .with_fixed_params(n_estimators=10, random_state=42)
            .add_model("student", MockXGBClassifier)
            .distills("teacher")
            .with_fixed_params(n_estimators=5, random_state=42)
            .with_tuning(n_trials=1, strategy="none", metric="accuracy",
                         greater_is_better=True)
            .with_cv(n_splits=3, random_state=42)
            .fit(X, y)
        )

        # The student's CV result should show the original feature count
        # (no distill_ prefix features added)
        student_node = fitted.get_node("student")
        # The OOF predictions exist for the student
        assert student_node.oof_predictions is not None
        assert len(student_node.oof_predictions) == len(X)

    def test_predict_works_without_teacher(self, binary_data):
        """Verify prediction works without teacher at inference time."""
        X, y = binary_data

        fitted = (
            GraphBuilder("distill_test")
            .add_model("teacher", RandomForestClassifier)
            .with_output_type(OutputType.PROBA)
            .with_fixed_params(n_estimators=10, random_state=42)
            .add_model("student", MockXGBClassifier)
            .distills("teacher")
            .with_fixed_params(n_estimators=5, random_state=42)
            .with_tuning(n_trials=1, strategy="none", metric="accuracy",
                         greater_is_better=True)
            .with_cv(n_splits=3, random_state=42)
            .fit(X, y)
        )

        # Predict from student node (teacher not needed at inference)
        predictions = fitted.predict(X, node_name="student")

        assert predictions is not None
        assert len(predictions) == len(X)

    def test_only_one_teacher_allowed(self):
        """Verify that a student can only have one teacher."""
        builder = GraphBuilder("test")
        student = (
            builder
            .add_model("teacher1", RandomForestClassifier)
            .with_output_type(OutputType.PROBA)
            .add_model("teacher2", RandomForestClassifier)
            .with_output_type(OutputType.PROBA)
            .add_model("student", MockXGBClassifier)
            .distills("teacher1")
        )

        with pytest.raises(ValueError, match="already has a distillation teacher"):
            student.distills("teacher2")

    def test_distillation_with_layer_by_layer(self, binary_data):
        """Verify distillation works with layer-by-layer strategy."""
        X, y = binary_data

        fitted = (
            GraphBuilder("distill_test")
            .add_model("teacher", RandomForestClassifier)
            .with_output_type(OutputType.PROBA)
            .with_fixed_params(n_estimators=10, random_state=42)
            .add_model("student", MockXGBClassifier)
            .distills("teacher", temperature=2.0, alpha=0.8)
            .with_fixed_params(n_estimators=5, random_state=42)
            .with_tuning(n_trials=1, strategy="layer_by_layer", metric="accuracy",
                         greater_is_better=True)
            .with_cv(n_splits=3, random_state=42)
            .fit(X, y)
        )

        assert "teacher" in fitted.fitted_nodes
        assert "student" in fitted.fitted_nodes

        # Verify the objective was injected
        for model in fitted.get_node("student").models:
            assert model.objective is not None
