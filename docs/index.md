# Auto-sklearn Documentation

> A Python library for automated machine learning with meta-learning capabilities, hyperparameter optimization, and model stacking.

---

## Overview

Auto-sklearn provides a powerful framework for building automated machine learning pipelines. It combines hyperparameter optimization with advanced techniques like reparameterization, feature selection, and model stacking to achieve state-of-the-art results with minimal configuration.

```mermaid
graph TB
    subgraph "Auto-sklearn Pipeline"
        A[Raw Data] --> B[DataContext]
        B --> C[Feature Selection]
        C --> D[Model Graph]
        D --> E[Hyperparameter Tuning]
        E --> F[Cross-Validation]
        F --> G[Fitted Models]
        G --> H[Predictions]
    end

    subgraph "Key Components"
        I[SearchSpace] -.-> E
        J[Reparameterization] -.-> E
        K[Plugins] -.-> D
    end
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Model Graphs** | Define complex pipelines as directed acyclic graphs (DAGs) |
| **Hyperparameter Optimization** | Backend-agnostic search with Optuna integration |
| **Reparameterization** | Orthogonal parameter transformations for faster convergence |
| **Cross-Validation** | Stratified, grouped, nested, and time-series strategies |
| **Feature Selection** | Shadow feature-based selection with entropy matching |
| **Model Stacking** | Multi-layer stacking with out-of-fold predictions |
| **Joint Quantile Regression** | Model correlated targets with uncertainty quantification |
| **Plugin System** | Extensible plugins for model-specific behavior |
| **Caching** | Hash-based caching for expensive operations |

---

## Documentation

### Getting Started
- [Installation & Quickstart](getting-started.md) — Get up and running in minutes

### Core Concepts
- [Model Graphs](model-graphs.md) — Building ML pipelines as DAGs
- [Search Spaces](search-spaces.md) — Defining hyperparameter search spaces
- [Cross-Validation](cross-validation.md) — CV strategies and configuration

### Advanced Topics
- [Tuning & Optimization](tuning.md) — Hyperparameter optimization strategies
- [Reparameterization](reparameterization.md) — Meta-learning parameter transforms
- [Model Stacking](stacking.md) — Multi-layer ensemble methods
- [Feature Selection](feature-selection.md) — Automated feature selection
- [Plugins](plugins.md) — Extending functionality with plugins

### Reference
- [API Reference](api-reference.md) — Complete API documentation

---

## Architecture

```mermaid
graph LR
    subgraph "Core Layer"
        DC[DataContext]
        MG[ModelGraph]
        TO[TuningOrchestrator]
    end

    subgraph "Search Layer"
        SS[SearchSpace]
        OB[OptunaBackend]
        RP[Reparameterization]
    end

    subgraph "Execution Layer"
        EX[Executor]
        CA[Cache]
        AU[AuditLogger]
    end

    DC --> TO
    MG --> TO
    SS --> OB
    RP --> SS
    OB --> TO
    TO --> EX
    EX --> CA
    TO --> AU
```

---

## Quick Example

```python
from sklearn.ensemble import RandomForestClassifier
from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVStrategy
from auto_sklearn.core.model.node import ModelNode
from auto_sklearn.core.model.graph import ModelGraph
from auto_sklearn.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from auto_sklearn.search.space import SearchSpace

# Define search space
space = SearchSpace()
space.add_int("n_estimators", 50, 200)
space.add_int("max_depth", 3, 15)
space.add_float("min_samples_split", 0.01, 0.3)

# Create model node
node = ModelNode(
    name="random_forest",
    estimator_class=RandomForestClassifier,
    search_space=space,
    fixed_params={"random_state": 42, "n_jobs": -1},
)

# Build graph and tune
graph = ModelGraph()
graph.add_node(node)

config = TuningConfig(n_trials=50, metric="roc_auc")
orchestrator = TuningOrchestrator(graph, config)

ctx = DataContext(X=X_train, y=y_train)
fitted = orchestrator.fit(ctx)

# Predict
predictions = fitted.predict(X_test)
```

---

## License

MIT License — see [LICENSE](../LICENSE) for details.
