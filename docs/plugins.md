# Plugins

The plugin system extends sklearn-meta with model-specific behavior, custom parameter transformations, and specialized fitting logic. Plugins hook into the model lifecycle without modifying core code.

---

## Plugin Architecture

```mermaid
graph TB
    subgraph "Plugin Lifecycle"
        A[Search Space] --> |modify_search_space| B[Modified Space]
        B --> |modify_params| C[Sampled Params]
        C --> |modify_fit_params| D[Fit Params]
        D --> |pre_fit| E[Pre-fit Hook]
        E --> F[Model.fit]
        F --> |post_fit| G[Post-fit Hook]
        G --> |post_tune| H[Final Params]
    end
```

---

## Creating a Plugin

### Basic Plugin

```python
from sklearn_meta.plugins.base import ModelPlugin

class MyPlugin(ModelPlugin):
    """Custom plugin for specific model behavior."""

    @property
    def name(self) -> str:
        return "my_plugin"

    def applies_to(self, estimator_class) -> bool:
        """Return True if this plugin applies to the estimator."""
        return estimator_class.__name__ == "MyCustomModel"

    def modify_params(self, params: dict, context) -> dict:
        """Modify parameters before fitting."""
        modified = params.copy()
        # Your custom logic here
        return modified
```

### Plugin Hooks

| Hook | When Called | Purpose |
|------|-------------|---------|
| `applies_to` | Registration | Check if plugin applies to model |
| `modify_search_space` | Before tuning | Add/modify search parameters |
| `modify_params` | Each trial | Transform sampled parameters |
| `modify_fit_params` | Before fit | Add fit-specific parameters |
| `pre_fit` | Before fit | Setup before fitting |
| `post_fit` | After fit | Cleanup after fitting |
| `post_tune` | After all trials | Final parameter selection |

---

## Built-in Plugins

### XGBMultiplierPlugin

Optimizes learning rate / n_estimators trade-off after tuning:

```python
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin

plugin = XGBMultiplierPlugin(
    multipliers=[0.5, 1.0, 2.0],  # Test these multipliers
    cv_folds=3,                   # CV for evaluation
    enable_post_tune=True,        # Enable post-tuning optimization
)
```

**How it works:**
1. After tuning, test different learning_rate × n_estimators combinations
2. Multiplier 0.5: halve learning_rate, double n_estimators
3. Multiplier 2.0: double learning_rate, halve n_estimators
4. Select best performing combination

```mermaid
graph LR
    subgraph "Multiplier Search"
        A[lr=0.1, n=100] --> B[0.5×: lr=0.05, n=200]
        A --> C[1.0×: lr=0.1, n=100]
        A --> D[2.0×: lr=0.2, n=50]
    end

    B --> E{Evaluate}
    C --> E
    D --> E
    E --> F[Select Best]
```

### XGBImportancePlugin

Extracts feature importance from XGBoost models:

```python
from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin

plugin = XGBImportancePlugin(importance_type="gain")

# After fitting
model = plugin.get_fitted_model()
importance = plugin.get_importance(model)
```

**Importance types:**
- `"gain"`: Average gain from splits using feature
- `"weight"`: Number of times feature is used
- `"cover"`: Average coverage of splits using feature

---

## Plugin Registry

### Registering Plugins

```python
from sklearn_meta.plugins.registry import PluginRegistry

registry = PluginRegistry()

# Register single plugin
registry.register(MyPlugin())

# Register with priority (higher = runs first)
registry.register(CriticalPlugin(), priority=100)
registry.register(OptionalPlugin(), priority=10)
```

### Getting Applicable Plugins

```python
from xgboost import XGBClassifier

# Get plugins that apply to XGBoost
plugins = registry.get_plugins_for(XGBClassifier)

for plugin in plugins:
    print(f"Plugin: {plugin.name}")
```

### Global Registry

```python
from sklearn_meta.plugins.registry import get_global_registry

registry = get_global_registry()
registry.register(MyPlugin())
```

---

## Composite Plugins

Combine multiple plugins:

```python
from sklearn_meta.plugins.base import CompositePlugin

composite = CompositePlugin([
    XGBMultiplierPlugin(),
    XGBImportancePlugin(),
    MyCustomPlugin(),
])

# All plugins execute in order
composite.modify_params(params, context)
```

---

## Example: Custom Early Stopping Plugin

```python
from sklearn_meta.plugins.base import ModelPlugin
import numpy as np

class EarlyStoppingPlugin(ModelPlugin):
    """Adds early stopping to compatible models."""

    def __init__(self, patience: int = 10, validation_fraction: float = 0.1):
        self.patience = patience
        self.validation_fraction = validation_fraction

    @property
    def name(self) -> str:
        return "early_stopping"

    def applies_to(self, estimator_class) -> bool:
        """Apply to models that support early stopping."""
        supported = ["XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"]
        return estimator_class.__name__ in supported

    def modify_fit_params(self, params: dict, context) -> dict:
        """Add early stopping parameters."""
        modified = params.copy()

        # Create validation set
        n_samples = len(context.X)
        n_val = int(n_samples * self.validation_fraction)
        indices = np.random.permutation(n_samples)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        modified["eval_set"] = [(context.X.iloc[val_indices], context.y.iloc[val_indices])]
        modified["early_stopping_rounds"] = self.patience
        modified["verbose"] = False

        # Store train indices for actual fitting
        context._train_indices = train_indices

        return modified

    def pre_fit(self, estimator, context) -> None:
        """Subset training data."""
        if hasattr(context, "_train_indices"):
            context._original_X = context.X
            context._original_y = context.y
            context.X = context.X.iloc[context._train_indices]
            context.y = context.y.iloc[context._train_indices]

    def post_fit(self, estimator, context) -> None:
        """Restore original data."""
        if hasattr(context, "_original_X"):
            context.X = context._original_X
            context.y = context._original_y
```

---

## Example: Parameter Constraint Plugin

```python
class ParameterConstraintPlugin(ModelPlugin):
    """Enforces parameter constraints."""

    def __init__(self, constraints: dict):
        """
        Args:
            constraints: Dict of {param: (min, max)} bounds
        """
        self.constraints = constraints

    @property
    def name(self) -> str:
        return "param_constraints"

    def applies_to(self, estimator_class) -> bool:
        return True  # Apply to all models

    def modify_params(self, params: dict, context) -> dict:
        """Clip parameters to constraints."""
        modified = params.copy()

        for param, (min_val, max_val) in self.constraints.items():
            if param in modified:
                modified[param] = max(min_val, min(max_val, modified[param]))

        return modified
```

Usage:
```python
plugin = ParameterConstraintPlugin({
    "max_depth": (1, 20),
    "n_estimators": (10, 1000),
})

registry.register(plugin)
```

---

## Example: Logging Plugin

```python
import logging
from datetime import datetime

class LoggingPlugin(ModelPlugin):
    """Logs model training events."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "logging"

    def applies_to(self, estimator_class) -> bool:
        return True

    def pre_fit(self, estimator, context) -> None:
        self.start_time = datetime.now()
        self.logger.info(f"Starting fit: {estimator.__class__.__name__}")
        self.logger.info(f"Data shape: {context.X.shape}")

    def post_fit(self, estimator, context) -> None:
        duration = datetime.now() - self.start_time
        self.logger.info(f"Fit completed in {duration.total_seconds():.2f}s")

    def post_tune(self, params, node, context) -> dict:
        self.logger.info(f"Best params for {node.name}: {params}")
        return params
```

---

## Using Plugins with Nodes

### Attach to Specific Node

```python
from sklearn_meta.core.model.node import ModelNode

node = ModelNode(
    name="xgb",
    estimator_class=XGBClassifier,
    search_space=space,
    plugins=[XGBMultiplierPlugin(), LoggingPlugin()],
)
```

### Global Plugin Application

```python
from sklearn_meta.plugins.registry import get_global_registry

# Register globally
registry = get_global_registry()
registry.register(EarlyStoppingPlugin())
registry.register(LoggingPlugin())

# All XGBoost nodes will use these plugins
```

---

## Plugin Execution Order

Plugins execute in priority order (higher priority first):

```python
registry.register(CriticalPlugin(), priority=100)   # Runs first
registry.register(ImportantPlugin(), priority=50)   # Runs second
registry.register(DefaultPlugin(), priority=0)      # Runs last
```

Within same priority, registration order is preserved.

---

## Complete Example

```python
from sklearn.datasets import make_classification
import pandas as pd
import xgboost as xgb

from sklearn_meta.core.data.context import DataContext
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
from sklearn_meta.core.data.manager import DataManager
from sklearn_meta.core.model.node import ModelNode
from sklearn_meta.core.model.graph import ModelGraph
from sklearn_meta.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin

# Data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
ctx = DataContext(X=pd.DataFrame(X), y=pd.Series(y))

# Search space
space = (
    SearchSpace()
    .add_float("learning_rate", 0.01, 0.3, log=True)
    .add_int("n_estimators", 50, 300)
    .add_int("max_depth", 3, 10)
)

# Plugin for learning rate optimization
multiplier_plugin = XGBMultiplierPlugin(
    multipliers=[0.5, 1.0, 2.0],
    enable_post_tune=True,
)

# Model with plugin
node = ModelNode(
    name="xgb",
    estimator_class=xgb.XGBClassifier,
    search_space=space,
    fixed_params={"random_state": 42, "eval_metric": "logloss"},
    plugins=[multiplier_plugin],
)

# Graph and tuning
graph = ModelGraph()
graph.add_node(node)

cv_config = CVConfig(n_splits=5, strategy=CVStrategy.STRATIFIED)
tuning_config = TuningConfig(n_trials=20, cv_config=cv_config, metric="roc_auc")

orchestrator = TuningOrchestrator(graph, DataManager(cv_config), tuning_config)

print("Tuning with XGBMultiplierPlugin...")
fitted = orchestrator.fit(ctx)

print(f"\nBest params: {fitted.best_params['xgb']}")
# Plugin may have adjusted learning_rate/n_estimators after tuning
```

---

## Best Practices

### 1. Keep Plugins Focused

```python
# Good: Single responsibility
class EarlyStoppingPlugin(ModelPlugin): ...
class ImportancePlugin(ModelPlugin): ...

# Avoid: Kitchen sink plugin
class DoEverythingPlugin(ModelPlugin): ...
```

### 2. Don't Mutate Input

```python
def modify_params(self, params, context):
    # Good: Copy first
    modified = params.copy()
    modified["new_param"] = value
    return modified

    # Bad: Mutate in place
    params["new_param"] = value  # Don't do this!
    return params
```

### 3. Use Priority Wisely

```python
# Critical setup (e.g., data validation): priority 100
# Parameter modification: priority 50
# Logging/monitoring: priority 0
```

### 4. Handle Missing Models Gracefully

```python
def applies_to(self, estimator_class):
    try:
        return estimator_class.__name__.startswith("XGB")
    except Exception:
        return False
```

---

## Next Steps

- [Model Graphs](model-graphs.md) — Integrate plugins with nodes
- [Tuning](tuning.md) — Plugin hooks during optimization
- [Feature Selection](feature-selection.md) — XGBoost importance extraction
