# API Reference

Complete API documentation for auto-sklearn.

---

## Core Module

### DataContext

```python
from auto_sklearn.core.data.context import DataContext
```

Immutable container for training data and metadata.

```python
DataContext(
    X: pd.DataFrame | np.ndarray,    # Features
    y: pd.Series | np.ndarray,       # Target
    groups: pd.Series | None = None, # Group labels for GroupKFold
    sample_weight: np.ndarray | None = None,
    upstream_predictions: dict | None = None,
    base_margin: np.ndarray | None = None,
)
```

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `X` | DataFrame | Feature matrix |
| `y` | Series | Target values |
| `groups` | Series | Group labels |
| `n_samples` | int | Number of samples |
| `n_features` | int | Number of features |
| `feature_columns` | list | Feature column names |

**Methods:**
```python
# Create subset with specific indices
ctx.with_subset(indices: np.ndarray) -> DataContext

# Add upstream model predictions
ctx.with_upstream(predictions: dict) -> DataContext
```

---

### CVConfig

```python
from auto_sklearn.core.data.cv import CVConfig, CVStrategy
```

Cross-validation configuration.

```python
CVConfig(
    n_splits: int = 5,
    strategy: CVStrategy = CVStrategy.STRATIFIED,
    n_repeats: int = 1,
    shuffle: bool = True,
    random_state: int | None = None,
)
```

**CVStrategy Enum:**
| Value | Description |
|-------|-------------|
| `STRATIFIED` | Preserve class ratios |
| `KFOLD` | Simple random splits |
| `GROUP` | Keep groups together |
| `TIME_SERIES` | Temporal ordering |

---

### DataManager

```python
from auto_sklearn.core.data.manager import DataManager
```

Manages CV fold creation and data routing.

```python
DataManager(cv_config: CVConfig)
```

**Methods:**
```python
# Create CV folds
create_folds(ctx: DataContext) -> list[CVFold]

# Get train/val data for a fold
align_to_fold(ctx: DataContext, fold: CVFold, split: str) -> DataContext

# Combine OOF predictions
route_oof_predictions(
    fold_predictions: dict[int, np.ndarray],
    n_samples: int
) -> np.ndarray
```

---

### ModelNode

```python
from auto_sklearn.core.model.node import ModelNode
```

Represents a single model in the graph.

```python
ModelNode(
    name: str,
    estimator_class: type,
    search_space: SearchSpace | None = None,
    fixed_params: dict | None = None,
    reparameterizations: list | None = None,
    plugins: list | None = None,
    include_original_features: bool = False,
)
```

**Methods:**
```python
# Create estimator instance with params
create_estimator(params: dict) -> estimator

# Get combined fixed + tuned params
get_params(tuned_params: dict) -> dict
```

---

### ModelGraph

```python
from auto_sklearn.core.model.graph import ModelGraph
```

Directed acyclic graph of model nodes.

```python
ModelGraph()
```

**Methods:**
```python
# Add node to graph
add_node(node: ModelNode) -> ModelGraph

# Add dependency between nodes
add_dependency(
    from_node: str,
    to_node: str,
    dependency: Dependency
) -> ModelGraph

# Get topologically sorted node names
topological_sort() -> list[str]

# Get nodes grouped by layer
get_layers() -> dict[int, list[str]]

# Validate graph structure
validate() -> None  # Raises ValueError if invalid

# Get node by name
get_node(name: str) -> ModelNode
```

---

### Dependency Types

```python
from auto_sklearn.core.model.dependency import (
    PredictionDependency,
    ProbaDependency,
    TransformDependency,
)
```

| Class | Description |
|-------|-------------|
| `PredictionDependency` | Pass class predictions |
| `ProbaDependency` | Pass probability predictions |
| `TransformDependency` | Pass transformed features |

---

### TuningConfig

```python
from auto_sklearn.core.tuning.orchestrator import TuningConfig
from auto_sklearn.core.tuning.strategy import OptimizationStrategy
```

```python
TuningConfig(
    strategy: OptimizationStrategy = OptimizationStrategy.OPTUNA,
    n_trials: int = 100,
    timeout: float | None = None,
    cv_config: CVConfig | None = None,
    metric: str = "accuracy",
    greater_is_better: bool = True,
    random_state: int | None = None,
    n_jobs: int = 1,
    patience: int | None = None,
)
```

**OptimizationStrategy Enum:**
| Value | Description |
|-------|-------------|
| `OPTUNA` | Bayesian optimization with TPE |
| `RANDOM` | Random search |
| `GRID` | Grid search |

---

### TuningOrchestrator

```python
from auto_sklearn.core.tuning.orchestrator import TuningOrchestrator
```

```python
TuningOrchestrator(
    graph: ModelGraph,
    data_manager: DataManager,
    tuning_config: TuningConfig,
    cache: FitCache | None = None,
    logger: AuditLogger | None = None,
)
```

**Methods:**
```python
# Run optimization
fit(ctx: DataContext) -> FittedGraph
```

---

## Search Module

### SearchSpace

```python
from auto_sklearn.search.space import SearchSpace
```

```python
SearchSpace()
```

**Methods:**
```python
# Add parameters (all return self for chaining)
add_float(name, low, high, log=False, step=None) -> SearchSpace
add_int(name, low, high, log=False, step=None) -> SearchSpace
add_categorical(name, choices) -> SearchSpace
add_parameter(param: Parameter) -> SearchSpace

# Conditional parameters
add_conditional(
    name: str,
    parent_name: str,
    parent_value: Any,
    parameter: Parameter
) -> SearchSpace

# Shorthand notation
add_from_shorthand(**kwargs) -> SearchSpace

# Operations
copy() -> SearchSpace
merge(other: SearchSpace) -> SearchSpace
remove_parameter(name: str) -> SearchSpace
get_parameter(name: str) -> Parameter | None

# Sampling
sample_optuna(trial) -> dict

# Properties
parameter_names: list[str]
```

**Class Methods:**
```python
SearchSpace.from_dict(config: dict) -> SearchSpace
```

---

### Parameter Classes

```python
from auto_sklearn.search.parameter import (
    FloatParameter,
    IntParameter,
    CategoricalParameter,
    ConditionalParameter,
)
```

```python
FloatParameter(name, low, high, log=False, step=None)
IntParameter(name, low, high, log=False, step=None)
CategoricalParameter(name, choices)
ConditionalParameter(name, parent_name, parent_value, parameter)
```

---

## Meta Module

### Reparameterizations

```python
from auto_sklearn.meta.reparameterization import (
    LogProductReparameterization,
    RatioReparameterization,
    LinearReparameterization,
)
```

```python
LogProductReparameterization(
    name: str,
    param1: str,
    param2: str,
)

RatioReparameterization(
    name: str,
    param1: str,
    param2: str,
)

LinearReparameterization(
    name: str,
    params: list[str],
    weights: list[float] | None = None,
)
```

**Methods (all reparameterizations):**
```python
forward(params: dict) -> dict   # Original → transformed
inverse(params: dict) -> dict   # Transformed → original
```

---

### Prebaked Configs

```python
from auto_sklearn.meta.prebaked import get_prebaked_reparameterizations

# Get recommended reparameterizations for a model
reparams = get_prebaked_reparameterizations(XGBClassifier)
```

---

## Selection Module

### ShadowFeatureSelector

```python
from auto_sklearn.selection.shadow import ShadowFeatureSelector
```

```python
ShadowFeatureSelector(
    estimator,                    # Model with feature_importances_
    n_iterations: int = 10,
    threshold_multiplier: float = 1.0,
    random_state: int | None = None,
)
```

**Methods:**
```python
fit(X, y) -> ShadowFeatureSelector
transform(X) -> np.ndarray
fit_transform(X, y) -> np.ndarray
get_selected_features() -> list[str]
get_selection_report() -> dict
```

**Properties:**
```python
feature_importances_: np.ndarray
shadow_max_: float
selected_mask_: np.ndarray
```

---

## Plugins Module

### ModelPlugin

```python
from auto_sklearn.plugins.base import ModelPlugin
```

Abstract base class for plugins.

```python
class ModelPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def applies_to(self, estimator_class) -> bool: ...
    def modify_search_space(self, space, context) -> SearchSpace: ...
    def modify_params(self, params, context) -> dict: ...
    def modify_fit_params(self, params, context) -> dict: ...
    def pre_fit(self, estimator, context) -> None: ...
    def post_fit(self, estimator, context) -> None: ...
    def post_tune(self, params, node, context) -> dict: ...
```

---

### CompositePlugin

```python
from auto_sklearn.plugins.base import CompositePlugin
```

```python
CompositePlugin(plugins: list[ModelPlugin])
```

---

### PluginRegistry

```python
from auto_sklearn.plugins.registry import PluginRegistry, get_global_registry
```

```python
registry = PluginRegistry()
registry.register(plugin, priority=0)
registry.unregister(plugin_name)
registry.get_plugins_for(estimator_class) -> list[ModelPlugin]

# Global registry
registry = get_global_registry()
```

---

### XGBoost Plugins

```python
from auto_sklearn.plugins.xgboost.multiplier import XGBMultiplierPlugin
from auto_sklearn.plugins.xgboost.importance import XGBImportancePlugin
```

```python
XGBMultiplierPlugin(
    multipliers: list[float] = [0.5, 1.0, 2.0],
    cv_folds: int = 3,
    enable_post_tune: bool = True,
)

XGBImportancePlugin(
    importance_type: str = "gain",  # "gain", "weight", "cover"
)
```

---

## Persistence Module

### FitCache

```python
from auto_sklearn.persistence.cache import FitCache
```

```python
FitCache(
    max_memory_mb: int = 500,
    disk_path: str | None = None,
    max_disk_mb: int | None = None,
)
```

**Methods:**
```python
get(key: str) -> Any | None
set(key: str, value: Any) -> None
clear() -> None
```

---

### ArtifactStore

```python
from auto_sklearn.persistence.store import ArtifactStore
```

```python
ArtifactStore(base_path: str)
```

**Methods:**
```python
save_model(name: str, model) -> str
load_model(name: str) -> Any
save_params(name: str, params: dict) -> str
load_params(name: str) -> dict
save_cv_result(name: str, result: CVResult) -> str
load_cv_result(name: str) -> CVResult
```

---

## Audit Module

### AuditLogger

```python
from auto_sklearn.audit.logger import AuditLogger
```

```python
AuditLogger(log_dir: str = "./logs")
```

**Methods:**
```python
log_trial(node_name: str, trial: TrialLog) -> None
log_fold(node_name: str, fold: FoldLog) -> None
get_trial_history(node_name: str) -> list[TrialLog]
get_best_params(node_name: str) -> dict
```

---

### Log Dataclasses

```python
from auto_sklearn.audit.logger import TrialLog, FoldLog

@dataclass
class TrialLog:
    trial_number: int
    params: dict
    score: float
    duration_seconds: float
    timestamp: datetime

@dataclass
class FoldLog:
    fold_index: int
    train_score: float
    val_score: float
    n_train: int
    n_val: int
```

---

## Execution Module

### Executor

```python
from auto_sklearn.execution.base import Executor
from auto_sklearn.execution.local import LocalExecutor, SequentialExecutor
```

```python
# Parallel execution
executor = LocalExecutor(n_jobs=-1)

# Sequential execution
executor = SequentialExecutor()
```

**Methods:**
```python
execute(tasks: list[Callable]) -> list[Any]
```

---

## Quick Import Reference

```python
# Core
from auto_sklearn.core.data.context import DataContext
from auto_sklearn.core.data.cv import CVConfig, CVStrategy
from auto_sklearn.core.data.manager import DataManager
from auto_sklearn.core.model.node import ModelNode
from auto_sklearn.core.model.graph import ModelGraph
from auto_sklearn.core.model.dependency import PredictionDependency, ProbaDependency
from auto_sklearn.core.tuning.orchestrator import TuningConfig, TuningOrchestrator
from auto_sklearn.core.tuning.strategy import OptimizationStrategy

# Search
from auto_sklearn.search.space import SearchSpace
from auto_sklearn.search.parameter import FloatParameter, IntParameter, CategoricalParameter

# Meta
from auto_sklearn.meta.reparameterization import (
    LogProductReparameterization,
    RatioReparameterization,
    LinearReparameterization,
)
from auto_sklearn.meta.prebaked import get_prebaked_reparameterizations

# Selection
from auto_sklearn.selection.shadow import ShadowFeatureSelector

# Plugins
from auto_sklearn.plugins.base import ModelPlugin, CompositePlugin
from auto_sklearn.plugins.registry import PluginRegistry, get_global_registry
from auto_sklearn.plugins.xgboost.multiplier import XGBMultiplierPlugin
from auto_sklearn.plugins.xgboost.importance import XGBImportancePlugin

# Persistence
from auto_sklearn.persistence.cache import FitCache
from auto_sklearn.persistence.store import ArtifactStore

# Audit
from auto_sklearn.audit.logger import AuditLogger, TrialLog, FoldLog

# Execution
from auto_sklearn.execution.local import LocalExecutor, SequentialExecutor
```
