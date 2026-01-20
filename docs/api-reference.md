# API Reference

Complete API documentation for sklearn-meta.

---

## GraphBuilder (Fluent API)

```python
from sklearn_meta import GraphBuilder
```

A fluent API for building model graphs with minimal boilerplate. Provides a chainable interface for defining models, search spaces, dependencies, and tuning configuration.

```python
GraphBuilder(name: str = "pipeline")
```

**Methods:**
```python
# Add a model to the graph
add_model(name: str, estimator_class: type) -> NodeBuilder

# Configure cross-validation
with_cv(
    n_splits: int = 5,
    n_repeats: int = 1,
    strategy: str | CVStrategy = "stratified",
    shuffle: bool = True,
    random_state: int = 42,
    nested: bool = False,
    inner_splits: int = 3,
) -> GraphBuilder

# Configure hyperparameter tuning
with_tuning(
    n_trials: int = 100,
    timeout: float | None = None,
    strategy: str | OptimizationStrategy = "layer_by_layer",
    metric: str = "neg_mean_squared_error",
    greater_is_better: bool = False,
    early_stopping_rounds: int | None = None,
) -> GraphBuilder

# Configure feature selection
with_feature_selection(
    method: str = "shadow",
    n_shadows: int = 5,
    threshold_mult: float = 1.414,
    retune_after_pruning: bool = True,
    min_features: int = 1,
    max_features: int | None = None,
) -> GraphBuilder

# Configure reparameterization
with_reparameterization(
    reparameterizations: list[Reparameterization] | None = None,
    use_prebaked: bool = True,
) -> GraphBuilder

# Build the ModelGraph
build() -> ModelGraph

# Create TuningOrchestrator
create_orchestrator(search_backend: SearchBackend | None = None) -> TuningOrchestrator

# Build and fit in one step
fit(X, y, groups=None, search_backend=None) -> FittedGraph
```

### NodeBuilder

Returned by `add_model()` for configuring individual nodes.

```python
# Set search space (shorthand syntax)
with_search_space(
    space: SearchSpace | None = None,
    **kwargs,  # e.g., n_estimators=(50, 500), max_depth=(3, 20)
) -> NodeBuilder

# Set output type
with_output_type(output_type: str) -> NodeBuilder  # "prediction", "proba", "transform"

# Set execution condition
with_condition(condition: Callable[[DataContext], bool]) -> NodeBuilder

# Add plugins
with_plugins(*plugins: str) -> NodeBuilder

# Set fixed (non-tuned) parameters
with_fixed_params(**params) -> NodeBuilder

# Set fit parameters
with_fit_params(**params) -> NodeBuilder

# Specify features to use
with_features(*feature_cols: str) -> NodeBuilder

# Add description
with_description(description: str) -> NodeBuilder

# Add dependencies
depends_on(*sources: str, dep_type: DependencyType = DependencyType.PREDICTION) -> NodeBuilder
stacks(*sources: str) -> NodeBuilder           # Shortcut for prediction dependencies
stacks_proba(*sources: str) -> NodeBuilder     # Shortcut for probability dependencies
```

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_meta import GraphBuilder

# Build a stacking pipeline with fluent API
fitted = (
    GraphBuilder("stacking_pipeline")
    .add_model("rf", RandomForestClassifier)
    .with_search_space(
        n_estimators=(50, 300),
        max_depth=(3, 15),
    )
    .with_fixed_params(random_state=42, n_jobs=-1)
    .add_model("gbm", GradientBoostingClassifier)
    .with_search_space(
        n_estimators=(50, 200),
        learning_rate=(0.01, 0.3, "log"),
        max_depth=(3, 8),
    )
    .add_model("meta", LogisticRegression)
    .stacks("rf", "gbm")
    .with_cv(n_splits=5, strategy="stratified")
    .with_tuning(n_trials=50, metric="roc_auc", greater_is_better=True)
    .fit(X_train, y_train)
)

predictions = fitted.predict(X_test)
```

---

## Core Module

### DataContext

```python
from sklearn_meta.core.data.context import DataContext
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
from sklearn_meta.core.data.cv import CVConfig, CVStrategy
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
from sklearn_meta.core.data.manager import DataManager
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
from sklearn_meta.core.model.node import ModelNode
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
from sklearn_meta.core.model.graph import ModelGraph
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
from sklearn_meta.core.model.dependency import (
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
from sklearn_meta.core.tuning.orchestrator import TuningConfig
from sklearn_meta.core.tuning.strategy import OptimizationStrategy
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
from sklearn_meta.core.tuning.orchestrator import TuningOrchestrator
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
from sklearn_meta.search.space import SearchSpace
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
from sklearn_meta.search.parameter import (
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

### CorrelationAnalyzer

```python
from sklearn_meta.meta.correlation import CorrelationAnalyzer, HyperparameterCorrelation, CorrelationType
```

Analyzes optimization history to discover hyperparameter correlations. This helps identify parameters that provide similar effects, have tradeoff relationships, or should be tuned together.

```python
CorrelationAnalyzer(
    min_trials: int = 20,              # Minimum trials for reliable analysis
    significance_threshold: float = 0.1,  # P-value threshold
    correlation_threshold: float = 0.3,   # Minimum correlation to report
)
```

**Methods:**
```python
# Analyze optimization results for correlations
analyze(
    optimization_result: OptimizationResult,
    param_names: list[str] | None = None,
) -> list[HyperparameterCorrelation]

# Suggest reparameterizations based on discovered correlations
suggest_reparameterization(
    correlations: list[HyperparameterCorrelation],
) -> dict  # {"substitutes": [...], "tradeoffs": [...], "correlation_details": [...]}
```

**CorrelationType Enum:**
| Value | Description |
|-------|-------------|
| `SUBSTITUTE` | Parameters providing similar effects (e.g., L1 and L2 regularization) |
| `COMPLEMENT` | Parameters that work together and move in the same direction |
| `TRADEOFF` | Parameters with inverse relationship (e.g., learning_rate × n_estimators) |
| `CONDITIONAL` | One parameter's optimal value depends on another |

**HyperparameterCorrelation:**
```python
@dataclass
class HyperparameterCorrelation:
    params: list[str]                    # Correlated parameter names
    correlation_type: CorrelationType    # Type of correlation
    strength: float                      # Correlation strength (0 to 1)
    functional_form: str                 # Description of relationship
    transform: Callable | None           # Transform to effective value
    inverse_transform: Callable | None   # Recover original params
    confidence: float                    # Confidence based on sample size

    # Methods
    effective_value(param_values: dict[str, float]) -> float
    decompose(effective: float, ratio: float = 0.5) -> dict[str, float]
```

**Example:**
```python
from sklearn_meta.meta.correlation import CorrelationAnalyzer

# After running optimization
analyzer = CorrelationAnalyzer(min_trials=30)
correlations = analyzer.analyze(optimization_result)

for corr in correlations:
    print(f"{corr.params}: {corr.correlation_type.value} (strength={corr.strength:.2f})")
    print(f"  Relationship: {corr.functional_form}")

# Get suggested reparameterizations
suggestions = analyzer.suggest_reparameterization(correlations)
print(f"Tradeoffs found: {suggestions['tradeoffs']}")
```

---

### Reparameterizations

```python
from sklearn_meta.meta.reparameterization import (
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
from sklearn_meta.meta.prebaked import get_prebaked_reparameterizations

# Get recommended reparameterizations for a model
reparams = get_prebaked_reparameterizations(XGBClassifier)
```

---

## Selection Module

### ShadowFeatureSelector

```python
from sklearn_meta.selection.shadow import ShadowFeatureSelector
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
from sklearn_meta.plugins.base import ModelPlugin
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
from sklearn_meta.plugins.base import CompositePlugin
```

```python
CompositePlugin(plugins: list[ModelPlugin])
```

---

### PluginRegistry

```python
from sklearn_meta.plugins.registry import PluginRegistry, get_global_registry
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
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin
from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin
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
from sklearn_meta.persistence.cache import FitCache
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
from sklearn_meta.persistence.store import ArtifactStore, ArtifactMetadata
```

File-based storage for models, parameters, and fitted graphs with metadata tracking.

```python
ArtifactStore(base_path: str = ".sklearn_meta_artifacts")
```

**Methods:**
```python
# Save a fitted model
save_model(
    model: Any,
    node_name: str,
    fold_idx: int = 0,
    params: dict | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> str  # Returns artifact_id

# Load a saved model
load_model(artifact_id: str) -> Any

# Save all models from a fitted node
save_fitted_node(
    fitted_node: FittedNode,
    tags: dict[str, str] | None = None,
) -> list[str]  # Returns list of artifact IDs

# Save an entire fitted graph
save_fitted_graph(
    fitted_graph: FittedGraph,
    name: str,
    tags: dict[str, str] | None = None,
) -> str  # Returns graph artifact ID

# Save hyperparameters
save_params(
    params: dict,
    node_name: str,
    description: str = "",
) -> str

# Load saved parameters
load_params(artifact_id: str) -> dict

# List stored artifacts
list_artifacts(
    artifact_type: str | None = None,  # "model", "params", "graph"
    node_name: str | None = None,
) -> list[ArtifactMetadata]

# Delete an artifact
delete_artifact(artifact_id: str) -> bool
```

**ArtifactMetadata:**
```python
@dataclass
class ArtifactMetadata:
    artifact_id: str
    artifact_type: str
    created_at: str
    node_name: str | None
    params: dict
    metrics: dict[str, float]
    tags: dict[str, str]
```

**Example:**
```python
from sklearn_meta.persistence.store import ArtifactStore

# Save a fitted graph
store = ArtifactStore("./my_artifacts")
graph_id = store.save_fitted_graph(
    fitted_graph,
    name="production_model",
    tags={"version": "1.0", "dataset": "train_2024"}
)

# List all model artifacts
models = store.list_artifacts(artifact_type="model")
for m in models:
    print(f"{m.node_name}: {m.metrics}")

# Load a specific model
model = store.load_model(models[0].artifact_id)
```

---

## Audit Module

### AuditLogger

```python
from sklearn_meta.audit.logger import AuditLogger
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
from sklearn_meta.audit.logger import TrialLog, FoldLog

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
from sklearn_meta.execution.base import Executor
from sklearn_meta.execution.local import LocalExecutor, SequentialExecutor
```

Abstract base class for execution backends. Executors handle parallel or distributed execution of tasks, allowing easy swapping between local, multiprocessing, or distributed backends.

**Base Executor Interface:**
```python
class Executor(ABC):
    # Apply function to list of items
    map(fn: Callable[[T], R], items: list[T]) -> list[R]

    # Submit for async execution
    submit(fn: Callable[..., R], *args, **kwargs) -> Future[R]

    # Shutdown executor
    shutdown(wait: bool = True) -> None

    # Properties
    n_workers: int          # Number of workers available
    is_distributed() -> bool  # Whether runs on multiple machines
```

**LocalExecutor:**
```python
LocalExecutor(n_jobs: int = -1)  # -1 means use all CPU cores
```

Parallel execution using Python's concurrent.futures.

**SequentialExecutor:**
```python
SequentialExecutor()
```

Sequential execution for debugging or when parallelism isn't needed.

**Context Manager Support:**
```python
with LocalExecutor(n_jobs=4) as executor:
    results = executor.map(process_item, items)
# Automatically shuts down on exit
```

**Example:**
```python
from sklearn_meta.execution.local import LocalExecutor, SequentialExecutor

# Parallel execution
with LocalExecutor(n_jobs=-1) as executor:
    results = executor.map(fit_model, model_configs)

# Sequential for debugging
with SequentialExecutor() as executor:
    results = executor.map(fit_model, model_configs)
```

---

## Quick Import Reference

```python
# Fluent API (recommended for most use cases)
from sklearn_meta import GraphBuilder

# Core
from sklearn_meta import (
    DataContext,
    CVConfig,
    DataManager,
    ModelNode,
    ModelGraph,
    DependencyType,
    DependencyEdge,
    TuningOrchestrator,
    TuningConfig,
    OptimizationStrategy,
)

# Search
from sklearn_meta import SearchSpace, OptunaBackend
from sklearn_meta.search.parameter import FloatParameter, IntParameter, CategoricalParameter

# Meta-learning
from sklearn_meta import (
    CorrelationAnalyzer,
    HyperparameterCorrelation,
    Reparameterization,
    LogProductReparameterization,
    RatioReparameterization,
    LinearReparameterization,
    ReparameterizedSpace,
    get_prebaked_reparameterization,
)

# Selection
from sklearn_meta import FeatureSelector, FeatureSelectionConfig

# Plugins
from sklearn_meta.plugins.base import ModelPlugin, CompositePlugin
from sklearn_meta.plugins.registry import PluginRegistry, get_global_registry
from sklearn_meta.plugins.xgboost.multiplier import XGBMultiplierPlugin
from sklearn_meta.plugins.xgboost.importance import XGBImportancePlugin

# Persistence
from sklearn_meta import FitCache, AuditLogger
from sklearn_meta.persistence.store import ArtifactStore, ArtifactMetadata

# Execution
from sklearn_meta.execution.local import LocalExecutor, SequentialExecutor
```
