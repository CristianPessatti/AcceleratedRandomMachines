# AcceleratedRandomMachines
Implementation of accelerated SVM strategies for the method of Random Machines

## Documentation status
The R source files under `activeLearning/`, `functions/`, and `automation/` have
been updated with descriptive headers and function-level comments. These
comments explain the purpose, inputs, outputs, and important implementation
notes without changing any logic.

Key modules:
- `activeLearning/active_learning.R`: main active learning routine with pooled
  nearest-enemy selection and per-class best additions, logging, and plateau
  stopping.
- `activeLearning/fit_model.R`: helper to train a single SVM and collect
  validation and support-vector diagnostics.
- `functions/sampling/*`: localized sampling, nearest-enemy sampling, and
  systematic samplers used by initialization and candidate selection.
- `functions/utils/*`: utilities for timing, partitioning, class proportions,
  and nearest-enemy distance.
- `functions/math/*`: entropy and scaling helpers used by samplers.
- `functions/automation/run_benchmark.R`: repeated holdout benchmark comparing
  full, localized, and nearest-enemy strategies.

No functional changes were made; only comments and documentation were added.