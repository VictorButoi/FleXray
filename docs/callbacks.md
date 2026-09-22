# FleXray Callbacks

`fxr.callbacks` is the framework-neutral training callback surface for
FleXray. It defines typed event contexts, named hook conventions, a small
fail-fast dispatcher, an import-string config builder, and stdout-only built-in
callbacks. It does not own training loops or experiment runtime behavior.

## Event Contexts

All event contexts are frozen dataclasses. Mapping fields are copied into
read-only mapping proxies so callbacks cannot accidentally mutate shared event
metadata through the context object.

`fxr.callbacks` exports:

- `TrainContext(metadata={...})` for train-level events.
- `EpochContext(phase, epoch, num_epochs=None, metadata={...})` for epoch-level
  events.
- `BatchContext(phase, epoch, batch_idx, num_batches, metadata={...})` for
  batch-level events. `batch_idx` is zero-based.
- `MetricsContext(records, metadata={...})` for metric records. Records are a
  list of mappings with `epoch`, `phase`, optional `dataset`, and metric-value
  columns.

## Hook Names

Callbacks use named methods by convention. A callback may implement any subset:

- `on_train_start(context)`
- `on_train_end(context)`
- `on_epoch_start(context)`
- `on_epoch_end(context)`
- `on_batch_start(context)`
- `on_batch_end(context)`
- `on_metrics(context)`

Missing hooks are ignored.

## Runner

`CallbackRunner(callbacks)` stores callback instances in order and dispatches
hooks in that order. It ignores missing hooks and propagates callback exceptions
immediately, so later callbacks are not run after a failure.

```python
from fxr.callbacks import BatchContext, BatchProgressLogger, CallbackRunner

runner = CallbackRunner([BatchProgressLogger(every_n_batches=25)])
runner.on_batch_end(
    BatchContext(phase="train", epoch=1, batch_idx=0, num_batches=100)
)
```

The runner also exposes `dispatch(hook_name, context)` for custom hook names.

## Config Builder

`build_callback_runner(specs)` builds a runner from a list of mappings. Each
mapping must define `_class` as a dotted import string. Remaining keys are
passed as constructor keyword arguments. Resolution uses local
`importlib`; there is no external config integration.

```python
from fxr.callbacks import build_callback_runner

runner = build_callback_runner(
    [
        {
            "_class": "fxr.callbacks.BatchProgressLogger",
            "every_n_batches": 10,
            "phases": ["train"],
        },
        {
            "_class": "fxr.callbacks.MetricsTablePrinter",
            "floatfmt": ".3f",
        },
    ]
)
```

## Built-In Callbacks

`BatchProgressLogger` implements `on_batch_end(BatchContext)`. It prints the
first batch, every configured interval, and the final batch to stdout. Use
`phases` to restrict logging to one phase or a list of phases.

`MetricsTablePrinter` implements `on_metrics(MetricsContext)`. It accepts
records shaped like:

```python
[
    {"epoch": 1, "phase": "train", "loss": 0.42, "dice": 0.81},
    {"epoch": 1, "phase": "val", "dataset": "HipRay", "loss": 0.51, "dice": 0.77},
]
```

The printer uses `pandas` and `tabulate` to pivot metric columns into rows and
phase labels into columns. When `dataset` is present, the phase label becomes
`phase/dataset` so multiple validation datasets remain distinct.

## Package Boundary

`fxr.callbacks` remains independent of an experiment instance. The
experiment-coupled WandB sample logger and eval-set metric callback live in
`fxr.experiment`, where they are constructed with the active experiment and
dispatched by callback group. Checkpoint policy and training-loop behavior also
belong to `fxr.experiment`; inference sidecars belong to `fxr.inference`.
