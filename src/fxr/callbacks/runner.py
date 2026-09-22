from __future__ import annotations

from collections.abc import Iterable

from .contexts import BatchContext, EpochContext, MetricsContext, TrainContext


class CallbackRunner:
    """Ordered fail-fast dispatcher for framework-neutral callbacks.

    Attributes:
        _callbacks: Immutable callback sequence stored for dispatch.
        callbacks: Callback objects dispatched in the order they were supplied.
    """

    def __init__(self, callbacks: Iterable[object] | None = None) -> None:
        """Store callbacks for ordered hook dispatch.

        Args:
            callbacks: Callback instances to dispatch, or ``None`` for an empty
                runner.

        Returns:
            ``None``.
        """

        self._callbacks = tuple(callbacks) if callbacks is not None else ()

    @property
    def callbacks(self) -> tuple[object, ...]:
        """Return callbacks in dispatch order.

        Returns:
            Tuple of callback objects.
        """

        return self._callbacks

    def dispatch(self, hook_name: str, context: object) -> None:
        """Dispatch one named hook to every callback that defines it.

        Args:
            hook_name: Hook method name to look up on each callback.
            context: Event context passed to each hook method.

        Returns:
            ``None``.

        Raises:
            TypeError: If ``hook_name`` is not a string or a present hook is not
                callable.
            ValueError: If ``hook_name`` is empty.
            Exception: Any exception raised by a callback hook is propagated
                immediately and stops later callbacks from running.
        """

        if not isinstance(hook_name, str):
            raise TypeError(
                f"hook_name must be a string, got {type(hook_name).__name__}."
            )
        if not hook_name:
            raise ValueError("hook_name must be non-empty.")

        for callback in self._callbacks:
            hook = getattr(callback, hook_name, None)
            if hook is None:
                continue
            if not callable(hook):
                raise TypeError(
                    f"Callback {callback!r} defines non-callable hook {hook_name!r}."
                )
            hook(context)

    def on_train_start(self, context: TrainContext) -> None:
        """Dispatch the train-start event.

        Args:
            context: Train-level event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_train_start", context)

    def on_train_end(self, context: TrainContext) -> None:
        """Dispatch the train-end event.

        Args:
            context: Train-level event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_train_end", context)

    def on_epoch_start(self, context: EpochContext) -> None:
        """Dispatch the epoch-start event.

        Args:
            context: Epoch-level event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_epoch_start", context)

    def on_epoch_end(self, context: EpochContext) -> None:
        """Dispatch the epoch-end event.

        Args:
            context: Epoch-level event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_epoch_end", context)

    def on_batch_start(self, context: BatchContext) -> None:
        """Dispatch the batch-start event.

        Args:
            context: Batch-level event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_batch_start", context)

    def on_batch_end(self, context: BatchContext) -> None:
        """Dispatch the batch-end event.

        Args:
            context: Batch-level event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_batch_end", context)

    def on_metrics(self, context: MetricsContext) -> None:
        """Dispatch the metrics event.

        Args:
            context: Metrics event context.

        Returns:
            ``None``.
        """

        self.dispatch("on_metrics", context)
