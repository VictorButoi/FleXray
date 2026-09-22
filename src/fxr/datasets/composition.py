from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import gcd
from typing import Any

from torch.utils.data import Dataset, Sampler

from fxr.config._proportions import schedule_counts_from_proportions

from .schemas import NamedBatch, TrainingSample


@dataclass(frozen=True)
class SourceSpec:
    """Named dataset source used by source-aware composition helpers.

    Attributes:
        name: Stable source name injected into emitted samples as
            ``dataset_name``.
        dataset: Torch dataset that provides samples for this source.
        modality: Optional modality label injected into emitted samples. When
            omitted, ``CompositeSourceDataset`` preserves an existing sample
            modality or falls back to ``name``.
    """

    name: str
    dataset: Dataset
    modality: str | None = None


class CompositeSourceDataset(Dataset):
    """Concatenate named datasets while injecting source metadata into samples.

    Attributes:
        sources: Ordered source definitions included in the composite dataset.
        source_indices: Mapping from source name to the composite indices owned
            by that source.
        source_modalities: Mapping from source name to its configured modality,
            or ``None`` when no modality was provided.
        _offsets: Internal ``(start, stop, source)`` index ranges used to map a
            composite index to its source dataset.
        _length: Total number of samples across all sources.
    """

    def __init__(
        self,
        sources: Mapping[str, Dataset] | Sequence[SourceSpec],
        *,
        modalities: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the composite from named datasets or source specs.

        Args:
            sources: Mapping of source names to datasets, or an ordered sequence
                of ``SourceSpec`` values.
            modalities: Optional source-name-to-modality mapping used when
                ``sources`` is passed as a mapping.

        Returns:
            ``None``.

        Raises:
            ValueError: If no sources are provided.
        """

        if isinstance(sources, Mapping):
            self.sources = tuple(
                SourceSpec(
                    name=str(name),
                    dataset=dataset,
                    modality=None if modalities is None else modalities.get(str(name)),
                )
                for name, dataset in sources.items()
            )
        else:
            self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("CompositeSourceDataset requires at least one source.")
        self._offsets: list[tuple[int, int, SourceSpec]] = []
        offset = 0
        for source in self.sources:
            length = len(source.dataset)
            self._offsets.append((offset, offset + length, source))
            offset += length
        self.source_indices = {
            source.name: tuple(range(start, stop))
            for start, stop, source in self._offsets
        }
        self.source_modalities = {
            source.name: source.modality for source in self.sources
        }
        self._length = offset

    def __len__(self) -> int:
        """Return the total number of samples across all sources.

        Returns:
            Composite sample count.
        """

        return self._length

    def __getitem__(self, index: int) -> TrainingSample:
        """Return one sample with source metadata injected.

        Args:
            index: Composite dataset index. Negative indices follow standard
                Python sequence semantics.

        Returns:
            Training sample copied from the owning source dataset with
            ``dataset_name`` and ``modality`` populated.

        Raises:
            IndexError: If ``index`` is outside the composite dataset range.
        """

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for start, stop, source in self._offsets:
            if start <= index < stop:
                sample = dict(source.dataset[index - start])
                sample["dataset_name"] = source.name
                if source.modality is not None:
                    sample["modality"] = source.modality
                elif "modality" not in sample:
                    sample["modality"] = source.name
                return sample
        raise IndexError(index)


class HomogeneousSourceBatchSampler(Sampler[list[int]]):
    """Yield batches whose indices all come from the same source.

    Attributes:
        batch_size: Number of indices to emit per batch.
        drop_last: Whether to drop a final short batch for each source.
        source_indices: Mapping from source name to the indices available for
            that source.
    """

    def __init__(
        self,
        source_indices: Mapping[str, Sequence[int]] | CompositeSourceDataset,
        *,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        """Initialize the sampler from source index groups.

        Args:
            source_indices: Source index mapping or a ``CompositeSourceDataset``
                whose ``source_indices`` should be used.
            batch_size: Number of samples per emitted batch.
            drop_last: Whether to discard incomplete trailing batches within
                each source.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.drop_last = drop_last
        if isinstance(source_indices, CompositeSourceDataset):
            self.source_indices = source_indices.source_indices
        else:
            self.source_indices = {
                str(source): tuple(indices)
                for source, indices in source_indices.items()
            }

    def __iter__(self) -> Iterator[list[int]]:
        """Yield homogeneous batches source by source.

        Returns:
            Iterator of composite dataset index lists.
        """

        for _, indices in self.source_indices.items():
            for start in range(0, len(indices), self.batch_size):
                batch = list(indices[start : start + self.batch_size])
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        """Return the number of batches that iteration will emit.

        Returns:
            Batch count after applying ``drop_last`` within each source.
        """

        total = 0
        for indices in self.source_indices.values():
            full, partial = divmod(len(indices), self.batch_size)
            total += full
            if partial and not self.drop_last:
                total += 1
        return total


class MixedDataLoader:
    """Interleave named loaders with deterministic, weighted epoch schedules.

    Fixed-length epochs apportion every step directly from the configured
    relative weights. The exact-size source multiset is then dispersed with a
    seed-controlled stratified shuffle, so fractional weights do not depend on
    a repeating-schedule boundary and sources do not form one grouped block.
    Starting another iteration advances to the next deterministic epoch without
    changing Python's process-wide random state.

    Attributes:
        loaders: Mapping from source name to iterable dataloader.
        modalities: Optional modality labels keyed by source name.
        schedule: Source schedule for the most recently started epoch, or the
            seed's epoch-zero schedule before iteration starts.
        iters_per_epoch: Optional fixed number of batches yielded per epoch.
        seed: Integer seed controlling source-order shuffling between epochs.
        _schedule_counts: Reduced positive integer representation of the
            configured relative source weights.
        _next_epoch: Zero-based epoch index assigned to the next iteration.
    """

    def __init__(
        self,
        loaders: Mapping[str, Iterable[Any]],
        *,
        modalities: Mapping[str, str] | None = None,
        proportions: Mapping[str, Any] | None = None,
        iters_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the mixed loader schedule.

        Args:
            loaders: Mapping from source name to iterable dataloader.
            modalities: Optional modality labels keyed by source name.
            proportions: Optional positive numeric weights keyed by source name.
            iters_per_epoch: Optional fixed number of batches to emit. When set,
                source loaders are cycled as needed instead of exhausting early.
            seed: Integer seed used only for deterministic source scheduling.

        Returns:
            ``None``.

        Raises:
            ValueError: If no loaders are provided or any proportion is not
                positive, if ``iters_per_epoch`` is not positive, if ``seed`` is
                not an integer, or if a sized source emits zero batches.
        """

        self.loaders = dict(loaders)
        if not self.loaders:
            raise ValueError("MixedDataLoader requires at least one loader.")
        self.modalities = dict(modalities or {})
        self.iters_per_epoch = _normalize_iters_per_epoch(iters_per_epoch)
        self.seed = _normalize_seed(seed)
        self._schedule_counts = _weighted_schedule_counts(
            self.loaders,
            proportions,
        )
        self._next_epoch = 0
        _reject_sized_empty_loaders(self.loaders)
        self.schedule = self._build_epoch_schedule(0)

    def __iter__(self) -> Iterator[NamedBatch]:
        """Yield named batches according to the next seeded epoch schedule.

        Returns:
            Iterator of ``NamedBatch`` values with source and modality metadata.

        Raises:
            ValueError: If a source iterable emits no batches.
        """

        epoch = self._next_epoch
        self._next_epoch += 1
        self.schedule = self._build_epoch_schedule(epoch)
        if self.iters_per_epoch is not None:
            yield from self._iter_fixed_epoch(self.schedule)
            return

        iterators = {name: iter(self.loaders[name]) for name in sorted(self.loaders)}
        active = set(iterators)
        emitted = {name: False for name in iterators}
        while active:
            progressed = False
            for name in self.schedule:
                if name not in active:
                    continue
                try:
                    batch = next(iterators[name])
                except StopIteration:
                    if not emitted[name]:
                        raise ValueError(
                            f"MixedDataLoader source {name!r} produced no batches."
                        )
                    active.remove(name)
                    continue
                emitted[name] = True
                progressed = True
                yield NamedBatch(
                    source_name=name,
                    modality=self.modalities.get(name),
                    batch=batch,
                )
            if not progressed:
                break

    def __len__(self) -> int:
        """Return the configured epoch length or the one-pass loader length.

        Returns:
            Batch count emitted by one iteration of this mixed loader.

        Raises:
            TypeError: If no fixed epoch length is configured and an underlying
                loader does not define ``len``.
        """

        if self.iters_per_epoch is not None:
            return self.iters_per_epoch
        return sum(len(loader) for loader in self.loaders.values())  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch used by the next iteration.

        This hook lets checkpoint restoration resume source scheduling at the
        same epoch as an uninterrupted training run.

        Args:
            epoch: Non-negative, zero-based epoch index for the next iteration.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``epoch`` is not a non-negative integer.
        """

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("MixedDataLoader epoch must be a non-negative integer.")
        self._next_epoch = epoch
        self.schedule = self._build_epoch_schedule(epoch)

    def _build_epoch_schedule(self, epoch: int) -> tuple[str, ...]:
        """Build one exact-count, seed-controlled source schedule.

        Args:
            epoch: Zero-based iteration index used to derive the local shuffle.

        Returns:
            Source names in their deterministic order for the requested epoch.
        """

        counts = self._schedule_counts
        if self.iters_per_epoch is not None:
            counts = _apportion_schedule_counts(
                counts,
                num_steps=self.iters_per_epoch,
            )
        return _stratified_source_schedule(
            counts,
            seed=self.seed,
            epoch=epoch,
        )

    def _iter_fixed_epoch(
        self,
        schedule: Sequence[str],
    ) -> Iterator[NamedBatch]:
        """Yield exactly ``iters_per_epoch`` batches by cycling source loaders.

        Args:
            schedule: Exact-length source sequence for the current epoch.

        Returns:
            Iterator of ``NamedBatch`` values following the weighted source
            schedule.
        """

        iterators = {name: iter(self.loaders[name]) for name in sorted(self.loaders)}
        for name in schedule:
            batch = self._next_cycled_batch(name, iterators)
            yield NamedBatch(
                source_name=name,
                modality=self.modalities.get(name),
                batch=batch,
            )

    def _next_cycled_batch(
        self,
        name: str,
        iterators: dict[str, Iterator[Any]],
    ) -> Any:
        """Return the next batch for ``name``, restarting its loader once.

        Args:
            name: Source loader name.
            iterators: Mutable per-source iterators.

        Returns:
            The next batch from the selected source.

        Raises:
            ValueError: If the selected source loader emits no batches.
        """

        try:
            return next(iterators[name])
        except StopIteration:
            iterators[name] = iter(self.loaders[name])
        try:
            return next(iterators[name])
        except StopIteration as exc:
            raise ValueError(
                f"MixedDataLoader source {name!r} produced no batches and cannot "
                "be cycled for a fixed epoch."
            ) from exc


class SequentialDataLoader:
    """Iterate named loaders one after another for validation-style passes.

    Attributes:
        loaders: Mapping from source name to iterable dataloader.
        modalities: Optional modality labels keyed by source name.
    """

    def __init__(
        self,
        loaders: Mapping[str, Iterable[Any]],
        *,
        modalities: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize sequential iteration over named loaders.

        Args:
            loaders: Mapping from source name to iterable dataloader.
            modalities: Optional modality labels keyed by source name.

        Returns:
            ``None``.

        Raises:
            ValueError: If no loaders are provided.
        """

        self.loaders = dict(loaders)
        if not self.loaders:
            raise ValueError("SequentialDataLoader requires at least one loader.")
        self.modalities = dict(modalities or {})

    def __iter__(self) -> Iterator[NamedBatch]:
        """Yield all batches from each loader before advancing sources.

        Returns:
            Iterator of NamedBatch values with source and modality metadata.
        """

        for name, loader in self.loaders.items():
            for batch in loader:
                yield NamedBatch(
                    source_name=name,
                    modality=self.modalities.get(name),
                    batch=batch,
                )


def _weighted_schedule_counts(
    loaders: Mapping[str, Iterable[Any]],
    proportions: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Return insertion-order-independent integer source weights.

    Args:
        loaders: Loader mapping whose keys define source names.
        proportions: Optional positive numeric source weights. Missing sources
            default to weight 1.

    Returns:
        Alphabetically keyed, reduced positive integer source weights.

    Raises:
        ValueError: If any source weight is not positive.
    """

    source_names = sorted(loaders)
    if proportions is None:
        return {name: 1 for name in source_names}
    counts = schedule_counts_from_proportions(
        source_names,
        proportions,
        name="MixedDataLoader proportions",
    )
    common_divisor = 0
    for count in counts.values():
        common_divisor = gcd(common_divisor, count)
    return {name: count // common_divisor for name, count in counts.items()}


def _apportion_schedule_counts(
    weights: Mapping[str, int],
    *,
    num_steps: int,
) -> dict[str, int]:
    """Apportion an exact number of epoch steps from relative integer weights.

    The largest-remainder method keeps each emitted count within one of its
    ideal fractional allocation. Equal remainders are resolved by source name,
    making the result independent of loader insertion order.

    Args:
        weights: Positive relative integer weights keyed by source name.
        num_steps: Exact number of source selections to allocate.

    Returns:
        Per-source non-negative counts summing exactly to ``num_steps``.
    """

    total_weight = sum(weights.values())
    counts: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for name in sorted(weights):
        count, remainder = divmod(num_steps * weights[name], total_weight)
        counts[name] = count
        remainders[name] = remainder

    remaining = num_steps - sum(counts.values())
    ranked = sorted(remainders, key=lambda name: (-remainders[name], name))
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def _stratified_source_schedule(
    counts: Mapping[str, int],
    *,
    seed: int,
    epoch: int,
) -> tuple[str, ...]:
    """Disperse exact source counts with deterministic per-epoch jitter.

    Each source contributes one selection to each of its own equal-width epoch
    strata. Random jitter within those strata changes order between epochs
    without the long source blocks produced by concatenating repeated names.

    Args:
        counts: Exact number of selections required for every source.
        seed: User-provided source scheduling seed.
        epoch: Zero-based epoch index.

    Returns:
        Exact-length tuple containing each source the requested number of times.
    """

    rng = random.Random(f"{seed}:{epoch}")
    positioned: list[tuple[float, str]] = []
    for name in sorted(counts):
        count = counts[name]
        for occurrence in range(count):
            position = (occurrence + rng.random()) / count
            positioned.append((position, name))
    positioned.sort()
    return tuple(name for _, name in positioned)


def _reject_sized_empty_loaders(
    loaders: Mapping[str, Iterable[Any]],
) -> None:
    """Reject source loaders known to emit zero batches.

    Args:
        loaders: Source loader mapping to validate.

    Returns:
        ``None``.

    Raises:
        ValueError: If a loader defines ``len`` and reports zero batches.
    """

    for name in sorted(loaders):
        try:
            length = len(loaders[name])  # type: ignore[arg-type]
        except (TypeError, NotImplementedError):
            continue
        if length == 0:
            raise ValueError(f"MixedDataLoader source {name!r} produced no batches.")


def _normalize_seed(value: Any) -> int:
    """Validate and return a source-schedule seed.

    Args:
        value: Raw seed value.

    Returns:
        Integer seed.

    Raises:
        ValueError: If ``value`` is not an integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("MixedDataLoader seed must be an integer.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    """Return ``value`` as a positive integer without silent truncation.

    Args:
        value: Raw config value.
        name: Human-readable value name used in errors.

    Returns:
        Positive integer value.

    Raises:
        ValueError: If ``value`` is not a positive integer.
    """

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer.") from exc
    else:
        raise ValueError(f"{name} must be a positive integer.")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _normalize_iters_per_epoch(value: int | None) -> int | None:
    """Validate an optional fixed epoch length.

    Args:
        value: Raw epoch length or ``None``.

    Returns:
        Positive integer epoch length, or ``None`` when unset.

    Raises:
        ValueError: If ``value`` is not positive.
    """

    if value is None:
        return None
    return _positive_integer(value, "MixedDataLoader iters_per_epoch")
