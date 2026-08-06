"""Parameter sweeps — expand one :class:`Experiment` into many variants.

The runner consumes a single :class:`Experiment` at a time, and an
``Experiment`` carries fixed ``models``, ``prompts``, and
``sampling``. Sweep helpers generate a tuple of variant Experiments
when the caller wants to compare configurations side by side.

Two helpers ship:

- :func:`sweep_sampling` — vary ``SamplingParams`` axes (temperature,
  top_p, max_tokens) and produce one Experiment per Cartesian
  combination.
- :func:`sweep` — generic: vary any top-level attribute of
  ``Experiment`` across the supplied values.

Both helpers name variants by appending a deterministic suffix to
the base experiment name so downstream consumers can identify which
variant produced which Outcome.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from strata_forge.evals.experiment import Experiment, SamplingParams

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "sweep",
    "sweep_sampling",
]


def sweep_sampling(
    base: Experiment,
    *,
    temperatures: Sequence[float] | None = None,
    max_tokens: Sequence[int] | None = None,
    top_p: Sequence[float] | None = None,
) -> tuple[Experiment, ...]:
    """Generate one :class:`Experiment` variant per sampling-axis combination.

    Each provided sequence becomes an axis; axes not provided keep
    the base experiment's value. The variant name is
    ``{base.name}/{axis}={value}/{axis}={value}/...`` with axes in
    a deterministic order (temperature, top_p, max_tokens).

    Returns a single-element tuple when no axes are provided (the
    base experiment itself, with its original name preserved).
    """
    axes: list[tuple[str, Sequence[Any]]] = []
    if temperatures is not None:
        axes.append(("temperature", temperatures))
    if top_p is not None:
        axes.append(("top_p", top_p))
    if max_tokens is not None:
        axes.append(("max_tokens", max_tokens))

    if not axes:
        return (base,)

    variants: list[Experiment] = []
    for combination in itertools.product(*(values for _, values in axes)):
        overrides = dict(zip((name for name, _ in axes), combination, strict=True))
        suffix = "/".join(f"{name}={value}" for name, value in overrides.items())
        new_sampling = base.sampling.model_copy(update=overrides)
        variants.append(
            base.model_copy(
                update={
                    "name": f"{base.name}/{suffix}",
                    "sampling": new_sampling,
                }
            )
        )
    return tuple(variants)


def sweep(
    base: Experiment,
    *,
    name_separator: str = "/",
    **axes: Sequence[Any],
) -> tuple[Experiment, ...]:
    """Generic sweep across arbitrary top-level :class:`Experiment` fields.

    Each keyword argument is one axis. The function emits one
    variant per combination across the axes, using
    :meth:`Experiment.model_copy` to apply the update.

    The variant name is ``{base.name}{sep}{axis}={value}...`` with
    axes in keyword-argument order.

    Raises:
        ValueError: When an axis name isn't an :class:`Experiment`
            field, or when no axes are supplied.
    """
    if not axes:
        msg = "sweep needs at least one axis"
        raise ValueError(msg)
    valid_fields = set(Experiment.model_fields)
    bad = [name for name in axes if name not in valid_fields]
    if bad:
        msg = f"Unknown Experiment fields: {bad!r}"
        raise ValueError(msg)

    axis_items = list(axes.items())
    variants: list[Experiment] = []
    for combination in itertools.product(*(values for _, values in axis_items)):
        overrides: dict[str, Any] = dict(
            zip((name for name, _ in axis_items), combination, strict=True)
        )
        suffix = name_separator.join(
            f"{name}={_format_value(value)}" for name, value in overrides.items()
        )
        variants.append(
            base.model_copy(update={**overrides, "name": f"{base.name}{name_separator}{suffix}"})
        )
    return tuple(variants)


def _format_value(value: Any) -> str:
    """Render a value for inclusion in a variant name.

    Tuples become comma-separated lists; SamplingParams render as
    ``temp=X-top_p=Y-max=Z`` (compact). Everything else falls back
    to ``str``.
    """
    if isinstance(value, tuple):
        tup: tuple[Any, ...] = value  # type: ignore[assignment]
        return ",".join(str(v) for v in tup)
    if isinstance(value, SamplingParams):
        parts: list[str] = []
        if value.temperature is not None:
            parts.append(f"temp={value.temperature}")
        if value.top_p is not None:
            parts.append(f"top_p={value.top_p}")
        if value.max_tokens is not None:
            parts.append(f"max={value.max_tokens}")
        return "-".join(parts) if parts else "default"
    return str(value)
