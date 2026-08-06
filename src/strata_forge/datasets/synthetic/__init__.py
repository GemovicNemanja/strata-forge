"""Synthetic-data primitives for :class:`strata_forge.datasets.Dataset`.

Two helpers, both built on :class:`strata_forge.llm.LLMClient`:

- :func:`self_instruct` — generate new items from seed examples by
  prompting an LLM with the seeds and an instruction.
- :func:`distill` — fill in missing ``expected_output`` fields by
  running a teacher LLM over each item's input.

Both are pure async functions; they return a fresh
:class:`~strata_forge.datasets.Dataset` rather than mutating any input.
"""

from strata_forge.datasets.synthetic.distillation import distill
from strata_forge.datasets.synthetic.self_instruct import (
    SelfInstructBatch,
    SelfInstructItem,
    self_instruct,
)

__all__ = [
    "SelfInstructBatch",
    "SelfInstructItem",
    "distill",
    "self_instruct",
]
