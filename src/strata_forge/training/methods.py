"""The registry of fine-tuning methods — one entry per method, read by everything.

Forge ships five trainers across two runner classes, each with its own config shape, its own
acceptable dataset formats, and (for GRPO) its own extra requirement. Anything that drives training
from a declaration rather than from Python — an orchestrator's inert job spec, a CLI flag, a form —
needs to answer the same three questions about a method it was handed by name: does it exist, what
dataset shapes can it consume, and what does it build. Answering those in each caller is how a
method ends up half-added: routable in one place and unknown in another.

So there is one table. :func:`pick_method` is the only supported way to go from a method name to
anything, and a method that is not in the table does not exist.

``enabled`` is what keeps GRPO honest. It is fully implemented in
:class:`strata_forge.training.preference.PreferenceRunner`, but it needs ``reward_funcs`` —
callables, or the id of a reward model. A caller building a run from an INERT spec (no code, by
construction: see :mod:`strata_forge.pipelines.finetune_runner`) cannot supply a callable, so GRPO
is registered, described, and declined with a reason rather than quietly missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from strata_forge.training.preference import DPOConfig, KTOConfig, ORPOConfig, PreferenceRunner
from strata_forge.training.sft import SFTConfig, SFTRunner

if TYPE_CHECKING:
    from strata_forge.training.dataset_format import DatasetFormat
    from strata_forge.training.peft import LoRAConfig, QLoRAConfig

__all__ = [
    "METHODS",
    "MethodName",
    "MethodSpec",
    "UnsupportedMethodError",
    "check_format",
    "enabled_methods",
    "pick_method",
]


type MethodName = Literal["sft", "dpo", "orpo", "kto", "grpo"]

# The head a method trains. Every shipped method is decoder-only today; the field exists because it
# is the seam a second head (sequence classification, seq2seq) arrives through — a registry entry
# with a different task_type, rather than a branch threaded through the runners.
type TaskType = Literal["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"]


class UnsupportedMethodError(ValueError):
    """A method that is unknown, or known but not usable from a declaration. Names why."""


@dataclass(frozen=True)
class MethodSpec:
    """Everything a caller needs to know about one fine-tuning method, without importing TRL."""

    name: MethodName
    label: str
    task_type: TaskType
    config_cls: type[SFTConfig | DPOConfig | ORPOConfig | KTOConfig]
    runner_cls: type[SFTRunner | PreferenceRunner]
    formats: frozenset[DatasetFormat]
    enabled: bool = True
    # Why a registered method is declined. Surfaced verbatim, so it has to read as an explanation
    # to a user rather than a note to a maintainer.
    disabled_reason: str = ""
    # Knobs beyond the shared base that this method accepts, for a caller rendering a form. The
    # config's own ``extra="forbid"`` remains the authority; this only says what to offer.
    tuning_knobs: tuple[str, ...] = field(default_factory=tuple)

    def build_runner(
        self, config: Any, *, peft_config: LoRAConfig | QLoRAConfig | None = None
    ) -> SFTRunner | PreferenceRunner:
        """Instantiate this method's runner. Constructs nothing heavy — see the runner classes."""
        return self.runner_cls(config, peft_config=peft_config)


_SFT_FORMATS: frozenset[DatasetFormat] = frozenset({"text", "prompt_completion", "conversational"})

METHODS: dict[str, MethodSpec] = {
    "sft": MethodSpec(
        name="sft",
        label="Supervised fine-tuning",
        task_type="CAUSAL_LM",
        config_cls=SFTConfig,
        runner_cls=SFTRunner,
        formats=_SFT_FORMATS,
        tuning_knobs=("max_seq_length", "packing"),
    ),
    "dpo": MethodSpec(
        name="dpo",
        label="Direct Preference Optimization",
        task_type="CAUSAL_LM",
        config_cls=DPOConfig,
        runner_cls=PreferenceRunner,
        formats=frozenset({"preference"}),
        tuning_knobs=("beta", "loss_type", "max_length"),
    ),
    "orpo": MethodSpec(
        name="orpo",
        label="Odds-Ratio Preference Optimization",
        task_type="CAUSAL_LM",
        config_cls=ORPOConfig,
        runner_cls=PreferenceRunner,
        formats=frozenset({"preference"}),
        tuning_knobs=("beta", "max_length"),
    ),
    "kto": MethodSpec(
        name="kto",
        label="Kahneman-Tversky Optimization",
        task_type="CAUSAL_LM",
        config_cls=KTOConfig,
        runner_cls=PreferenceRunner,
        formats=frozenset({"unpaired_preference"}),
        tuning_knobs=(
            "beta",
            "desirable_weight",
            "undesirable_weight",
            "max_length",
        ),
    ),
    "grpo": MethodSpec(
        name="grpo",
        label="Group Relative Policy Optimization",
        task_type="CAUSAL_LM",
        # GRPOConfig is deliberately absent from the imports above: nothing may build one through
        # this table while the method is declined, and importing it would suggest otherwise.
        config_cls=SFTConfig,
        runner_cls=PreferenceRunner,
        formats=frozenset(),
        enabled=False,
        disabled_reason=(
            "GRPO needs a reward function, which cannot be supplied from a job specification. "
            "Use strata_forge.training.PreferenceRunner directly with reward_funcs="
        ),
    ),
}


def enabled_methods() -> list[MethodSpec]:
    """The methods a declaration may name, in registry order."""
    return [spec for spec in METHODS.values() if spec.enabled]


def pick_method(name: str) -> MethodSpec:
    """The :class:`MethodSpec` for ``name``.

    Raises :class:`UnsupportedMethodError` for an unknown method (listing what is available) and for
    a registered-but-declined one (carrying its ``disabled_reason``) — two different problems that
    a caller reports differently.
    """
    spec = METHODS.get(name)
    if spec is None:
        known = ", ".join(s.name for s in enabled_methods())
        msg = f"unknown fine-tuning method {name!r}; expected one of: {known}"
        raise UnsupportedMethodError(msg)
    if not spec.enabled:
        raise UnsupportedMethodError(spec.disabled_reason)
    return spec


def check_format(spec: MethodSpec, fmt: str) -> None:
    """Reject a dataset format the method cannot train on, naming the ones it can."""
    if fmt not in spec.formats:
        accepted = ", ".join(sorted(spec.formats))
        msg = (
            f"{spec.name} cannot train on the {fmt!r} dataset format; "
            f"it accepts: {accepted or '(none)'}"
        )
        raise UnsupportedMethodError(msg)
