"""Classifier heads: how a sequence-classification model is built from a checkpoint, and checked.

Three ways to get a primary head, each declared rather than inferred (``head_init``):

- ``new``: an encoder base whose checkpoint has no classifier. The classifier is the stock
  random initialisation, and the ONLY keys allowed to be missing are that architecture's declared
  classifier keys (:data:`ENCODER_NEW_HEAD`).
- ``decoder_lm_head``: a decoder base whose classifier starts as its own LM head restricted to the
  labels' single-token verbalizers, so step 0 equals the zero-shot restricted softmax. The
  prediction head (dense + norm) is copied from the checkpoint's ``lm_head`` and the classifier
  rows from the output projection (``decoder.weight``, or the tied embedding when the checkpoint
  does not store it) AND its bias (``decoder.bias``). A stock classifier without a bias cannot
  represent a nonzero ``decoder.bias``, so the classifier is rebuilt with one, and the prediction
  head's dense layer gains a bias that is zero and frozen (the saved config then sets
  ``classifier_bias``, and the reloaded model computes the same function with every key present).
  The allowed missing keys are exactly the declared head keys (:data:`DECODER_LM_HEAD`).
- ``existing``: a checkpoint that already has this classifier (a previous run's output). Nothing
  may be missing, and its label order must equal the run's.

Every other missing key, every unexpected key outside the declared set, and every shape mismatch
refuses: a silently re-randomised layer trains, converges and reports success, so it is only ever
caught here. Mappings are explicit tables per architecture; there is no key-name heuristic and no
repository code.

The auxiliary heads read one pooled vector per row (the first token for an encoder, the answer
position for a decoder): the *flag* head is ``Linear(hidden, 1)`` and the *support* head is the
bilinear ``h^T W m_pid + b`` over each premise's marker token. Both start from
``normal(0, initializer_range)`` with a zero bias and are saved under fixed tensor names in
:data:`AUX_HEADS_FILENAME`.

Torch, transformers and safetensors are imported inside the functions that use them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "AUX_HEADS_FILENAME",
    "DECODER_LM_HEAD",
    "ENCODER_NEW_HEAD",
    "HeadLoadError",
    "HeadPlan",
    "LoadReceipt",
    "aux_outputs",
    "build_aux_heads",
    "check_load",
    "head_plan",
    "init_decoder_head",
    "load_aux_heads",
    "read_checkpoint_tensors",
    "save_aux_heads",
]

AUX_HEADS_FILENAME = "strata_aux_heads.safetensors"

#: ``model_type`` -> the keys a classifier-less encoder checkpoint is missing (exactly these), and
#: the checkpoint keys the sequence-classification model does not use.
ENCODER_NEW_HEAD: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "modernbert": (
        frozenset({"classifier.weight", "classifier.bias"}),
        frozenset({"decoder.weight", "decoder.bias"}),
    ),
}

#: ``model_type`` -> (prediction-head prefix in the seq-cls model, the LM-head prefix it is copied
#: from, the output-projection key, its bias key, the tied-embedding key, and the checkpoint keys
#: the seq-cls model leaves unused because the mapping consumes them).
DECODER_LM_HEAD: dict[str, dict[str, str]] = {
    "modernbert-decoder": {
        "head_prefix": "head.",
        "lm_head_prefix": "lm_head.",
        "projection": "decoder.weight",
        "projection_bias": "decoder.bias",
        "tied_embedding": "model.embeddings.tok_embeddings.weight",
    },
}


class HeadLoadError(ValueError):
    """A checkpoint that does not load into the declared head exactly. Names the keys."""


@dataclass(frozen=True)
class HeadPlan:
    """What a load is allowed to leave missing and unused, for one ``head_init``."""

    allowed_missing: frozenset[str]
    allowed_unexpected: frozenset[str]


@dataclass
class LoadReceipt:
    """What the load guard saw and decided, recorded in the run manifest."""

    head_init: str
    model_type: str
    missing: list[str] = field(default_factory=list[str])
    unexpected: list[str] = field(default_factory=list[str])
    initialised_from: dict[str, str] = field(default_factory=dict[str, str])
    verbalizer_token_ids: dict[str, int] = field(default_factory=dict[str, int])
    added_parameters: list[str] = field(default_factory=list[str])
    #: The checkpoint config's ``classifier_pooling`` when the method replaced it with ``cls``.
    pooling_changed_from: str | None = None
    #: Parameters held at exactly zero (and frozen there): added by this run, or carried by the
    #: continued checkpoint (its ``strata_verifier.json``, else its config's ``classifier_bias``).
    zero_parameters: list[str] = field(default_factory=list[str])


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:  # pragma: no cover
        msg = "the [finetuning] extra is required: pip install 'strata-forge[finetuning]'"
        raise ImportError(msg) from exc


def _decoder_plan(model: Any, table: dict[str, str]) -> HeadPlan:
    head_prefix = table["head_prefix"]
    head_keys = {name for name, _ in model.named_parameters() if name.startswith(head_prefix)}
    classifier = {name for name, _ in model.named_parameters() if name.startswith("classifier.")}
    lm_prefix = table["lm_head_prefix"]
    unused = {lm_prefix + key[len(head_prefix) :] for key in head_keys}
    unused |= {table["projection"], table["projection_bias"]}
    return HeadPlan(
        allowed_missing=frozenset(head_keys | classifier), allowed_unexpected=frozenset(unused)
    )


def head_plan(model: Any, head_init: str) -> HeadPlan:
    """The declared plan for ``model``'s architecture under ``head_init``, or a named refusal."""
    model_type = str(getattr(model.config, "model_type", ""))
    if head_init == "existing":
        return HeadPlan(allowed_missing=frozenset(), allowed_unexpected=frozenset())
    if head_init == "new":
        if model_type not in ENCODER_NEW_HEAD:
            msg = (
                f"head_init 'new' has no declared head keys for model type {model_type!r}; "
                f"supported: {', '.join(sorted(ENCODER_NEW_HEAD))}"
            )
            raise HeadLoadError(msg)
        missing, unused = ENCODER_NEW_HEAD[model_type]
        return HeadPlan(allowed_missing=missing, allowed_unexpected=unused)
    if head_init == "decoder_lm_head":
        if model_type not in DECODER_LM_HEAD:
            msg = (
                f"head_init 'decoder_lm_head' has no declared mapping for model type "
                f"{model_type!r}; supported: {', '.join(sorted(DECODER_LM_HEAD))}"
            )
            raise HeadLoadError(msg)
        return _decoder_plan(model, DECODER_LM_HEAD[model_type])
    msg = f"unknown head_init {head_init!r}"
    raise HeadLoadError(msg)


def check_load(model: Any, loading_info: dict[str, Any], head_init: str) -> LoadReceipt:
    """Refuse a load that left anything missing, unused or mis-shaped beyond the declared plan.

    ``loading_info`` is what ``from_pretrained(..., output_loading_info=True)`` returns. Missing
    keys must EQUAL the plan's set: a declared head key that loaded from the checkpoint means the
    base already has a head, and ``new`` would silently train over it (``existing`` is the
    declared way to continue one).
    """
    plan = head_plan(model, head_init)
    missing = set(loading_info.get("missing_keys") or [])
    unexpected = set(loading_info.get("unexpected_keys") or [])
    mismatched = list(loading_info.get("mismatched_keys") or [])
    problems: list[str] = []
    if mismatched:
        problems.append(f"shape mismatches: {sorted(str(m) for m in mismatched)}")
    if frozenset(missing) != plan.allowed_missing:
        extra_missing = sorted(missing - plan.allowed_missing)
        present = sorted(plan.allowed_missing - missing)
        if extra_missing:
            problems.append(f"missing beyond the declared head: {extra_missing}")
        if present:
            problems.append(
                f"declared head keys already in the checkpoint: {present} "
                "(continue an existing classifier with head_init 'existing')"
            )
    stray = sorted(unexpected - plan.allowed_unexpected)
    if stray:
        problems.append(f"unexpected keys: {stray}")
    if problems:
        msg = f"the checkpoint does not load into the declared '{head_init}' head: " + "; ".join(
            problems
        )
        raise HeadLoadError(msg)
    return LoadReceipt(
        head_init=head_init,
        model_type=str(getattr(model.config, "model_type", "")),
        missing=sorted(missing),
        unexpected=sorted(unexpected),
    )


def _checkpoint_files(path: Path) -> dict[str, Path]:
    """Map each tensor name to the safetensors file that holds it (single file or sharded)."""
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map: Any = json.loads(index.read_text()).get("weight_map") or {}
        if not isinstance(weight_map, dict):
            msg = "the checkpoint's safetensors index has no weight_map"
            raise HeadLoadError(msg)
        return {str(k): path / str(v) for k, v in weight_map.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType] - parsed JSON
    single = path / "model.safetensors"
    if not single.is_file():
        msg = "the checkpoint has no safetensors weights"
        raise HeadLoadError(msg)
    safetensors_mod: Any = __import__("safetensors")
    with safetensors_mod.safe_open(str(single), framework="pt") as handle:
        return {str(k): single for k in handle.keys()}  # noqa: SIM118 - safe_open has no __iter__


def read_checkpoint_tensors(path: Path, keys: list[str]) -> dict[str, Any]:
    """Read named tensors straight from a local safetensors checkpoint; absent keys are omitted."""
    files = _checkpoint_files(path)
    safetensors_mod: Any = __import__("safetensors")
    out: dict[str, Any] = {}
    for key in keys:
        source = files.get(key)
        if source is None:
            continue
        with safetensors_mod.safe_open(str(source), framework="pt") as handle:
            out[key] = handle.get_tensor(key)
    return out


def init_decoder_head(
    model: Any,
    checkpoint: Path,
    receipt: LoadReceipt,
    *,
    verbalizer_ids: list[int],
) -> None:
    """Fill the declared decoder head from the checkpoint's LM head, in place.

    ``verbalizer_ids`` are the single-token ids of the labels in label order; row ``k`` of the
    classifier becomes row ``verbalizer_ids[k]`` of the output projection, and its bias entry the
    same row of ``decoder.bias``.
    """
    torch: Any = _torch()
    table = DECODER_LM_HEAD[receipt.model_type]
    head_prefix, lm_prefix = table["head_prefix"], table["lm_head_prefix"]
    head_params = [name for name, _ in model.named_parameters() if name.startswith(head_prefix)]
    sources = [lm_prefix + name[len(head_prefix) :] for name in head_params]
    wanted = [*sources, table["projection"], table["projection_bias"], table["tied_embedding"]]
    tensors = read_checkpoint_tensors(checkpoint, wanted)
    absent = [s for s in sources if s not in tensors]
    if absent:
        msg = f"the checkpoint has no {absent}; the decoder head cannot be initialised from it"
        raise HeadLoadError(msg)
    projection_key = table["projection"]
    if projection_key not in tensors:
        projection_key = table["tied_embedding"]
        if projection_key not in tensors:
            msg = "the checkpoint stores neither the output projection nor the tied embedding"
            raise HeadLoadError(msg)
    projection: Any = tensors[projection_key]
    vocab = int(projection.shape[0])
    if any(i < 0 or i >= vocab for i in verbalizer_ids):
        msg = f"a verbalizer token id is outside the vocabulary of {vocab}"
        raise HeadLoadError(msg)
    bias: Any = tensors.get(table["projection_bias"])

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, source in zip(head_params, sources, strict=True):
            target: Any = params[name]
            value: Any = tensors[source]
            if tuple(value.shape) != tuple(target.shape):
                msg = f"{source} has shape {tuple(value.shape)}, {name} needs {tuple(target.shape)}"
                raise HeadLoadError(msg)
            target.copy_(value.to(target.dtype))
            receipt.initialised_from[name] = source
        index: Any = torch.tensor(verbalizer_ids, dtype=torch.long)
        rows: Any = projection.index_select(0, index)
        classifier: Any = model.classifier
        if bias is not None and classifier.bias is None:
            _add_classifier_bias(model, receipt)
            classifier = model.classifier
        classifier.weight.copy_(rows.to(classifier.weight.dtype))
        receipt.initialised_from["classifier.weight"] = f"{projection_key}[verbalizers]"
        if bias is not None:
            classifier.bias.copy_(bias.index_select(0, index).to(classifier.bias.dtype))
            receipt.initialised_from["classifier.bias"] = f"{table['projection_bias']}[verbalizers]"
        elif classifier.bias is not None:
            classifier.bias.zero_()


def _add_classifier_bias(model: Any, receipt: LoadReceipt) -> None:
    """Give the classifier a bias, and the prediction head's dense layer a frozen zero one.

    ``classifier_bias`` in this architecture's config adds BOTH biases, so a saved config that
    carries the classifier's bias must also carry the dense one; zero and frozen, it changes
    nothing the model computes.
    """
    torch: Any = _torch()
    nn: Any = torch.nn
    old: Any = model.classifier
    new: Any = nn.Linear(old.in_features, old.out_features, bias=True)
    new.to(device=old.weight.device, dtype=old.weight.dtype)
    model.classifier = new
    dense: Any = model.head.dense
    if dense.bias is None:
        replacement: Any = nn.Linear(dense.in_features, dense.out_features, bias=True)
        replacement.to(device=dense.weight.device, dtype=dense.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(dense.weight)
            replacement.bias.zero_()
        replacement.bias.requires_grad_(False)
        model.head.dense = replacement
        receipt.added_parameters.append("head.dense.bias (zero, frozen)")
    receipt.added_parameters.append("classifier.bias")
    model.config.classifier_bias = True


def build_aux_heads(hidden_size: int, *, flag: bool, support: bool, std: float) -> Any:
    """The auxiliary heads as one ``nn.Module`` (``flag`` and/or ``support`` submodules)."""
    torch: Any = _torch()
    nn: Any = torch.nn

    class _AuxHeads(nn.Module):  # pyright: ignore[reportUntypedBaseClass] - torch is lazy
        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType] - torch is lazy
            self.flag: Any = nn.Linear(hidden_size, 1) if flag else None
            self.support: Any = _Bilinear(hidden_size) if support else None
            for module in (self.flag, self.support):
                if module is not None:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    nn.init.zeros_(module.bias)

    class _Bilinear(nn.Module):  # pyright: ignore[reportUntypedBaseClass] - torch is lazy
        def __init__(self, size: int) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType] - torch is lazy
            self.weight: Any = nn.Parameter(torch.empty(size, size))
            self.bias: Any = nn.Parameter(torch.zeros(()))

        def forward(self, pooled: Any, markers: Any) -> Any:
            # pooled [B, H], markers [B, P, H] -> [B, P]
            projected: Any = pooled @ self.weight
            return torch.einsum("bh,bph->bp", projected, markers) + self.bias

    return _AuxHeads()


def aux_outputs(
    torch: Any,
    aux: Any,
    hidden: Any,
    attention_mask: Any,
    *,
    last_token: bool,
    support_positions: Any = None,
) -> dict[str, Any]:
    """The auxiliary heads' logits from the base model's last hidden state.

    Every head reads one pooled vector: the final non-padding token for a decoder (right padding),
    the first token for an encoder. The support head also reads each premise marker's token. The
    trainer and the scorer both call this, so a verifier is scored as it was trained.
    """
    result: dict[str, Any] = {}
    if aux.flag is None and aux.support is None:
        return result
    rows: Any = torch.arange(hidden.shape[0], device=hidden.device)
    position: Any = attention_mask.sum(-1) - 1 if last_token else torch.zeros_like(rows)
    pooled: Any = hidden[rows, position]
    if aux.flag is not None:
        result["flag_logits"] = aux.flag(pooled).squeeze(-1)
    if aux.support is not None and support_positions is not None:
        markers: Any = hidden[rows.unsqueeze(-1), support_positions]
        result["support_logits"] = aux.support(pooled, markers)
    return result


def save_aux_heads(heads: Any, directory: Path) -> Path | None:
    """Write the auxiliary heads' tensors under fixed names; ``None`` when there are none."""
    state: dict[str, Any] = {
        k: v.detach().contiguous().cpu() for k, v in heads.state_dict().items()
    }
    if not state:
        return None
    safetensors_torch: Any = __import__("safetensors.torch", fromlist=["save_file"])
    path = directory / AUX_HEADS_FILENAME
    safetensors_torch.save_file(state, str(path))
    return path


def load_aux_heads(heads: Any, directory: Path) -> list[str]:
    """Load saved auxiliary heads strictly: every declared tensor present, nothing extra."""
    expected = set(heads.state_dict())
    path = directory / AUX_HEADS_FILENAME
    if not expected:
        return []
    if not path.is_file():
        msg = f"the checkpoint has no {AUX_HEADS_FILENAME}, but the run declares {sorted(expected)}"
        raise HeadLoadError(msg)
    safetensors_torch: Any = __import__("safetensors.torch", fromlist=["load_file"])
    state: dict[str, Any] = safetensors_torch.load_file(str(path))
    if set(state) != expected:
        msg = (
            f"{AUX_HEADS_FILENAME} holds {sorted(state)}, the run declares {sorted(expected)}; "
            "a head is never re-initialised silently"
        )
        raise HeadLoadError(msg)
    heads.load_state_dict(state, strict=True)
    return sorted(state)
