"""Local auxiliary verifier SDK with explicit shared-primary ownership and strict reload."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from strata_forge.core.repro import set_seed
from strata_forge.training._classifier_heads import (
    HeadInitializationReceipt,
    InitializedClassifier,
    classifier_modules,
)
from strata_forge.training._load_guard import _guard_loading_result
from strata_forge.training._verifier_heads import (
    VerifierForward,
    auxiliary_keys,
    verifier_classifier_class,
)
from strata_forge.training.verifier_geometry import (
    VerifierExample,
    VerifierGeometryConfig,
    VerifierGeometryError,
    render_verifier_geometry,
    resolve_pid_markers,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# Native tensors/tokenizers are optional dynamic library objects, never serialized callbacks.
type _ML = Any


class VerifierPrediction(TypedDict):
    """Local prediction values; absent optional heads are explicit nulls."""

    item_id: str
    probabilities: dict[str, float]
    p_inconsistent: float | None
    support_scores: dict[str, float] | None


class AuxiliaryInitializationReceipt(BaseModel):
    """Exact local initialization metadata, not a loading exemption or source attestation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    version: Literal[1] = 1
    primary: HeadInitializationReceipt
    auxiliary_keys: tuple[str, ...]
    seed: int = Field(ge=0, le=2**32 - 1)
    initializer_range: float = Field(ge=0, allow_inf_nan=False)
    primary_state_key_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    primary_state_key_count: int = Field(ge=1)
    ownership: Literal["shared_primary"] = "shared_primary"

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("verifier_receipt_invalid")
        return value


@dataclass(frozen=True, repr=False)
class InitializedVerifierClassifier:
    """Caller-owned native model; shared primary aliases must no longer be used independently."""

    model: _ML
    geometry: VerifierGeometryConfig
    receipt: AuxiliaryInitializationReceipt
    tokenizer: _ML = None


@dataclass(frozen=True, repr=False)
class VerifierBatch:
    """Resolved geometry and mutable native tensors; freezing this wrapper does not freeze tensors."""

    geometry_sha256: str
    item_ids: tuple[str, ...]
    model_inputs: dict[str, _ML]
    marker_positions: tuple[tuple[int, ...], ...]
    premise_ids: tuple[tuple[str, ...], ...]

    def forward_kwargs(self) -> dict[str, _ML]:
        """Use the exact resolved geometry in an ordinary differentiable native forward."""
        return {
            **self.model_inputs,
            "marker_positions": self.marker_positions,
            "premise_ids": self.premise_ids,
        }


def _initialize(
    primary: InitializedClassifier, geometry: VerifierGeometryConfig, seed: int, ownership: str
) -> InitializedVerifierClassifier:
    if (
        type(primary) is not InitializedClassifier
        or ownership != "share"
        or type(seed) is not int
        or seed != primary.receipt.seed
    ):
        raise ValueError()
    geometry = VerifierGeometryConfig.model_validate(geometry)
    receipt = HeadInitializationReceipt.model_validate(
        {name: getattr(primary.receipt, name) for name in HeadInitializationReceipt.model_fields}
    )
    if geometry.labels != receipt.labels or (geometry.pooling == "cls") != (
        receipt.family == "bert"
    ):
        raise ValueError()
    config = copy.deepcopy(primary.model.config)
    if getattr(config, "auto_map", None):
        raise ValueError()
    if (
        config.model_type != receipt.family
        or tuple(config.id2label[index] for index in range(config.num_labels)) != geometry.labels
    ):
        raise ValueError()
    body = (
        primary.model.bert.state_dict()
        if receipt.family == "bert"
        else primary.model.backbone.state_dict()
    )
    prefix = "bert." if receipt.family == "bert" else "backbone."
    if (
        len(body) != receipt.body_key_count
        or _key_hash(body) != receipt.body_key_sha256
        or set(primary.model.state_dict())
        != {prefix + key for key in body} | set(receipt.created_keys)
    ):
        raise ValueError()
    config.strata_verifier_outputs = list(geometry.outputs)
    config.strata_verifier_geometry_sha256 = geometry.digest()
    config.strata_verifier_schema_version = 2
    config.strata_verifier_implementation = "forge_auxiliary_verifier_v1"
    before = primary.model.state_dict()
    parameters = dict(primary.model.named_parameters())
    set_seed(seed)
    model = verifier_classifier_class()(config, primary=primary.model)
    keys = auxiliary_keys(geometry.outputs)
    if set(model.state_dict()) != set(before) | set(keys):
        raise ValueError()
    after = dict(model.named_parameters())
    if any(after.get(name) is not parameter for name, parameter in parameters.items()):
        raise ValueError()
    model.training = primary.model.training
    return InitializedVerifierClassifier(
        model,
        geometry,
        AuxiliaryInitializationReceipt(
            primary=receipt,
            auxiliary_keys=keys,
            seed=seed,
            initializer_range=float(config.initializer_range),
            primary_state_key_sha256=_key_hash(before),
            primary_state_key_count=len(before),
        ),
    )


async def initialize_verifier_classifier(
    primary: InitializedClassifier,
    *,
    geometry: VerifierGeometryConfig,
    seed: int,
    ownership: Literal["share"],
) -> InitializedVerifierClassifier:
    """Share existing primary modules explicitly; mode/device/optimizer mutations affect aliases."""
    result = None
    try:
        result = await asyncio.to_thread(_initialize, primary, geometry, seed, ownership)
    except ImportError:
        raise ImportError(
            "Install 'strata-forge[finetuning]' for auxiliary verifier models."
        ) from None
    except Exception:
        result = None
    if result is None:
        raise VerifierGeometryError("verifier_initialization_invalid")
    return result


def _encode(
    tokenizer: _ML, examples: tuple[VerifierExample, ...], geometry: VerifierGeometryConfig
) -> VerifierBatch:
    geometry = VerifierGeometryConfig.model_validate(geometry)
    if (
        type(examples) is not tuple
        or not examples
        or len(examples) > 1024
        or tokenizer.is_fast is not True
    ):
        raise VerifierGeometryError("verifier_tokenizer_unsupported")
    examples = tuple(VerifierExample.model_validate(example) for example in examples)
    if len({row.item_id for row in examples}) != len(examples):
        raise VerifierGeometryError("verifier_batch_invalid")
    rendered = tuple(render_verifier_geometry(geometry, example) for example in examples)
    kwargs: dict[str, _ML] = {
        "padding": True,
        "truncation": False,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
        "return_tensors": "pt",
    }
    texts = [row.text for row in rendered]
    if geometry.pooling == "cls":
        encoded = tokenizer(texts, text_pair=[row.text_pair for row in rendered], **kwargs)
    else:
        encoded = tokenizer(texts, **kwargs)
    ids, attention = encoded["input_ids"], encoded["attention_mask"]
    if (
        ids.ndim != 2
        or attention.shape != ids.shape
        or ids.shape[0] != len(examples)
        or not bool(((attention == 0) | (attention == 1)).all())
        or not bool((attention.sum(-1) > 0).all())
    ):
        raise VerifierGeometryError("verifier_batch_invalid")
    if ids.shape[1] > geometry.max_length or any(
        int(value) > geometry.max_length for value in attention.sum(-1)
    ):
        raise VerifierGeometryError("verifier_overflow")
    if geometry.pooling == "cls" and (
        type(tokenizer.cls_token_id) is not int
        or not bool((ids[:, 0] == tokenizer.cls_token_id).all())
        or not bool((attention[:, 0] == 1).all())
    ):
        raise VerifierGeometryError("verifier_tokenizer_unsupported")
    markers = tuple(
        resolve_pid_markers(
            row,
            offsets=encoded["offset_mapping"][index].tolist(),
            sequence_ids=encoded.sequence_ids(index),
            special_tokens_mask=encoded["special_tokens_mask"][index].tolist(),
            attention_mask=attention[index].tolist(),
        )
        for index, row in enumerate(rendered)
    )
    inputs = {"input_ids": ids, "attention_mask": attention}
    if geometry.pooling == "cls" and "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"]
    return VerifierBatch(
        geometry.digest(),
        tuple(row.item_id for row in examples),
        inputs,
        markers,
        tuple(row.premise_ids for row in rendered),
    )


async def encode_verifier_batch(
    tokenizer: _ML, examples: tuple[VerifierExample, ...], *, geometry: VerifierGeometryConfig
) -> VerifierBatch:
    """Resolve actual fast-tokenizer offsets without truncating or dropping any row."""
    result = None
    code = "verifier_batch_invalid"
    try:
        result = await asyncio.to_thread(_encode, tokenizer, examples, geometry)
    except VerifierGeometryError as error:
        code = str(error)
    except Exception:
        result = None
    if result is None:
        raise VerifierGeometryError(code)
    return result


def _score(
    initialized: InitializedVerifierClassifier, batch: VerifierBatch
) -> tuple[VerifierPrediction, ...]:
    torch, _ = classifier_modules()
    if type(initialized.model) is not verifier_classifier_class():
        raise ValueError()
    _validate_binding(initialized.model.config, initialized.geometry, initialized.receipt)
    if batch.geometry_sha256 != initialized.geometry.digest():
        raise ValueError()
    modes = tuple((module, module.training) for module in initialized.model.modules())
    try:
        initialized.model.eval()
        with torch.no_grad():
            output: VerifierForward = initialized.model(**batch.forward_kwargs())
            tensors = [
                output.logits,
                *(() if output.inconsistent_logit is None else (output.inconsistent_logit,)),
                *(() if output.support_logits is None else output.support_logits),
            ]
            if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
                raise ValueError()
            probabilities = output.logits.float().softmax(-1).cpu().tolist()
            flags = (
                None
                if output.inconsistent_logit is None
                else output.inconsistent_logit.float().sigmoid().cpu().tolist()
            )
            supports = (
                None
                if output.support_logits is None
                else [value.float().cpu().tolist() for value in output.support_logits]
            )
    finally:
        for module, mode in modes:
            module.training = mode
    return tuple(
        {
            "item_id": item,
            "probabilities": dict(
                zip(initialized.geometry.labels, probabilities[index], strict=True)
            ),
            "p_inconsistent": None if flags is None else flags[index],
            "support_scores": None
            if supports is None
            else dict(zip(batch.premise_ids[index], supports[index], strict=True)),
        }
        for index, item in enumerate(batch.item_ids)
    )


async def score_verifier_batch(
    initialized: InitializedVerifierClassifier, batch: VerifierBatch
) -> tuple[VerifierPrediction, ...]:
    """Return local model scores; support values are raw logits and absent heads return null."""
    result = None
    try:
        result = await asyncio.to_thread(_score, initialized, batch)
    except ImportError:
        raise ImportError(
            "Install 'strata-forge[finetuning]' for auxiliary verifier models."
        ) from None
    except Exception:
        result = None
    if result is None:
        raise VerifierGeometryError("verifier_output_invalid")
    return result


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _metadata(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError()
    with path.open("rb") as file:
        raw = file.read(65_537)
    if len(raw) > 65_536:
        raise ValueError()
    value: object = json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError())
    )
    if type(value) is not dict:
        raise ValueError()
    return cast("dict[str, object]", value)


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_777_216:
        raise ValueError()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key_hash(keys: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(keys), separators=(",", ":")).encode()).hexdigest()


def _validate_state(model: _ML, receipt: AuxiliaryInitializationReceipt) -> None:
    state = model.state_dict()
    body = set(state) - set(receipt.auxiliary_keys)
    hidden = model.config.hidden_size
    expected = {
        "inconsistency_head.weight": (1, hidden),
        "inconsistency_head.bias": (1,),
        "support_head.weight": (hidden, hidden),
        "support_head.bias": (1,),
    }
    if (
        len(body) != receipt.primary_state_key_count
        or _key_hash(body) != receipt.primary_state_key_sha256
        or any(
            name not in state or tuple(state[name].shape) != expected[name]
            for name in receipt.auxiliary_keys
        )
    ):
        raise ValueError()


def _export(initialized: InitializedVerifierClassifier, tokenizer: _ML, path: Path) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError()
    path.mkdir(parents=True, exist_ok=True)
    geometry, receipt, model = initialized.geometry, initialized.receipt, initialized.model
    _validate_binding(model.config, geometry, receipt)
    _validate_state(model, receipt)
    if type(model) is not verifier_classifier_class() or tokenizer.is_fast is not True:
        raise ValueError()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if not (path / "model.safetensors").is_file() or not (path / "tokenizer.json").is_file():
        raise ValueError()
    tokenizer_files = {
        name: _file_hash(path / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (path / name).exists()
    }
    manifest = {
        "version": 1,
        "implementation": "forge_auxiliary_verifier_v1",
        "geometry_sha256": geometry.digest(),
        "initialization": receipt.model_dump(mode="json"),
        "tokenizer_sha256": tokenizer_files,
    }
    (path / "strata_verifier.json").write_text(geometry.model_dump_json(indent=2) + "\n")
    raw = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(raw.encode("utf-8")) > 65_536:
        raise ValueError()
    (path / "strata_classifier.json").write_text(raw)


def _validate_binding(
    config: _ML, geometry: VerifierGeometryConfig, receipt: AuxiliaryInitializationReceipt
) -> None:
    if (
        type(receipt.version) is not int
        or receipt.version != 1
        or receipt.seed != receipt.primary.seed
        or receipt.auxiliary_keys != auxiliary_keys(geometry.outputs)
        or geometry.labels != receipt.primary.labels
        or config.model_type != receipt.primary.family
        or (config.model_type == "bert") != (geometry.pooling == "cls")
        or tuple(config.id2label[index] for index in range(config.num_labels)) != geometry.labels
        or config.strata_verifier_geometry_sha256 != geometry.digest()
        or type(config.strata_verifier_schema_version) is not int
        or config.strata_verifier_schema_version != 2
        or config.strata_verifier_implementation != "forge_auxiliary_verifier_v1"
        or tuple(config.strata_verifier_outputs) != geometry.outputs
        or float(config.initializer_range) != receipt.initializer_range
        or getattr(config, "auto_map", None)
    ):
        raise ValueError()


def _load(path: Path) -> InitializedVerifierClassifier:
    manifest = _metadata(path / "strata_classifier.json")
    if (
        set(manifest)
        != {"version", "implementation", "geometry_sha256", "initialization", "tokenizer_sha256"}
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or manifest["implementation"] != "forge_auxiliary_verifier_v1"
    ):
        raise ValueError()
    raw_geometry = _metadata(path / "strata_verifier.json")
    if set(raw_geometry) != set(VerifierGeometryConfig.model_fields):
        raise ValueError()
    geometry = VerifierGeometryConfig.model_validate_json(json.dumps(raw_geometry))
    raw_receipt = manifest["initialization"]
    if type(raw_receipt) is not dict or set(cast("dict[str, object]", raw_receipt)) != set(
        AuxiliaryInitializationReceipt.model_fields
    ):
        raise ValueError()
    receipt = AuxiliaryInitializationReceipt.model_validate_json(json.dumps(raw_receipt))
    if manifest["geometry_sha256"] != geometry.digest():
        raise ValueError()
    hashes = manifest["tokenizer_sha256"]
    if (
        type(hashes) is not dict
        or not {"tokenizer.json", "tokenizer_config.json"} <= hashes.keys()
        or not hashes.keys()
        <= {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    ):
        raise ValueError()
    typed_hashes = cast("dict[str, object]", hashes)
    if any(
        type(digest) is not str or _file_hash(path / name) != digest
        for name, digest in typed_hashes.items()
    ):
        raise ValueError()
    for name in ("config.json", "model.safetensors"):
        if (path / name).is_symlink() or not (path / name).is_file():
            raise ValueError()
    tokenizer_config = _metadata(path / "tokenizer_config.json")
    if tokenizer_config.get("auto_map"):
        raise ValueError()
    for key in (
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "special_tokens_map_file",
        "added_tokens_file",
    ):
        value = tokenizer_config.get(key)
        if value is not None and (type(value) is not str or value not in typed_hashes):
            raise ValueError()
    if tokenizer_config.get("tokenizer_class") not in (
        "TokenizersBackend",
        "PreTrainedTokenizerFast",
        "BertTokenizer",
        "BertTokenizerFast",
        "GPT2Tokenizer",
        "GPT2TokenizerFast",
    ):
        raise ValueError()
    native = _metadata(path / "config.json")
    if native.get("architectures") != ["ForgeAuxiliaryVerifier"]:
        raise ValueError()
    _, transformers = classifier_modules()
    config = transformers.AutoConfig.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, token=False
    )
    _validate_binding(config, geometry, receipt)
    model, _ = _guard_loading_result(
        verifier_classifier_class().from_pretrained(
            path,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            token=False,
            output_loading_info=True,
        )
    )
    _validate_state(model, receipt)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        tokenizer_file=str(path / "tokenizer.json"),
        token=False,
    )
    if tokenizer.is_fast is not True:
        raise ValueError()
    return InitializedVerifierClassifier(model, geometry, receipt, tokenizer)


async def export_verifier_classifier(
    initialized: InitializedVerifierClassifier, tokenizer: _ML, path: Path
) -> None:
    """Write a fresh local safe artifact; this is not a training or publication receipt."""
    failed = False
    try:
        await asyncio.to_thread(_export, initialized, tokenizer, path)
    except ImportError:
        raise ImportError(
            "Install 'strata-forge[finetuning]' for auxiliary verifier models."
        ) from None
    except Exception:
        failed = True
    if failed:
        raise VerifierGeometryError("verifier_artifact_invalid")


async def load_verifier_classifier(path: Path) -> InitializedVerifierClassifier:
    """Reload a complete local artifact; no initializer exemption or native fallback is allowed."""
    result = None
    try:
        result = await asyncio.to_thread(_load, path)
    except ImportError:
        raise ImportError(
            "Install 'strata-forge[finetuning]' for auxiliary verifier models."
        ) from None
    except Exception:
        result = None
    if result is None:
        raise VerifierGeometryError("verifier_artifact_invalid")
    return result
