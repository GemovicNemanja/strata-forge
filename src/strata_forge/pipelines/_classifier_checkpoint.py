"""Fixed native classifier checkpoint loading, with no remote code or fresh-head exemption."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.pipelines.classifier_spec import ClassifierWorkerError, parse_json
from strata_forge.training._classifier_heads import (
    HeadInitializationReceipt,  # noqa: TC001 — Pydantic resolves this field at runtime
)
from strata_forge.training._load_guard import _guard_loading_result
from strata_forge.training.classifier import (
    ClassifierConfig,
)
from strata_forge.training.classifier import (
    _load as _load_complete_classifier,  # pyright: ignore[reportPrivateUsage] — same fixed SDK reload used by its async public wrapper
)
from strata_forge.training.verifier_config import VerifierConfig

# Native ML/Hub modules expose version-dependent dynamic objects at this lazy installed seam.
type _ML = Any
_SHA = re.compile(r"[a-f0-9]{40}")
_SIDECAR_BYTES = 65_536
_METADATA_BYTES = 32 * 1024 * 1024
_INVENTORY_ENTRIES = 128
_SAFE_NAMES = frozenset(
    {
        "config.json",
        "strata_classifier.json",
        "strata_verifier.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "generation_config.json",
        "model.safetensors.index.json",
    }
)


class _ArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    version: Literal[1]
    implementation: Literal["forge_primary_classifier_v1"]
    config: ClassifierConfig
    initialization: HeadInitializationReceipt
    checkpoint_selection: Literal["final"]
    steps: int = Field(ge=1, le=2**53 - 1)
    trained_rows: int = Field(ge=1, le=100_000)
    overflow_ids: tuple[str, ...] = Field(max_length=32)
    overflow_count: int = Field(ge=0, le=100_000)
    overflow_omitted: int = Field(ge=0, le=100_000)
    environment: dict[str, str | None]

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError()
        return value

    @model_validator(mode="after")
    def _binding(self) -> Self:
        if (
            self.config.labels != self.initialization.labels
            or self.overflow_count != len(self.overflow_ids) + self.overflow_omitted
            or len(set(self.overflow_ids)) != len(self.overflow_ids)
            or any(not item.strip() or len(item) > 1024 for item in self.overflow_ids)
        ):
            raise ValueError()
        return self


def _artifact_metadata(value: object) -> _ArtifactMetadata:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(raw) > 65_536:
        raise ValueError()
    return _ArtifactMetadata.model_validate_json(raw)


@dataclass(frozen=True, repr=False)
class NativeClassifierBase:
    model: _ML
    tokenizer: _ML


def _metadata(path: Path, cap: int) -> object:
    descriptor = os.open(path.resolve(strict=True), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ClassifierWorkerError("classifier_artifact_invalid")
        return parse_json(stream.read(cap + 1), cap)


def _local_files(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ClassifierWorkerError("classifier_checkpoint_invalid")
    files = list(path.iterdir())
    if not files or len(files) > 128:
        raise ClassifierWorkerError("classifier_checkpoint_invalid")
    for file in files:
        if file.is_symlink() or not file.is_file():
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
        if file.name not in _SAFE_NAMES and not re.fullmatch(
            r"[A-Za-z0-9_.-]+\.safetensors", file.name
        ):
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
        if file.suffix != ".safetensors" and file.stat().st_size > 32 * 1024 * 1024:
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
    if not (path / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ClassifierWorkerError("classifier_checkpoint_invalid")
    names = {file.name for file in files}
    index = path / "model.safetensors.index.json"
    if index.exists():
        value = _metadata(index, 32 * 1024 * 1024)
        if type(value) is not dict:
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
        mapping = cast("dict[str,object]", value).get("weight_map")
        if type(mapping) is not dict or not mapping:
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
        if any(
            type(name) is not str or name not in names or not name.endswith(".safetensors")
            for name in cast("dict[str,object]", mapping).values()
        ):
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
    tokenizer = path / "tokenizer_config.json"
    if tokenizer.exists():
        value = _metadata(tokenizer, 32 * 1024 * 1024)
        if type(value) is not dict:
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
        for key, child in cast("dict[str,object]", value).items():
            if (
                key == "_name_or_path"
                or not key.endswith(("_file", "_files", "_path", "_paths"))
                or child is None
            ):
                continue
            if type(child) is not str or child not in names or child not in _SAFE_NAMES:
                raise ClassifierWorkerError("classifier_checkpoint_invalid")


def _base(path: Path) -> NativeClassifierBase:
    result: NativeClassifierBase | None = None
    try:
        _local_files(path)
        transformers: _ML = __import__("transformers")
        config = transformers.AutoConfig.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        factory = {
            "bert": transformers.BertModel,
            "gpt2": transformers.GPT2LMHeadModel,
            "modernbert-decoder": transformers.ModernBertDecoderForCausalLM,
        }.get(config.model_type)
        if factory is None:
            raise ValueError()
        model, _ = _guard_loading_result(
            factory.from_pretrained(
                path,
                config=config,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                output_loading_info=True,
            )
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError()
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        result = NativeClassifierBase(model=model, tokenizer=tokenizer)
    except ImportError:
        raise ImportError("classifier worker requires strata-forge[finetuning,storage]") from None
    except Exception:
        result = None
    if result is None:
        raise ClassifierWorkerError("classifier_base_invalid")
    return result


async def load_classifier_base(path: Path) -> NativeClassifierBase:
    return await asyncio.to_thread(_base, path)


def _saved(path: Path) -> NativeClassifierBase:
    result: NativeClassifierBase | None = None
    try:
        _local_files(path)
        value = _metadata(path / "strata_classifier.json", 65_536)
        if type(value) is not dict:
            raise ValueError()
        # The SDK still owns complete native loading; typed metadata is not a head exemption.
        metadata = _artifact_metadata(cast("dict[str, object]", value))
        config = metadata.config
        rendering = path / "strata_verifier.json"
        if config.rendering is None:
            if rendering.exists():
                raise ValueError()
        elif (
            VerifierConfig.model_validate(_metadata(rendering, 65_536)).digest()
            != config.rendering.digest()
        ):
            raise ValueError()
        transformers: _ML = __import__("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        if config.rendering is not None and config.rendering.label_tokens is not None:
            ids = [
                tokenizer.encode(config.rendering.label_tokens[label], add_special_tokens=False)
                for label in config.labels
            ]
            if (
                metadata.initialization.initializer_id != "decoder_logit_head"
                or any(len(item) != 1 for item in ids)
                or tuple(item[0] for item in ids) != metadata.initialization.verbalizer_ids
            ):
                raise ValueError()
        loaded = _load_complete_classifier(path)
        result = NativeClassifierBase(model=loaded.model, tokenizer=tokenizer)
    except ImportError:
        raise ImportError("classifier worker requires strata-forge[finetuning,storage]") from None
    except Exception:
        result = None
    if result is None:
        raise ClassifierWorkerError("classifier_artifact_invalid")
    return result


def validate_single_sequence_rendering(
    metadata: object,
    native_config: object,
    *,
    original: VerifierConfig,
    effective: VerifierConfig,
) -> None:
    """A recognized pinned declaration enables no head exemption or render override drift."""
    valid = False
    try:
        if type(metadata) is not dict or type(native_config) is not dict:
            raise ValueError()
        manifest = _artifact_metadata(cast("dict[str, object]", metadata))
        native = cast("dict[str, object]", native_config)
        config, receipt = manifest.config, manifest.initialization
        if (
            config.rendering is None
            or original.pair_template is not None
            or config.rendering.digest() != original.digest()
            or effective.digest() != original.digest()
            or config.labels != receipt.labels
            or config.labels != original.labels
            or config.max_length != original.max_length
            or native.get("model_type") != receipt.family
            or native.get("id2label") != {str(i): label for i, label in enumerate(config.labels)}
            or native.get("architectures")
            != [
                "BertForSequenceClassification"
                if receipt.family == "bert"
                else "ForgeDecoderClassifier"
            ]
        ):
            raise ValueError()
        valid = True
    except Exception:
        valid = False
    if not valid:
        raise ClassifierWorkerError("classifier_rendering_invalid")


async def reload_classifier(path: Path) -> NativeClassifierBase:
    return await asyncio.to_thread(_saved, path)


def _preflight(hub: _ML, model_id: str, name: str, revision: str, token: _ML) -> int:
    info = hub.hf_hub_download(model_id, name, revision=revision, token=token, dry_run=True)
    maximum = _SIDECAR_BYTES if name.startswith("strata_") else _METADATA_BYTES
    if (
        not isinstance(info, hub.DryRunFileInfo)
        or info.commit_hash != revision
        or info.filename != name
        or type(info.file_size) is not int
        or not 0 <= info.file_size <= maximum
    ):
        raise ValueError()
    return info.file_size


def _copy_metadata(source: Path, target: Path, expected: int, maximum: int) -> None:
    descriptor = os.open(source.resolve(strict=True), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected:
            raise ValueError()
        raw = stream.read(maximum + 1)
    if len(raw) != expected or len(raw) > maximum:
        raise ValueError()
    with target.open("xb") as stream:
        stream.write(raw)


def _inventory(hub: _ML, model_id: str, revision: str, token: _ML) -> list[str]:
    entries = iter(
        hub.HfApi(token=token).list_repo_tree(
            model_id, revision=revision, repo_type="model", recursive=True, expand=True
        )
    )
    selected: list[str] = []
    seen: set[str] = set()
    try:
        for count, entry in enumerate(entries, 1):
            if count > _INVENTORY_ENTRIES or not isinstance(entry, (hub.RepoFile, hub.RepoFolder)):
                raise ValueError()
            name = entry.path
            if (
                type(name) is not str
                or not name
                or len(name) > 512
                or name.startswith("/")
                or "\\" in name
                or "\x00" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
                or name in seen
            ):
                raise ValueError()
            seen.add(name)
            if isinstance(entry, hub.RepoFile) and (
                name in _SAFE_NAMES or re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name)
            ):
                selected.append(name)
    finally:
        close = getattr(entries, "close", None)
        if close is not None:
            close()
    return selected


def probe_forge_classifier(
    model_id: str, *, revision: str, token: str | Literal[False]
) -> NativeClassifierBase | None:
    """Only a definite missing sidecar selects native HF loading; every other failure denies."""
    result: NativeClassifierBase | None = None
    failed = False
    try:
        if not _SHA.fullmatch(revision):
            raise ValueError()
        hub: _ML = __import__("huggingface_hub")
        errors: _ML = __import__("huggingface_hub.errors", fromlist=["RemoteEntryNotFoundError"])
        try:
            sidecar_size = _preflight(hub, model_id, "strata_classifier.json", revision, token)
        except errors.RemoteEntryNotFoundError as error:
            if error.response is not None and error.response.status_code == 404:
                return None
            raise ValueError() from None
        with TemporaryDirectory(prefix="strata-classifier-") as directory:
            root = Path(directory)
            sidecar = hub.hf_hub_download(
                model_id, "strata_classifier.json", revision=revision, token=token
            )
            _copy_metadata(
                Path(sidecar), root / "strata_classifier.json", sidecar_size, _SIDECAR_BYTES
            )
            _artifact_metadata(_metadata(root / "strata_classifier.json", _SIDECAR_BYTES))
            selected = _inventory(hub, model_id, revision, token)
            if "strata_classifier.json" not in selected:
                raise ValueError()
            # Preflight every metadata asset before downloading any remaining assets/weights.
            sizes = {
                name: _preflight(hub, model_id, name, revision, token)
                for name in selected
                if name != "strata_classifier.json" and not name.endswith(".safetensors")
            }
            for name in selected:
                if name == "strata_classifier.json":
                    continue
                downloaded = hub.hf_hub_download(model_id, name, revision=revision, token=token)
                if name.endswith(".safetensors"):
                    shutil.copyfile(downloaded, root / name)
                else:
                    maximum = _SIDECAR_BYTES if name.startswith("strata_") else _METADATA_BYTES
                    _copy_metadata(Path(downloaded), root / name, sizes[name], maximum)
            result = _saved(root)
    except Exception:
        failed = True
    if failed:
        raise ClassifierWorkerError("classifier_artifact_unavailable")
    return result
