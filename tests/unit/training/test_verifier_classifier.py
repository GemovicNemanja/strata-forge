"""Tiny random CPU optimizer/export/reload controls; no model downloads."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from strata_forge.training._classifier_heads import initialize_classifier
from strata_forge.training._verifier_heads import auxiliary_keys
from strata_forge.training.verifier_classifier import (
    encode_verifier_batch,
    export_verifier_classifier,
    initialize_verifier_classifier,
    load_verifier_classifier,
    score_verifier_batch,
)
from strata_forge.training.verifier_geometry import (
    VerifierExample,
    VerifierGeometryError,
    VerifierPremise,
)
from strata_forge.training.verifier_loss import (
    ResearchLossRow,
    ThreeWayTarget,
    research_verifier_loss,
)
from tests.unit.training.test_classifier import stack
from tests.unit.training.test_verifier_geometry import geometry


def tokenizer(encoder: bool) -> Any:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "p1": 4,
                "p2": 5,
                ":": 6,
                "a": 7,
                "b": 8,
                "q": 9,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    if encoder:
        backend.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask", "token_type_ids"]
        if encoder
        else ["input_ids", "attention_mask"],
    )


async def initialized(
    family: str, outputs: tuple[str, ...] = ("label", "inconsistent", "support_set")
) -> tuple[Any, Any, Any, Any]:
    torch, transformers = stack()
    if torch is None:
        return None, None, None, None
    torch.set_num_threads(1)
    if family == "bert":
        base = transformers.BertModel(
            transformers.BertConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=256,
            )
        )
    elif family == "gpt2":
        base = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=32,
                n_embd=8,
                n_layer=1,
                n_head=2,
                n_positions=256,
                pad_token_id=0,
                eos_token_id=0,
            )
        )
    else:
        base = transformers.ModernBertDecoderForCausalLM(
            transformers.ModernBertDecoderConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                intermediate_size=16,
                max_position_embeddings=256,
                pad_token_id=0,
                eos_token_id=0,
                decoder_bias=True,
                attn_implementation="eager",
            )
        )
    if family == "modernbert-decoder":
        with torch.no_grad():
            base.decoder.bias.copy_(torch.linspace(-3.0, 3.0, 32))
    equivalence = {
        "input_ids": torch.tensor([[7, 8, 0], [7, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
    }
    primary = await initialize_classifier(
        base,
        labels=("entailed", "contradicted", "undetermined"),
        initializer_id="encoder_random_head" if family == "bert" else "decoder_logit_head",
        verbalizer_ids=() if family == "bert" else (10, 11, 12),
        seed=13,
        equivalence_inputs=None if family == "bert" else equivalence,
    )
    if family != "bert":
        base.eval()
        primary.model.eval()
        with torch.no_grad():
            attention = equivalence["attention_mask"]
            positions = (torch.arange(attention.shape[1]) * attention).max(-1).values
            expected_logits = base(**equivalence).logits[
                torch.arange(attention.shape[0]), positions
            ][:, [10, 11, 12]]
            assert torch.allclose(
                primary.model(**equivalence).logits, expected_logits, atol=1e-6, rtol=1e-5
            )
        if family == "modernbert-decoder":
            assert primary.receipt.bias_copied is True
            assert torch.equal(primary.model.classifier.bias, base.decoder.bias[[10, 11, 12]])
            assert bool((primary.model.classifier.bias != 0).all())
    config = geometry("cls" if family == "bert" else "answer_position", outputs=outputs)
    verifier = await initialize_verifier_classifier(
        primary, geometry=config, seed=13, ownership="share"
    )
    return torch, primary, verifier, tokenizer(family == "bert")


def examples() -> tuple[VerifierExample, ...]:
    return (
        VerifierExample(
            item_id="one",
            premises=(
                VerifierPremise(pid="p1", text="a é 😀"),
                VerifierPremise(pid="p2", text="b"),
            ),
            hypothesis="q",
        ),
        VerifierExample(
            item_id="two", premises=(VerifierPremise(pid="p1", text="a"),), hypothesis="q"
        ),
    )


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert-decoder"])
async def test_tiny_full_step_export_reload_and_sharing(family: str, tmp_path: Path) -> None:
    torch, primary, verifier, tok = await initialized(family)
    if torch is None:
        return
    assert verifier.model.config is not primary.model.config
    original_keys = set(primary.model.state_dict())
    assert set(verifier.model.state_dict()) == original_keys | set(
        auxiliary_keys(verifier.geometry.outputs)
    )
    shared = dict(verifier.model.named_parameters())
    assert all(shared[name] is parameter for name, parameter in primary.model.named_parameters())
    batch = await encode_verifier_batch(tok, examples(), geometry=verifier.geometry)
    primary.model.eval()
    verifier.model.eval()
    with torch.no_grad():
        expected = primary.model(**batch.model_inputs).logits
        actual = verifier.model(**batch.forward_kwargs()).logits
        assert torch.equal(expected, actual)
    assert set(primary.model.state_dict()) == original_keys
    output = verifier.model(**batch.forward_kwargs())
    targets = (
        ResearchLossRow(
            primary=ThreeWayTarget(label_dist=(1.0, 0.0, 0.0)),
            flag_target=False,
            premise_ids=("p1", "p2"),
            support_set=("p2",),
        ),
        ResearchLossRow(
            primary=ThreeWayTarget(label_dist=(0.0, 1.0, 0.0)),
            flag_target=True,
            premise_ids=("p1",),
            support_set=(),
        ),
    )
    loss = research_verifier_loss(
        output.logits,
        class_order=("E", "C", "U"),
        rows=targets,
        flag_logits=output.inconsistent_logit,
        support_logits=output.support_logits,
    )
    optimizer = torch.optim.SGD(verifier.model.parameters(), lr=0.01)
    before = verifier.model.support_head.weight.detach().clone()
    loss.total.backward()
    optimizer.step()
    assert not torch.equal(before, verifier.model.support_head.weight)
    scores = await score_verifier_batch(verifier, batch)
    first_support, second_support = scores[0]["support_scores"], scores[1]["support_scores"]
    assert first_support is not None
    assert second_support is not None
    assert set(first_support) == {"p1", "p2"}
    assert set(second_support) == {"p1"}
    path = tmp_path / family
    await export_verifier_classifier(verifier, tok, path)
    restored = await load_verifier_classifier(path)
    from transformers import AutoTokenizer

    restored_tok = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    restored_batch = await encode_verifier_batch(
        restored_tok, examples(), geometry=restored.geometry
    )
    assert await score_verifier_batch(restored, restored_batch) == scores
    from safetensors.torch import load_file

    tensors = load_file(path / "model.safetensors")
    assert tensors["inconsistency_head.weight"].shape == (1, 8)
    assert tensors["inconsistency_head.bias"].shape == (1,)
    assert tensors["support_head.weight"].shape == (8, 8)
    assert tensors["support_head.bias"].shape == (1,)


async def test_headless_nulls_overflow_and_optional_target_refusal() -> None:
    torch, _, verifier, tok = await initialized("gpt2", ("label",))
    if torch is None:
        return
    batch = await encode_verifier_batch(tok, examples(), geometry=verifier.geometry)
    scores = await score_verifier_batch(verifier, batch)
    assert all(row["p_inconsistent"] is None and row["support_scores"] is None for row in scores)
    tiny = verifier.geometry.model_copy(update={"max_length": 1})
    with pytest.raises(VerifierGeometryError, match=r"^verifier_overflow$"):
        await encode_verifier_batch(tok, examples(), geometry=tiny)
    bad = (examples()[0].model_copy(update={"support_set": ("p99",)}),)
    with pytest.raises(VerifierGeometryError):
        await encode_verifier_batch(tok, bad, geometry=verifier.geometry)


@pytest.mark.parametrize(
    "damage", ["body", "flag", "support", "shape", "unknown", "metadata", "tokenizer", "bin_only"]
)
async def test_incomplete_or_inconsistent_artifacts_refuse(damage: str, tmp_path: Path) -> None:
    torch, _, verifier, tok = await initialized("gpt2")
    if torch is None:
        return
    path = tmp_path / "artifact"
    await export_verifier_classifier(verifier, tok, path)
    from safetensors.torch import load_file, save_file

    file = path / "model.safetensors"
    tensors = load_file(file)
    if damage in {"body", "flag", "support", "shape", "unknown"}:
        if damage == "body":
            del tensors[next(key for key in tensors if key.startswith("backbone."))]
        elif damage == "flag":
            del tensors["inconsistency_head.weight"]
        elif damage == "support":
            del tensors["support_head.bias"]
        elif damage == "shape":
            tensors["support_head.weight"] = torch.zeros(2, 2)
        else:
            tensors["unknown_head.weight"] = torch.zeros(1)
        save_file(tensors, file, metadata={"format": "pt"})
    elif damage == "metadata":
        body = json.loads((path / "strata_verifier.json").read_text())
        body["outputs"] = ["label"]
        (path / "strata_verifier.json").write_text(json.dumps(body))
    elif damage == "tokenizer":
        (path / "tokenizer.json").write_text("{}")
    else:
        file.rename(path / "pytorch_model.bin")
    with pytest.raises(VerifierGeometryError, match=r"^verifier_artifact_invalid$") as error:
        await load_verifier_classifier(path)
    assert error.value.__context__ is error.value.__cause__ is None


@pytest.mark.parametrize(
    "outputs", [("label",), ("label", "inconsistent"), ("label", "support_set")]
)
async def test_optional_head_roundtrips_and_empty_premises(
    outputs: tuple[str, ...], tmp_path: Path
) -> None:
    torch, _, verifier, tok = await initialized("gpt2", outputs)
    if torch is None:
        return
    empty = (VerifierExample(item_id="empty", premises=(), hypothesis="q"),)
    batch = await encode_verifier_batch(tok, empty, geometry=verifier.geometry)
    expected = await score_verifier_batch(verifier, batch)
    assert (
        expected[0]["p_inconsistent"] is None
        if "inconsistent" not in outputs
        else expected[0]["p_inconsistent"] is not None
    )
    assert expected[0]["support_scores"] == ({} if "support_set" in outputs else None)
    await export_verifier_classifier(verifier, tok, tmp_path / "artifact")
    restored = await load_verifier_classifier(tmp_path / "artifact")
    assert restored.tokenizer.is_fast is True
    restored_batch = await encode_verifier_batch(
        restored.tokenizer, empty, geometry=restored.geometry
    )
    assert await score_verifier_batch(restored, restored_batch) == expected


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "pooling",
        "labels",
        "marker",
        "version",
        "implementation",
        "tokenizer_path",
        "tokenizer_class",
        "extra_head_export",
    ],
)
async def test_metadata_drift_and_unsafe_paths_refuse(damage: str, tmp_path: Path) -> None:
    torch, _, verifier, tok = await initialized("gpt2")
    if torch is None:
        return
    path = tmp_path / "artifact"
    if damage == "extra_head_export":
        verifier.model.unknown_head = torch.nn.Linear(8, 1)
        with pytest.raises(VerifierGeometryError):
            await export_verifier_classifier(verifier, tok, path)
        return
    await export_verifier_classifier(verifier, tok, path)
    file = path / "strata_verifier.json"
    body = json.loads(file.read_text())
    if damage == "duplicate":
        file.write_text('{"schema_version":2,' + file.read_text()[1:])
    elif damage in {"pooling", "labels", "marker", "version"}:
        field, value = {
            "pooling": ("pooling", "cls"),
            "labels": ("labels", ["other", "two", "three"]),
            "marker": ("pid_marker", "last_token"),
            "version": ("schema_version", 2.0),
        }[damage]
        body[field] = value
        file.write_text(json.dumps(body))
    elif damage == "implementation":
        file = path / "strata_classifier.json"
        body = json.loads(file.read_text())
        body["implementation"] = "arbitrary.loader"
        file.write_text(json.dumps(body))
    else:
        import hashlib

        file = path / "tokenizer_config.json"
        body = json.loads(file.read_text())
        body["tokenizer_file" if damage == "tokenizer_path" else "tokenizer_class"] = (
            "/fictional/outside.json" if damage == "tokenizer_path" else "UnknownRemoteTokenizer"
        )
        file.write_text(json.dumps(body))
        manifest = json.loads((path / "strata_classifier.json").read_text())
        manifest["tokenizer_sha256"]["tokenizer_config.json"] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        (path / "strata_classifier.json").write_text(json.dumps(manifest))
    with pytest.raises(VerifierGeometryError, match=r"^verifier_artifact_invalid$") as error:
        await load_verifier_classifier(path)
    assert error.value.__cause__ is error.value.__context__ is None


async def test_tokenizer_offsets_required_and_no_truncation_is_requested() -> None:
    torch, _, verifier, tok = await initialized("bert")
    if torch is None:
        return

    class Spy:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.is_fast: bool = True

        def __getattr__(self, name: str) -> Any:
            return getattr(tok, name)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return tok(*args, **kwargs)

    spy = Spy()
    await encode_verifier_batch(spy, examples(), geometry=verifier.geometry)
    assert spy.calls[0]["truncation"] is False
    assert spy.calls[0]["return_offsets_mapping"] is True
    tok.padding_side = "left"
    with pytest.raises(VerifierGeometryError):
        await encode_verifier_batch(tok, examples(), geometry=verifier.geometry)
    spy.is_fast = False
    with pytest.raises(VerifierGeometryError):
        await encode_verifier_batch(spy, examples(), geometry=verifier.geometry)


async def test_existing_primary_loader_does_not_adopt_auxiliary_artifact(tmp_path: Path) -> None:
    torch, _, verifier, tok = await initialized("gpt2")
    if torch is None:
        return
    from strata_forge.core.errors import ValidationError
    from strata_forge.training.classifier import load_classifier

    path = tmp_path / "auxiliary"
    await export_verifier_classifier(verifier, tok, path)
    with pytest.raises(ValidationError):
        await load_classifier(path)
