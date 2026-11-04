"""Tiny random CPU classifier training and complete local artifacts; no Hub/model assets."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.training.classifier import (
    ClassifierConfig,
    ClassifierExample,
    ClassifierRunner,
    _loss,  # pyright: ignore[reportPrivateUsage] — known-value loss/gradient contract
    initialize_classifier,
    load_classifier,
)


def stack() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError:
        from strata_forge.training._classifier_heads import classifier_modules

        with pytest.raises(ImportError, match="finetuning"):
            classifier_modules()
        return None, None
    return torch, transformers


def tokenizer() -> Any:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
    )


def source(family: str, transformers: Any) -> Any:
    if family == "bert":
        return transformers.BertModel(
            transformers.BertConfig(
                vocab_size=8,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=32,
            )
        )
    if family == "gpt2":
        return transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=8,
                n_embd=8,
                n_layer=1,
                n_head=2,
                n_positions=32,
                pad_token_id=0,
                eos_token_id=0,
            )
        )
    return transformers.ModernBertDecoderForCausalLM(
        transformers.ModernBertDecoderConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            pad_token_id=0,
            eos_token_id=0,
            decoder_bias=True,
            classifier_bias=False,
            attn_implementation="eager",
        )
    )


def batch(torch: Any) -> dict[str, Any]:
    # EOS==pad is genuinely attended on row0 and padding on row1.
    return {
        "input_ids": torch.tensor([[2, 3, 0], [4, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"brier_weight": 0.1},
        {"objective": "ce_brier"},
        {"class_weights": (1.0, 2.0)},
        {"label_smoothing": 0.1},
        {"source_objective": "nli_binary_grouped"},
        {"checkpoint_selection": "metric", "selection_split": "dev"},
        {
            "checkpoint_selection": "metric",
            "selection_split": "gate",
            "evaluation_tiers": {"gate": "gate"},
        },
        {
            "checkpoint_selection": "metric",
            "selection_split": "test",
            "evaluation_tiers": {"test": "test"},
        },
        {"labels": ("a", "a")},
        {"max_steps": 0},
        {"resume_from_checkpoint": "arbitrary"},
        {"support_weight": 1.0},
    ],
)
def test_unsupported_objectives_roles_and_selection_refuse(changes: dict[str, Any]) -> None:
    with pytest.raises((PydanticValidationError, ValidationError)):
        ClassifierConfig.model_validate({"labels": ("a", "b"), **changes})


def test_generic_default_and_named_research_preset_are_distinct() -> None:
    generic = ClassifierConfig(labels=("a", "b"))
    research = ClassifierConfig.research_ce_brier(labels=("a", "b"))
    assert generic.brier_weight == 0
    assert generic.objective == "cross_entropy"
    assert research.brier_weight == 0.1
    assert research.objective == "ce_brier"
    assert generic.checkpoint_selection == "final"


def test_known_loss_and_gradient() -> None:
    torch, _ = stack()
    if torch is None:
        return
    logits = torch.tensor([[math.log(3.0), 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    loss = _loss(logits, targets, 0.1)
    assert math.isclose(float(loss.detach()), -math.log(0.75) + 0.1 * 0.125, rel_tol=1e-6)
    loss.backward()
    assert all(
        math.isclose(actual, expected, rel_tol=1e-6)
        for actual, expected in zip(logits.grad.tolist()[0], [-0.26875, 0.26875], strict=True)
    )
    assert math.isclose(
        float(_loss(torch.zeros((1, 2)), torch.tensor([[0.25, 0.75]]), 0)),
        math.log(2),
        rel_tol=1e-6,
    )


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert_decoder"])
@pytest.mark.asyncio
async def test_actual_tiny_optimizer_step_safe_export_and_complete_reload(
    family: str, tmp_path: Path
) -> None:
    torch, transformers = stack()
    if torch is None:
        return
    native = source(family, transformers)
    if family == "modernbert_decoder":
        with torch.no_grad():
            native.decoder.bias.copy_(torch.tensor([0.4, -0.6, 0.2, 1.1, -0.2, 0.9, 0.3, -0.1]))
    initialized = await initialize_classifier(
        native,
        labels=("yes", "no"),
        initializer_id="encoder_random_head" if family == "bert" else "decoder_logit_head",
        verbalizer_ids=() if family == "bert" else (2, 3),
        equivalence_inputs=None if family == "bert" else batch(torch),
    )
    if family != "bert":
        assert initialized.receipt.pooling == "last_attended_token"
        assert initialized.receipt.step_zero_max_probability_error is not None
        assert initialized.receipt.step_zero_max_probability_error < 1e-6
    if family == "modernbert_decoder":
        assert initialized.receipt.bias_copied
        assert initialized.model.head.dense.bias is None
        assert all(
            math.isclose(actual, expected, rel_tol=1e-6)
            for actual, expected in zip(
                initialized.model.classifier.bias.tolist(), [0.2, 1.1], strict=True
            )
        )
    before = initialized.model.classifier.weight.detach().clone()
    config = ClassifierConfig(
        labels=("yes", "no"), max_steps=1, batch_size=2, learning_rate=0.01, use_cpu=True
    )
    rows = (
        ClassifierExample(item_id="1", text="a b", label="yes"),
        ClassifierExample(item_id="2", text="c d", label="no"),
    )
    result = await ClassifierRunner(config).train(
        initialized=initialized, tokenizer=tokenizer(), rows=rows, output_dir=tmp_path
    )
    assert result.steps == 1
    assert not torch.equal(before, initialized.model.classifier.weight)
    artifact = Path(result.artifact_dir)
    assert (artifact / "model.safetensors").is_file()
    assert not await asyncio.to_thread(lambda: list(artifact.rglob("*.bin")))
    assert not await asyncio.to_thread(lambda: list(artifact.rglob("checkpoint-*")))
    manifest = json.loads((artifact / "strata_classifier.json").read_text())
    assert manifest["config"]["brier_weight"] == 0
    assert manifest["checkpoint_selection"] == "final"
    loaded = await load_classifier(artifact)
    initialized.model.eval()
    loaded.model.eval()
    with torch.no_grad():
        assert torch.allclose(
            initialized.model(**batch(torch)).logits, loaded.model(**batch(torch)).logits
        )
    assert loaded.receipt == initialized.receipt


@pytest.mark.asyncio
async def test_bad_verbalizers_and_missing_equivalence_refuse() -> None:
    torch, transformers = stack()
    if torch is None:
        return
    native = source("gpt2", transformers)
    for ids in ((2, 2), (2,), (2, 99), (True, 2)):
        with pytest.raises(ValidationError):
            await initialize_classifier(
                native,
                labels=("yes", "no"),
                initializer_id="decoder_logit_head",
                verbalizer_ids=ids,
                equivalence_inputs=batch(torch),
            )
    with pytest.raises(ValidationError, match="step_zero"):
        await initialize_classifier(
            native, labels=("yes", "no"), initializer_id="decoder_logit_head", verbalizer_ids=(2, 3)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"label": 0},
        {"label": "unknown"},
        {"label_dist": (0.2, 0.2)},
        {"label_dist": (float("nan"), 0.5)},
        {"label_dist": (0.1, 0.2, 0.7)},
    ],
)
@pytest.mark.asyncio
async def test_row_semantics_refuse_before_optimizer(
    changes: dict[str, Any], tmp_path: Path
) -> None:
    torch, transformers = stack()
    if torch is None:
        return
    initialized = await initialize_classifier(
        source("bert", transformers), labels=("yes", "no"), initializer_id="encoder_random_head"
    )
    row = ClassifierExample.model_validate({"item_id": "x", "text": "a", "label": "yes", **changes})
    with pytest.raises(ValidationError):
        await ClassifierRunner(
            ClassifierConfig(labels=("yes", "no"), max_steps=1, use_cpu=True)
        ).train(initialized=initialized, tokenizer=tokenizer(), rows=(row,), output_dir=tmp_path)
    assert not (tmp_path / ".classifier-attempt.json").exists()


@pytest.mark.asyncio
async def test_explicit_integer_map_soft_targets_overflow_and_rendering(tmp_path: Path) -> None:
    from strata_forge.training.verifier_config import VerifierConfig

    torch, transformers = stack()
    if torch is None:
        return
    rendering = VerifierConfig(
        labels=("yes", "no"),
        template="{premises} {hypothesis}",
        premise_format="numbered_lines",
        max_length=8,
        max_premises=2,
    )
    row = ClassifierExample.from_verifier(
        rendering=rendering, item_id="1", premises=["a"], hypothesis="b", label=7
    )
    initialized = await initialize_classifier(
        source("bert", transformers), labels=("yes", "no"), initializer_id="encoder_random_head"
    )
    config = ClassifierConfig(
        labels=("yes", "no"),
        label_index_map={7: "yes"},
        rendering=rendering,
        max_steps=1,
        use_cpu=True,
        max_length=8,
        overflow="drop",
    )
    dropped = ClassifierExample.from_verifier(
        rendering=rendering, item_id="2", premises=["a " * 20], hypothesis="b", label=7
    )
    result = await ClassifierRunner(config).train(
        initialized=initialized, tokenizer=tokenizer(), rows=(row, dropped), output_dir=tmp_path
    )
    assert result.trained_rows == 1
    assert result.overflow_ids == ("2",)
    written = VerifierConfig.model_validate_json(
        (Path(result.artifact_dir) / "strata_verifier.json").read_text()
    )
    assert written == rendering
    assert (await load_classifier(Path(result.artifact_dir))).receipt.labels == config.labels
    with pytest.raises(ValidationError, match="rendering"):
        await ClassifierRunner(config).train(
            initialized=initialized,
            tokenizer=tokenizer(),
            rows=(row.model_copy(update={"text": "different"}),),
            output_dir=tmp_path / "other",
        )


@pytest.mark.parametrize("kind", ["head", "body", "unexpected"])
@pytest.mark.asyncio
async def test_saved_receipt_never_exempts_incomplete_or_extra_weights(
    kind: str, tmp_path: Path
) -> None:
    torch, transformers = stack()
    if torch is None:
        return
    from safetensors.torch import (  # pyright: ignore[reportUnknownVariableType] — optional upstream PathLike stubs
        load_file,  # pyright: ignore[reportUnknownVariableType] — upstream PathLike stubs
        save_file,  # pyright: ignore[reportUnknownVariableType] — upstream PathLike stubs
    )

    from strata_forge.training._load_guard import ModelLoadingError

    initialized = await initialize_classifier(
        source("gpt2", transformers),
        labels=("yes", "no"),
        initializer_id="decoder_logit_head",
        verbalizer_ids=(2, 3),
        equivalence_inputs=batch(torch),
    )
    result = await ClassifierRunner(
        ClassifierConfig(labels=("yes", "no"), max_steps=1, use_cpu=True)
    ).train(
        initialized=initialized,
        tokenizer=tokenizer(),
        rows=(ClassifierExample(item_id="1", text="a b", label="yes"),),
        output_dir=tmp_path,
    )
    artifact = Path(result.artifact_dir)

    def damage() -> None:
        weights = load_file(artifact / "model.safetensors")
        if kind == "unexpected":
            weights["arbitrary_head.weight"] = torch.ones(1)
        else:
            key = next(
                key
                for key in weights
                if key.startswith("classifier." if kind == "head" else "backbone.")
            )
            weights.pop(key)
        save_file(weights, artifact / "model.safetensors", metadata={"format": "pt"})

    await asyncio.to_thread(damage)
    with pytest.raises(ModelLoadingError):
        await load_classifier(artifact)


@pytest.mark.asyncio
async def test_undeclared_source_keys_refuse() -> None:
    torch, transformers = stack()
    if torch is None:
        return
    native = source("gpt2", transformers)
    native.register_parameter("arbitrary_head", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValidationError, match="undeclared"):
        await initialize_classifier(
            native,
            labels=("yes", "no"),
            initializer_id="decoder_logit_head",
            verbalizer_ids=(2, 3),
            equivalence_inputs=batch(torch),
        )


@pytest.mark.asyncio
async def test_cancellation_cooperatively_stops_without_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    torch, transformers = stack()
    if torch is None:
        return
    initialized = await initialize_classifier(
        source("bert", transformers), labels=("yes", "no"), initializer_id="encoder_random_head"
    )
    started, release = threading.Event(), threading.Event()
    original = initialized.model.forward

    def blocked(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(initialized.model, "forward", blocked)
    task = asyncio.create_task(
        ClassifierRunner(ClassifierConfig(labels=("yes", "no"), max_steps=10, use_cpu=True)).train(
            initialized=initialized,
            tokenizer=tokenizer(),
            rows=(ClassifierExample(item_id="1", text="a", label="yes"),),
            output_dir=tmp_path,
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize("family", ["gpt2", "modernbert_decoder"])
@pytest.mark.asyncio
async def test_step_zero_matches_actual_score_engine_with_attended_eos(family: str) -> None:
    from strata_forge.pipelines._scoring import Scorer, ScoringInput
    from strata_forge.training._load_guard import LoadingAudit, PaddingPolicy

    torch, transformers = stack()
    if torch is None:
        return
    native = source(family, transformers)
    if family == "modernbert_decoder":
        with torch.no_grad():
            native.decoder.bias[2:4].copy_(torch.tensor([0.4, -0.3]))
    tok = tokenizer()
    texts = ["a b [PAD]", "c"]
    features = tok(texts, padding=True, return_tensors="pt")
    initialized = await initialize_classifier(
        native,
        labels=("yes", "no"),
        initializer_id="decoder_logit_head",
        verbalizer_ids=(2, 3),
        equivalence_inputs=dict(features),
    )
    native.eval()
    initialized.model.eval()
    score = await Scorer(
        engine="hf_label_logits",
        model=native,
        tokenizer=tok,
        torch=torch,
        labels=["yes", "no"],
        token_ids=[2, 3],
        max_length=32,
        device="cpu",
        loading_audit=LoadingAudit(status="accepted"),
        padding_policy=PaddingPolicy(
            source="tokenizer_pad", pad_token_id=0, original_model_pad_token_id=0
        ),
    ).score([ScoringInput(text=text) for text in texts], batch_size=2)
    with torch.no_grad():
        actual = initialized.model(**features).logits.softmax(-1).tolist()
    for row, values in zip(score.rows, actual, strict=True):
        assert row.error is None
        for label, probability in zip(("yes", "no"), values, strict=True):
            assert math.isclose(row.probabilities[label], probability, abs_tol=1e-6, rel_tol=1e-5)


@pytest.mark.parametrize(
    "case",
    ["empty", "duplicate", "overflow", "wrong_labels", "sample_weight", "bad_render", "metadata"],
)
@pytest.mark.asyncio
async def test_primary_refusal_boundaries(case: str, tmp_path: Path) -> None:
    torch, transformers = stack()
    if torch is None:
        return
    if case == "sample_weight":
        with pytest.raises(PydanticValidationError):
            ClassifierExample(item_id="x", text="a", label="yes", sample_weight=2.0)
        return
    if case == "metadata":
        (tmp_path / "strata_classifier.json").write_text("x" * 65537)
        with pytest.raises(ValidationError, match="too_large"):
            await load_classifier(tmp_path)
        return
    initialized = await initialize_classifier(
        source("bert", transformers), labels=("yes", "no"), initializer_id="encoder_random_head"
    )
    row = ClassifierExample(item_id="1", text="a b c", label="yes")
    rows = () if case == "empty" else (row, row) if case == "duplicate" else (row,)
    config = ClassifierConfig(
        labels=("a", "b") if case == "wrong_labels" else ("yes", "no"),
        max_length=1 if case == "overflow" else 32,
        max_steps=1,
        use_cpu=True,
    )
    if case == "bad_render":
        rows = (row.model_copy(update={"premises": ("a",), "hypothesis": "b"}),)
    with pytest.raises(ValidationError):
        await ClassifierRunner(config).train(
            initialized=initialized, tokenizer=tokenizer(), rows=rows, output_dir=tmp_path
        )


@pytest.mark.asyncio
async def test_dense_and_norm_bias_are_independent_of_output_bias() -> None:
    torch, transformers = stack()
    if torch is None:
        return
    config = source("modernbert_decoder", transformers).config
    config.classifier_bias = True
    config.norm_bias = True
    config.decoder_bias = False
    native = transformers.ModernBertDecoderForCausalLM(config)
    with torch.no_grad():
        native.lm_head.dense.bias.fill_(0.2)
        native.lm_head.norm.bias.fill_(-0.1)
    initialized = await initialize_classifier(
        native,
        labels=("yes", "no"),
        initializer_id="decoder_logit_head",
        verbalizer_ids=(2, 3),
        equivalence_inputs=batch(torch),
    )
    assert initialized.model.classifier.bias is None
    assert not initialized.receipt.bias_copied
    assert torch.equal(initialized.model.head.dense.bias, native.lm_head.dense.bias)
    assert torch.equal(initialized.model.head.norm.bias, native.lm_head.norm.bias)
    assert initialized.receipt.step_zero_max_probability_error is not None
    assert initialized.receipt.step_zero_max_probability_error < 1e-6


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.asyncio
async def test_three_class_soft_targets_checkpointing_and_precision(
    precision: str, tmp_path: Path
) -> None:
    torch, transformers = stack()
    if torch is None:
        return
    labels = ("entail", "contradict", "unknown")
    initialized = await initialize_classifier(
        source("bert", transformers), labels=labels, initializer_id="encoder_random_head"
    )
    config = ClassifierConfig.model_validate(
        {
            "labels": labels,
            "max_steps": 1,
            "use_cpu": True,
            "precision": precision,
            "gradient_checkpointing": True,
        }
    )
    result = await ClassifierRunner(config).train(
        initialized=initialized,
        tokenizer=tokenizer(),
        rows=(
            ClassifierExample(
                item_id="synthetic-3way", text="a b", label="unknown", label_dist=(0.1, 0.2, 0.7)
            ),
        ),
        output_dir=tmp_path,
    )
    assert result.steps == 1
    loaded = await load_classifier(Path(result.artifact_dir))
    assert loaded.receipt.labels == labels
    assert tuple(loaded.model.config.id2label[i] for i in range(3)) == labels


def test_primary_rendering_cannot_advertise_unimplemented_heads() -> None:
    from strata_forge.training.verifier_config import VerifierConfig

    rendering = VerifierConfig(
        labels=("yes", "no"),
        template="{premises} {hypothesis}",
        premise_format="numbered_lines",
        max_length=8,
        max_premises=2,
        output_capabilities=("classification", "support"),
    )
    with pytest.raises(PydanticValidationError, match="auxiliary_outputs_unsupported"):
        ClassifierConfig(labels=("yes", "no"), rendering=rendering)


@pytest.mark.asyncio
async def test_persisted_verbalizers_must_match_actual_initializer_ids(tmp_path: Path) -> None:
    from strata_forge.training.verifier_config import VerifierConfig

    torch, transformers = stack()
    if torch is None:
        return
    initialized = await initialize_classifier(
        source("gpt2", transformers),
        labels=("yes", "no"),
        initializer_id="decoder_logit_head",
        verbalizer_ids=(2, 3),
        equivalence_inputs=batch(torch),
    )
    rendering = VerifierConfig(
        labels=("yes", "no"),
        template="{premises} {hypothesis}",
        premise_format="numbered_lines",
        max_length=8,
        max_premises=2,
        label_tokens={"yes": "c", "no": "d"},
    )
    config = ClassifierConfig(labels=("yes", "no"), rendering=rendering, max_steps=1, use_cpu=True)
    with pytest.raises(ValidationError, match="verbalizer_metadata_mismatch"):
        await ClassifierRunner(config).train(
            initialized=initialized,
            tokenizer=tokenizer(),
            rows=(ClassifierExample(item_id="synthetic", text="a", label="yes"),),
            output_dir=tmp_path,
        )
    assert not (tmp_path / ".classifier-attempt.json").exists()
