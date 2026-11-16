"""Independent bilinear values/gradients and explicit ownership of native parameters."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

import pytest

from strata_forge.training.verifier_classifier import (
    encode_verifier_batch,
    initialize_verifier_classifier,
    score_verifier_batch,
)
from strata_forge.training.verifier_geometry import VerifierGeometryError
from strata_forge.training.verifier_loss import (
    ResearchLossRow,
    ThreeWayTarget,
    VerifierLossError,
    research_verifier_loss,
)
from tests.unit.training.test_verifier_classifier import examples, initialized


async def test_nonsymmetric_bilinear_orientation_and_gradients() -> None:
    torch, _, verifier, _ = await initialized("gpt2")
    if torch is None:
        return
    head = verifier.model.support_head
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(2)
        head.weight[:2, :2].copy_(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
    h = torch.tensor([1.0, 2.0, 0, 0, 0, 0, 0, 0], requires_grad=True)
    m = torch.tensor([[11.0, 13.0, 0, 0, 0, 0, 0, 0]], requires_grad=True)
    value = head(h, m)
    assert value.tolist() == [344.0]
    value.sum().backward()
    assert h.grad.tolist()[:2] == [50.0, 146.0]
    assert m.grad.tolist()[0][:2] == [11.0, 17.0]
    assert head.weight.grad.tolist()[0][:2] == [11.0, 13.0]
    assert head.weight.grad.tolist()[1][:2] == [22.0, 26.0]
    assert head.bias.grad.tolist() == [1.0]


async def test_seed_biases_and_sharing_are_explicit() -> None:
    torch, primary, first, _ = await initialized("bert")
    if torch is None:
        return
    snapshot = {key: value.detach().clone() for key, value in primary.model.state_dict().items()}
    second = await initialize_verifier_classifier(
        primary, geometry=first.geometry, seed=13, ownership="share"
    )
    assert torch.equal(first.model.support_head.weight, second.model.support_head.weight)
    assert torch.equal(
        first.model.inconsistency_head.weight, second.model.inconsistency_head.weight
    )
    assert first.model.inconsistency_head.bias.tolist() == [0.0]
    assert first.model.support_head.bias.tolist() == [0.0]
    assert all(
        torch.equal(value, primary.model.state_dict()[key]) for key, value in snapshot.items()
    )
    assert first.model.bert is primary.model.bert
    assert not hasattr(primary.model.config, "strata_verifier_outputs")
    with pytest.raises(VerifierGeometryError, match=r"^verifier_initialization_invalid$"):
        await initialize_verifier_classifier(
            primary, geometry=first.geometry, seed=14, ownership="share"
        )
    with pytest.raises(VerifierGeometryError):
        await initialize_verifier_classifier(
            primary, geometry=first.geometry, seed=13, ownership=cast("Literal['share']", "copy")
        )


@pytest.mark.parametrize("side", ["left", "right"])
async def test_decoder_pad_eos_mask_and_score_restore_modes(side: str) -> None:
    torch, _, verifier, tokenizer = await initialized("gpt2")
    if torch is None:
        return
    tokenizer.padding_side = side
    batch = await encode_verifier_batch(tokenizer, examples(), geometry=verifier.geometry)
    verifier.model.train()
    verifier.model.backbone.drop.eval()
    before = tuple(module.training for module in verifier.model.modules())
    values = await score_verifier_batch(verifier, batch)
    assert tuple(module.training for module in verifier.model.modules()) == before
    assert all(row["p_inconsistent"] is not None for row in values)
    changed = dict(batch.model_inputs)
    changed["attention_mask"] = changed["attention_mask"].clone()
    changed["attention_mask"][0, batch.marker_positions[0][0]] = 0
    with pytest.raises(VerifierGeometryError, match=r"^verifier_output_invalid$"):
        await score_verifier_batch(verifier, replace(batch, model_inputs=changed))


async def test_headless_nonnull_supervision_refuses_and_wrong_digest_never_scores() -> None:
    torch, _, verifier, tok = await initialized("gpt2", ("label",))
    if torch is None:
        return
    batch = await encode_verifier_batch(tok, examples(), geometry=verifier.geometry)
    output = verifier.model(**batch.forward_kwargs())
    targets = tuple(
        ResearchLossRow(primary=ThreeWayTarget(label_dist=(1.0, 0.0, 0.0)), flag_target=False)
        for _ in examples()
    )
    with pytest.raises(VerifierLossError):
        research_verifier_loss(
            output.logits,
            class_order=("E", "C", "U"),
            rows=targets,
            flag_logits=output.inconsistent_logit,
            support_logits=output.support_logits,
        )
    with pytest.raises(VerifierGeometryError):
        await score_verifier_batch(verifier, replace(batch, geometry_sha256="f" * 64))


@pytest.mark.parametrize("side", ["left", "right"])
async def test_attended_eos_is_the_decoder_answer_position(side: str) -> None:
    torch, primary, _, tok = await initialized("gpt2")
    if torch is None:
        return
    from strata_forge.training.verifier_geometry import VerifierExample, VerifierPremise
    from tests.unit.training.test_verifier_geometry import geometry

    config = geometry("answer_position", template="{premises}\n{hypothesis}")
    verifier = await initialize_verifier_classifier(
        primary, geometry=config, seed=13, ownership="share"
    )
    tok.padding_side = side
    rows = (
        VerifierExample(
            item_id="eos", premises=(VerifierPremise(pid="p1", text="a"),), hypothesis="[PAD]"
        ),
        examples()[0],
    )
    batch = await encode_verifier_batch(tok, rows, geometry=config)
    attention = batch.model_inputs["attention_mask"]
    last = (torch.arange(attention.shape[1]) * attention).max(-1).values
    assert (
        batch.model_inputs["input_ids"][0, last[0]].item() == tok.eos_token_id == tok.pad_token_id
    )
    primary.model.eval()
    verifier.model.eval()
    with torch.no_grad():
        actual = verifier.model(**batch.forward_kwargs())
        assert torch.equal(actual.logits, primary.model(**batch.model_inputs).logits)
        hidden = verifier.model.backbone(**batch.model_inputs, return_dict=True).last_hidden_state
        pooled = hidden[torch.arange(len(rows)), last]
        assert torch.equal(
            actual.inconsistent_logit, verifier.model.inconsistency_head(pooled).squeeze(-1)
        )


@pytest.mark.parametrize("family", ["bert", "modernbert-decoder"])
async def test_auxiliary_heads_use_raw_final_states_not_primary_transforms(family: str) -> None:
    torch, _, verifier, tok = await initialized(family)
    if torch is None:
        return
    batch = await encode_verifier_batch(tok, examples(), geometry=verifier.geometry)
    verifier.model.eval()
    with torch.no_grad():
        actual = verifier.model(**batch.forward_kwargs())
        if family == "bert":
            native = verifier.model.bert(**batch.model_inputs, return_dict=True)
            pooled = native.last_hidden_state[:, 0]
        else:
            native = verifier.model.backbone(**batch.model_inputs, return_dict=True)
            mask = batch.model_inputs["attention_mask"]
            last = (torch.arange(mask.shape[1]) * mask).max(-1).values
            pooled = native.last_hidden_state[torch.arange(mask.shape[0]), last]
        assert torch.equal(
            actual.inconsistent_logit, verifier.model.inconsistency_head(pooled).squeeze(-1)
        )
        for row, markers in enumerate(batch.marker_positions):
            expected = verifier.model.support_head(
                pooled[row], native.last_hidden_state[row, list(markers)]
            )
            assert torch.equal(actual.support_logits[row], expected)


@pytest.mark.parametrize("damage", ["labels", "remote_map", "negative_std", "body_removed"])
async def test_initialization_refuses_inconsistent_primary_contract(damage: str) -> None:
    torch, primary, verifier, _ = await initialized("bert")
    if torch is None:
        return
    geometry = verifier.geometry
    if damage == "labels":
        geometry = geometry.model_copy(update={"labels": tuple(reversed(geometry.labels))})
    elif damage == "remote_map":
        primary.model.config.auto_map = {"AutoModel": "fictional.module.Class"}
    elif damage == "negative_std":
        primary.model.config.initializer_range = -1.0
    else:
        primary.model.bert.pooler = None
    with pytest.raises(VerifierGeometryError, match=r"^verifier_initialization_invalid$"):
        await initialize_verifier_classifier(primary, geometry=geometry, seed=13, ownership="share")


@pytest.mark.parametrize(
    "damage", ["mask_shape", "all_padding", "row_count", "duplicate_marker", "token_types"]
)
async def test_native_forward_refuses_malformed_batch_geometry(damage: str) -> None:
    torch, _, verifier, tok = await initialized("gpt2")
    if torch is None:
        return
    batch = await encode_verifier_batch(tok, examples(), geometry=verifier.geometry)
    inputs = batch.forward_kwargs()
    if damage == "mask_shape":
        inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
    elif damage == "all_padding":
        inputs["attention_mask"] = torch.zeros_like(inputs["attention_mask"])
    elif damage == "row_count":
        inputs["premise_ids"] = ()
    elif damage == "duplicate_marker":
        first = batch.marker_positions[0][0]
        inputs["marker_positions"] = ((first, first), batch.marker_positions[1])
    else:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    with pytest.raises(VerifierGeometryError):
        verifier.model(**inputs)
