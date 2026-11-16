"""Generated fictional streams and exact local-cache witnesses, without external assets."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from strata_forge.training.classifier import ClassifierConfig
from strata_forge.training.population import (
    CompletePopulation,
    PopulationError,
    PopulationIdentity,
    PopulationLimits,
    PopulationRange,
    PopulationSourceIdentity,
    PrimaryPopulationTask,
    ResolvedPopulationTokenizer,
    TrainingPopulationPlanV2,
)
from strata_forge.training.population_runner import (
    prepare_training_population,
    train_prepared_population,
)

if TYPE_CHECKING:
    from pathlib import Path


def resolved(tokenizer: Any) -> ResolvedPopulationTokenizer:
    return ResolvedPopulationTokenizer(
        PopulationIdentity(repo_type="model", ref="fictional/tokenizer@" + "c" * 40), tokenizer
    )


class Source:
    def __init__(
        self,
        plan: TrainingPopulationPlanV2,
        rows: int | None = None,
        *,
        overflows: tuple[int, ...] = (),
        duplicate: int | None = None,
        padding: int = 0,
    ) -> None:
        self.identity = plan.source
        self.rows = plan.source.expected_rows if rows is None else rows
        self.ordinal = 0
        self.buffer = bytearray()
        self.closed = False
        self.overflows, self.duplicate, self.padding = overflows, duplicate, padding
        self.digest = hashlib.sha256()
        self.reads = 0

    async def read(self, maximum: int) -> bytes:
        self.reads += 1
        while len(self.buffer) < maximum and self.ordinal < self.rows:
            index = self.ordinal
            item_id = f"r{0 if index == self.duplicate else index:06d}"
            text = "a " * (40 if index in self.overflows else 1) + "b" * self.padding
            raw = (
                json.dumps(
                    {"item_id": item_id, "text": text, "label": "yes"}, separators=(",", ":")
                ).encode()
                + b"\n"
            )
            self.buffer.extend(raw)
            self.ordinal += 1
        chunk = bytes(self.buffer[:maximum])
        del self.buffer[:maximum]
        self.digest.update(chunk)
        return chunk

    async def aclose(self) -> None:
        self.closed = True


class Authority:
    def __init__(self, deny: str | None = None) -> None:
        self.phases: list[str] = []
        self.deny = deny

    async def check(self, plan: TrainingPopulationPlanV2, *, phase: str) -> None:
        self.phases.append(phase)
        if phase == self.deny:
            raise RuntimeError("PRIVATE_AUTHORITY_CANARY")


class Tokenizer:
    is_fast = True
    padding_side = "right"
    pad_token_id = 0
    cls_token_id = 2
    eos_token_id = 0
    model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"fixed":"fictional"}')

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {"text_pair": None, "truncation": False, "padding": False}
        ids = [3] * len(text.split())
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def plan(
    rows: int = 80, *, start: int | None = None, end: int | None = None, seed: int = 13
) -> TrainingPopulationPlanV2:
    return TrainingPopulationPlanV2(
        schema_version=2,
        source=PopulationSourceIdentity(
            ref="fictional/source@" + "a" * 40, config="default", split="train", expected_rows=rows
        ),
        model=PopulationIdentity(repo_type="model", ref="fictional/model@" + "b" * 40),
        tokenizer=PopulationIdentity(repo_type="model", ref="fictional/tokenizer@" + "c" * 40),
        family="bert",
        selection=CompletePopulation(mode="complete")
        if start is None
        else PopulationRange(mode="range", start=start, end=cast("int", end)),
        task=PrimaryPopulationTask(
            codec="primary_classifier_v1",
            config=ClassifierConfig(
                labels=("yes", "no"), batch_size=32, max_length=16, seed=seed, use_cpu=True
            ),
        ),
        limits=PopulationLimits(serialized_bytes=256 * 1024**2, disk_bytes=256 * 1024**2),
        seed=seed,
        overflow="error",
    )


def receipt(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "preflight-receipt.json").read_bytes())


def _fixed_backend(value: Any) -> Any:
    def factory(*_args: Any) -> Any:
        return value

    return factory


class CountingBackend:
    def __init__(self) -> None:
        self.profile = {"test": "counting_only"}
        self.versions = {"test": "synthetic"}
        self.ids: list[str] = []
        self.batches: list[int] = []
        self.steps = 0

    def apply(self, records: Any) -> None:
        self.ids.extend(row.item_id for row in records)
        self.batches.append(len(records))

    def after_step(self) -> None:
        self.steps += 1


@pytest.mark.asyncio
async def test_complete_source_selected_epoch_exact_partial_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = plan(start=10, end=75)
    source, tokenizer, authority = Source(p), Tokenizer(), Authority()
    prepared = await prepare_training_population(
        p,
        source=source,
        tokenizer=resolved(tokenizer),
        authority=authority,
        attempt_dir=tmp_path / "attempt",
    )
    assert source.closed
    assert prepared.receipt.scan_complete
    assert prepared.receipt.source_rows_observed == 80
    assert prepared.receipt.selected_rows_observed == 65
    assert prepared.receipt.source_stream_sha256 == source.digest.hexdigest()
    fake = CountingBackend()
    monkeypatch.setattr("strata_forge.training.population_runner._backend", _fixed_backend(fake))
    result = await train_prepared_population(
        prepared, initialized=cast("Any", None), tokenizer=resolved(tokenizer), authority=authority
    )
    assert sorted(fake.ids) == [f"r{index:06d}" for index in range(10, 75)]
    assert fake.batches == [32, 32, 1]
    assert fake.steps == result.confirmed_optimizer_steps == 3
    assert result.confirmed_applied_rows == 65
    assert result.applied_order_sha256 == result.planned_order_sha256
    assert authority.phases == ["before_read", "after_read", "before_train", "after_train"]
    with pytest.raises(PopulationError):
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=authority,
        )
    assert fake.steps == 3


@pytest.mark.parametrize(
    ("overflows", "first", "total", "selected"),
    [((3, 42, 78), "r000003", 3, 1), ((78,), "r000078", 1, 0)],
)
@pytest.mark.asyncio
async def test_entire_source_overflow_outside_pilot_blocks_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overflows: tuple[int, ...],
    first: str,
    total: int,
    selected: int,
) -> None:
    p = plan(start=10, end=75)
    source = Source(p, overflows=overflows)
    calls: list[object] = []

    def record_call(*args: Any) -> None:
        calls.append(args)

    monkeypatch.setattr("strata_forge.training.population_runner._backend", record_call)
    with pytest.raises(PopulationError, match=r"^population_overflow$"):
        await prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "attempt",
        )
    r = receipt(tmp_path / "attempt")
    assert r["scan_complete"] is True
    assert (
        r["first_overflow_item_id"],
        r["source_overflow_rows"],
        r["selected_overflow_rows"],
    ) == (first, total, selected)
    assert r["source_rows_observed"] == 80
    assert source.closed
    assert calls == []
    assert "PRIVATE" not in json.dumps(r)


@pytest.mark.parametrize(("rows", "duplicate"), [(79, None), (81, None), (80, 79)])
@pytest.mark.asyncio
async def test_incomplete_extra_and_duplicate_never_become_prepared(
    tmp_path: Path, rows: int, duplicate: int | None
) -> None:
    p = plan()
    source = Source(p, rows, duplicate=duplicate)
    with pytest.raises(PopulationError):
        await prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "attempt",
        )
    assert source.closed
    assert receipt(tmp_path / "attempt")["status"] == "invalid"
    assert receipt(tmp_path / "attempt")["scan_complete"] is False


@pytest.mark.asyncio
async def test_denial_before_read_and_fresh_attempt_refusal(tmp_path: Path) -> None:
    p, tokenizer = plan(), Tokenizer()
    source = Source(p)
    with pytest.raises(PopulationError, match=r"^population_authority_denied$") as caught:
        await prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(tokenizer),
            authority=Authority("before_read"),
            attempt_dir=tmp_path / "attempt",
        )
    assert caught.value.__context__ is None
    assert source.reads == 0
    assert source.closed
    (tmp_path / "attempt").mkdir()
    with pytest.raises(PopulationError, match=r"^population_attempt_unavailable$"):
        await prepare_training_population(
            p,
            source=Source(p),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
            attempt_dir=tmp_path / "attempt",
        )


@pytest.mark.asyncio
async def test_cache_or_tokenizer_changes_refuse_before_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, tokenizer = plan(), Tokenizer()
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "attempt",
    )

    def refuse_optimizer(*_args: Any) -> None:
        pytest.fail("optimizer must not exist")

    monkeypatch.setattr("strata_forge.training.population_runner._backend", refuse_optimizer)
    tokenizer.padding_side = "left"
    with pytest.raises(PopulationError):
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
        )
    tokenizer.padding_side = "right"
    with (prepared.directory / "cache.sqlite").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(PopulationError):
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
        )


@pytest.mark.asyncio
async def test_cancel_held_reader_closes_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = plan()
    held = asyncio.Event()
    source = Source(p)

    async def read(_: int) -> bytes:
        held.set()
        await asyncio.Event().wait()
        return b""

    monkeypatch.setattr(source, "read", read)
    task = asyncio.create_task(
        prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "attempt",
        )
    )
    await held.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed
    assert receipt(tmp_path / "attempt")["status"] == "cancelled"


class BytesSource:
    def __init__(self, identity: PopulationSourceIdentity, content: bytes) -> None:
        self.identity, self.content = identity, content
        self.position = 0
        self.closed = False

    async def read(self, maximum: int) -> bytes:
        value = self.content[self.position : self.position + maximum]
        self.position += len(value)
        return value

    async def aclose(self) -> None:
        self.closed = True


async def population_initialized(family: str) -> tuple[Any, Any, Any, Any]:
    from strata_forge.training._classifier_heads import initialize_classifier
    from strata_forge.training.verifier_classifier import initialize_verifier_classifier
    from tests.unit.training.test_classifier import stack
    from tests.unit.training.test_verifier_classifier import tokenizer
    from tests.unit.training.test_verifier_geometry import geometry

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
        with torch.no_grad():
            base.decoder.bias.copy_(torch.linspace(-3.0, 3.0, 32))
    primary = await initialize_classifier(
        base,
        labels=("E", "C", "U"),
        initializer_id="encoder_random_head" if family == "bert" else "decoder_logit_head",
        verbalizer_ids=() if family == "bert" else (10, 11, 12),
        seed=13,
        equivalence_inputs=None
        if family == "bert"
        else {
            "input_ids": torch.tensor([[7, 8, 0], [7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        },
    )
    config = geometry("cls" if family == "bert" else "answer_position", labels=("E", "C", "U"))
    auxiliary = await initialize_verifier_classifier(
        primary, geometry=config, seed=13, ownership="share"
    )
    if family == "modernbert-decoder":
        assert torch.equal(auxiliary.model.classifier.bias, base.decoder.bias[[10, 11, 12]])
        assert bool((auxiliary.model.classifier.bias != 0).all())
    return torch, primary, auxiliary, tokenizer(family == "bert")


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert-decoder"])
@pytest.mark.asyncio
async def test_tiny_auxiliary_actual_loss_optimizer_and_safe_reload(
    tmp_path: Path, family: str
) -> None:
    from strata_forge.training.population import AuxiliaryPopulationTask
    from strata_forge.training.verifier_classifier import (
        encode_verifier_batch,
        export_verifier_classifier,
        load_verifier_classifier,
        score_verifier_batch,
    )
    from strata_forge.training.verifier_loss import ResearchLossRow, ThreeWayTarget
    from tests.unit.training.test_verifier_classifier import examples

    torch, _, model, tokenizer = await population_initialized(family)
    if torch is None:
        return
    rows = examples()
    p = TrainingPopulationPlanV2(
        schema_version=2,
        source=PopulationSourceIdentity(
            ref="fictional/source@" + "a" * 40, config="default", split="train", expected_rows=2
        ),
        model=PopulationIdentity(repo_type="model", ref="fictional/model@" + "b" * 40),
        tokenizer=PopulationIdentity(repo_type="model", ref="fictional/tokenizer@" + "c" * 40),
        family=cast("Any", family),
        selection=CompletePopulation(mode="complete"),
        task=AuxiliaryPopulationTask(
            codec="auxiliary_verifier_v1", geometry=model.geometry, class_order=("E", "C", "U")
        ),
        limits=PopulationLimits(serialized_bytes=1024**2, disk_bytes=32 * 1024**2),
        seed=13,
        overflow="error",
    )
    content = b"".join(
        json.dumps(
            {
                "example": row.model_dump(mode="json"),
                "targets": ResearchLossRow(
                    primary=ThreeWayTarget(label_dist=(1.0, 0.0, 0.0)),
                    flag_target=False,
                    premise_ids=tuple(item.pid for item in row.premises),
                    support_set=(),
                ).model_dump(mode="json"),
            }
        ).encode()
        + b"\n"
        for row in rows
    )
    before = {key: value.detach().clone() for key, value in model.model.state_dict().items()}
    prepared = await prepare_training_population(
        p,
        source=BytesSource(p.source, content),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "attempt",
    )
    result = await train_prepared_population(
        prepared, initialized=model, tokenizer=resolved(tokenizer), authority=Authority()
    )
    assert result.confirmed_applied_rows == 2
    assert result.confirmed_optimizer_steps == 1
    assert result.optimizer_profile["optimizer"] == "adamw_torch_fused"
    for key in ("inconsistency_head.weight", "support_head.weight"):
        assert not torch.equal(before[key], model.model.state_dict()[key])
    await export_verifier_classifier(model, tokenizer, tmp_path / "artifact")
    loaded = await load_verifier_classifier(tmp_path / "artifact")
    batch = await encode_verifier_batch(tokenizer, rows, geometry=model.geometry)
    expected = await score_verifier_batch(model, batch)
    actual = await score_verifier_batch(loaded, batch)
    assert actual == expected


@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_600k_generated_population_crosses_both_v1_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = plan(600_000)
    source = Source(p, padding=80)
    prepared = await prepare_training_population(
        p,
        source=source,
        tokenizer=resolved(Tokenizer()),
        authority=Authority(),
        attempt_dir=tmp_path / "capacity",
    )
    assert prepared.receipt.source_rows_observed == 600_000
    assert prepared.receipt.selected_rows_observed == 600_000
    assert prepared.receipt.serialized_bytes > 64 * 1024**2
    assert prepared.receipt.source_stream_sha256 == source.digest.hexdigest()
    assert prepared.receipt.max_buffered_rows <= 32
    assert prepared.receipt.max_buffered_bytes <= 8 * 1024**2
    assert prepared.receipt.disk_high_water_bytes <= p.limits.disk_bytes
    seen = bytearray(600_000)
    calls = 0
    applied = 0

    class Counter:
        profile: ClassVar[dict[str, str]] = {"test": "counting_only_no_neural_forward"}
        versions: ClassVar[dict[str, str]] = {"test": "synthetic"}

        def apply(self, records: Any) -> None:
            nonlocal applied
            for record in records:
                assert not seen[record.source_ordinal]
                seen[record.source_ordinal] = 1
                assert record.item_id == f"r{record.source_ordinal:06d}"
                applied += 1

        def after_step(self) -> None:
            nonlocal calls
            calls += 1

    monkeypatch.setattr(
        "strata_forge.training.population_runner._backend", _fixed_backend(Counter())
    )
    result = await train_prepared_population(
        prepared,
        initialized=cast("Any", None),
        tokenizer=resolved(Tokenizer()),
        authority=Authority(),
    )
    assert calls == 18_750
    assert applied == 600_000
    assert all(seen)
    assert result.confirmed_applied_rows == 600_000
    assert result.planned_order_sha256 == result.applied_order_sha256


@pytest.mark.slow
@pytest.mark.asyncio
async def test_late_overflow_after_old_cap_reports_whole_source(tmp_path: Path) -> None:
    p = plan(100_003, start=10, end=75)
    source = Source(p, overflows=(3, 100_002))
    with pytest.raises(PopulationError, match=r"^population_overflow$"):
        await prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "late",
        )
    result = receipt(tmp_path / "late")
    assert result["source_rows_observed"] == 100_003
    assert result["source_overflow_rows"] == 2
    assert result["selected_overflow_rows"] == 0
    assert result["first_overflow_item_id"] == "r000003"
    assert result["scan_complete"] is True


@pytest.mark.parametrize("change", ["left", "not_fast", "missing_pad", "bytes"])
@pytest.mark.asyncio
async def test_tokenizer_and_serialized_budget_refusals(tmp_path: Path, change: str) -> None:
    p = plan()
    tokenizer = Tokenizer()
    if change == "left":
        tokenizer.padding_side = "left"
    elif change == "not_fast":
        tokenizer.is_fast = False
    elif change == "missing_pad":
        tokenizer.pad_token_id = cast("Any", None)
    else:
        p = TrainingPopulationPlanV2.model_validate(
            {
                **p.model_dump(),
                "limits": PopulationLimits(serialized_bytes=1, disk_bytes=32 * 1024**2),
            }
        )
    with pytest.raises(PopulationError):
        await prepare_training_population(
            p,
            source=Source(p),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
            attempt_dir=tmp_path / "invalid",
        )


@pytest.mark.asyncio
async def test_duplicate_keys_and_null_auxiliary_semantics_are_explicit(tmp_path: Path) -> None:
    p = plan(1)
    body = b'{"item_id":"one","item_id":"two","text":"a","label":"yes"}\n'
    with pytest.raises(PopulationError):
        await prepare_training_population(
            p,
            source=BytesSource(p.source, body),
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "duplicate",
        )
    assert receipt(tmp_path / "duplicate")["status"] == "invalid"


@pytest.mark.asyncio
async def test_failed_final_authority_never_leaves_complete_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, tokenizer = plan(), Tokenizer()
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "attempt",
    )
    fake = CountingBackend()
    monkeypatch.setattr("strata_forge.training.population_runner._backend", _fixed_backend(fake))
    with pytest.raises(PopulationError, match=r"^population_authority_denied$"):
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority("after_train"),
        )
    final = json.loads((prepared.directory / "epoch-receipt.json").read_bytes())
    assert final["status"] == "authority_denied"
    assert final["confirmed_applied_rows"] == 80
    assert fake.steps == 3


@pytest.mark.asyncio
async def test_cancel_native_scan_drains_before_return_with_no_late_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from strata_forge.training import population_runner

    entered, release = threading.Event(), threading.Event()
    original = population_runner._encode_row  # pyright: ignore[reportPrivateUsage] — hold actual native encoding leaf

    def held(*args: Any) -> Any:
        entered.set()
        release.wait(timeout=5)
        return original(*args)

    monkeypatch.setattr(population_runner, "_encode_row", held)
    source = Source(plan())
    task = asyncio.create_task(
        prepare_training_population(
            plan(),
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "held",
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = receipt(tmp_path / "held")
    assert result["status"] == "cancelled"
    assert source.closed
    before = (tmp_path / "held" / "cache.sqlite").stat().st_mtime_ns
    await asyncio.sleep(0.01)
    assert (tmp_path / "held" / "cache.sqlite").stat().st_mtime_ns == before


@pytest.mark.asyncio
async def test_primary_ce_uses_actual_partial_batch_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strata_forge.training import _population_native
    from strata_forge.training._classifier_heads import initialize_classifier
    from tests.unit.training.test_classifier import stack
    from tests.unit.training.test_verifier_classifier import tokenizer as native_tokenizer

    torch, transformers = stack()
    if torch is None:
        return
    torch.set_num_threads(1)
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=128,
        )
    )
    initialized = await initialize_classifier(
        model, labels=("yes", "no"), initializer_id="encoder_random_head", seed=13
    )
    p = plan(33)
    before = initialized.model.classifier.weight.detach().clone()
    sizes: list[int] = []
    old_loss = _population_native._loss  # pyright: ignore[reportPrivateUsage] — capture actual primary loss denominator

    def loss(logits: Any, targets: Any, weight: float) -> Any:
        sizes.append(int(targets.shape[0]))
        return old_loss(logits, targets, weight)

    monkeypatch.setattr(_population_native, "_loss", loss)
    tokenizer = native_tokenizer(True)
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "primary",
    )
    result = await train_prepared_population(
        prepared, initialized=initialized, tokenizer=resolved(tokenizer), authority=Authority()
    )
    assert sizes == [32, 1]
    assert result.confirmed_applied_rows == 33
    assert result.confirmed_optimizer_steps == 2
    assert not torch.equal(before, initialized.model.classifier.weight)


@pytest.mark.asyncio
async def test_order_is_seeded_once_and_different_seed_changes_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orders = []
    for number, seed in enumerate((13, 13, 14)):
        p, tokenizer, fake = plan(seed=seed), Tokenizer(), CountingBackend()
        monkeypatch.setattr(
            "strata_forge.training.population_runner._backend", _fixed_backend(fake)
        )
        prepared = await prepare_training_population(
            p,
            source=Source(p),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
            attempt_dir=tmp_path / str(number),
        )
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
        )
        orders.append(fake.ids)
    assert orders[0] == orders[1]
    assert orders[0] != orders[2]
    assert sorted(orders[0]) == [f"r{index:06d}" for index in range(80)]


@pytest.mark.asyncio
async def test_issued_handle_cannot_be_coherently_rebound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, tokenizer = plan(), Tokenizer()
    first = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "first",
    )
    second = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "second",
    )
    object.__setattr__(first, "directory", second.directory)
    object.__setattr__(first, "receipt", second.receipt)
    object.__setattr__(first, "receipt_sha256", second.receipt_sha256)

    def refuse(*_args: Any) -> None:
        pytest.fail("optimizer must not exist")

    monkeypatch.setattr("strata_forge.training.population_runner._backend", refuse)
    with pytest.raises(PopulationError, match=r"^population_cache_invalid$"):
        await train_prepared_population(
            first,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
        )


@pytest.mark.parametrize(
    ("field", "value"), [("schema_version", True), ("overflow", "drop"), ("seed", False)]
)
@pytest.mark.asyncio
async def test_frozen_model_copy_is_revalidated_before_source_read(
    tmp_path: Path, field: str, value: Any
) -> None:
    p = plan().model_copy(update={field: value})
    source = Source(p)
    with pytest.raises(PopulationError):
        await prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "bad",
        )
    assert source.reads == 0
    assert source.closed


@pytest.mark.parametrize(
    "text",
    [
        "NaN",
        "{",
        "[]",
        '{"item_id":"x","text":"a","label":false}',
        '{"item_id":"x","text":"a","label":"missing"}',
    ],
)
@pytest.mark.asyncio
async def test_malformed_source_values_are_fixed_and_content_free(
    tmp_path: Path, text: str
) -> None:
    p = plan(1)
    with pytest.raises(PopulationError, match=r"^population_source_invalid$") as caught:
        await prepare_training_population(
            p,
            source=BytesSource(p.source, text.encode() + b"\n"),
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "bad",
        )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_cache_change_during_authority_await_is_caught_before_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, tokenizer = plan(), Tokenizer()
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "cache",
    )

    class MutatingAuthority(Authority):
        async def check(self, plan: TrainingPopulationPlanV2, *, phase: str) -> None:
            if phase == "before_train":
                with (prepared.directory / "cache.sqlite").open("ab") as file:
                    file.write(b"changed")

    def refuse(*_args: Any) -> None:
        pytest.fail("optimizer must not exist")

    monkeypatch.setattr("strata_forge.training.population_runner._backend", refuse)
    with pytest.raises(PopulationError, match=r"^population_cache_invalid$"):
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=MutatingAuthority(),
        )


@pytest.mark.parametrize("change", ["heads", "pids", "missing_flag", "missing_space"])
@pytest.mark.asyncio
async def test_auxiliary_targets_never_guess_missing_or_ineligible_fields(
    tmp_path: Path, change: str
) -> None:
    from strata_forge.training.population import AuxiliaryPopulationTask
    from tests.unit.training.test_classifier import stack
    from tests.unit.training.test_verifier_classifier import tokenizer as native_tokenizer
    from tests.unit.training.test_verifier_geometry import geometry

    torch, _ = stack()
    if torch is None:
        return
    g = geometry(
        "cls",
        labels=("E", "C", "U"),
        outputs=("label",) if change == "heads" else ("label", "inconsistent", "support_set"),
    )
    p = TrainingPopulationPlanV2.model_validate(
        {
            **plan(1).model_dump(),
            "task": AuxiliaryPopulationTask(
                codec="auxiliary_verifier_v1", geometry=g, class_order=("E", "C", "U")
            ),
        }
    )
    row = {
        "example": {"item_id": "x", "premises": [{"pid": "p1", "text": "a"}], "hypothesis": "q"},
        "targets": {
            "primary": {"label_space": "3way", "label_dist": [1.0, 0.0, 0.0]},
            "flag_target": False,
            "premise_ids": ["p1"],
            "support_set": [],
        },
    }
    target = cast("dict[str, Any]", row["targets"])
    if change == "pids":
        target["premise_ids"] = ["p2"]
    elif change == "missing_flag":
        target.pop("flag_target")
    elif change == "missing_space":
        target["primary"].pop("label_space")
    with pytest.raises(PopulationError, match=r"^population_source_invalid$"):
        await prepare_training_population(
            p,
            source=BytesSource(p.source, json.dumps(row).encode() + b"\n"),
            tokenizer=resolved(native_tokenizer(True)),
            authority=Authority(),
            attempt_dir=tmp_path / "bad",
        )


@pytest.mark.parametrize("change", ["wrong_type", "labels", "seed", "pad", "family"])
@pytest.mark.asyncio
async def test_native_binding_refuses_before_optimizer_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from strata_forge.training._classifier_heads import initialize_classifier
    from tests.unit.training.test_classifier import stack
    from tests.unit.training.test_verifier_classifier import tokenizer as native_tokenizer

    torch, transformers = stack()
    if torch is None:
        return
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=128,
        )
    )
    initialized = await initialize_classifier(
        model, labels=("yes", "no"), initializer_id="encoder_random_head", seed=13
    )
    p, tokenizer = plan(1), native_tokenizer(True)
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "bad",
    )
    if change == "wrong_type":
        initialized = cast("Any", None)
    elif change == "labels":
        initialized.model.config.id2label[0] = "wrong"
    elif change == "seed":
        object.__setattr__(initialized.receipt, "seed", 14)
    elif change == "pad":
        initialized.model.config.pad_token_id = 17
    else:
        initialized.model.config.model_type = "gpt2"

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("optimizer must not be constructed")

    monkeypatch.setattr(torch.optim, "AdamW", refuse)
    with pytest.raises(PopulationError, match=r"^population_training_unsupported$"):
        await train_prepared_population(
            prepared, initialized=initialized, tokenizer=resolved(tokenizer), authority=Authority()
        )


@pytest.mark.parametrize("stage", ["apply", "scheduler"])
@pytest.mark.asyncio
async def test_optimizer_failure_records_confirmed_rows_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    p, tokenizer = plan(33), Tokenizer()
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "failed",
    )

    class Failing(CountingBackend):
        def apply(self, records: Any) -> None:
            if stage == "apply":
                raise RuntimeError("PRIVATE_OPTIMIZER_CANARY")
            super().apply(records)

        def after_step(self) -> None:
            raise RuntimeError("PRIVATE_OPTIMIZER_CANARY")

    monkeypatch.setattr(
        "strata_forge.training.population_runner._backend", _fixed_backend(Failing())
    )
    with pytest.raises(PopulationError, match=r"^population_training_failed$") as caught:
        await train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
        )
    result = json.loads((prepared.directory / "epoch-receipt.json").read_bytes())
    assert result["status"] == "failed"
    assert result["optimizer_in_doubt"] == (stage == "apply")
    assert result["confirmed_applied_rows"] == (0 if stage == "apply" else 32)
    assert "PRIVATE" not in json.dumps(result)
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_wrong_resolved_tokenizer_pin_denies_before_source_or_optimizer(
    tmp_path: Path,
) -> None:
    p, source = plan(), Source(plan())
    wrong = ResolvedPopulationTokenizer(
        PopulationIdentity(repo_type="model", ref="fictional/other@" + "d" * 40), Tokenizer()
    )
    with pytest.raises(PopulationError, match=r"^population_tokenizer_invalid$"):
        await prepare_training_population(
            p, source=source, tokenizer=wrong, authority=Authority(), attempt_dir=tmp_path / "wrong"
        )
    assert source.reads == 0
    assert source.closed
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(Tokenizer()),
        authority=Authority(),
        attempt_dir=tmp_path / "right",
    )
    with pytest.raises(PopulationError, match=r"^population_tokenizer_invalid$"):
        await train_prepared_population(
            prepared, initialized=cast("Any", None), tokenizer=wrong, authority=Authority()
        )


@pytest.mark.asyncio
async def test_cancel_actual_step_drains_and_never_starts_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    p, tokenizer = plan(), Tokenizer()
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "epoch",
    )
    entered, release = threading.Event(), threading.Event()

    class Held(CountingBackend):
        def apply(self, records: Any) -> None:
            entered.set()
            release.wait(timeout=5)
            super().apply(records)

    fake = Held()
    monkeypatch.setattr("strata_forge.training.population_runner._backend", _fixed_backend(fake))
    task = asyncio.create_task(
        train_prepared_population(
            prepared,
            initialized=cast("Any", None),
            tokenizer=resolved(tokenizer),
            authority=Authority(),
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    final = json.loads((prepared.directory / "epoch-receipt.json").read_bytes())
    assert final["status"] == "cancelled"
    assert final["confirmed_applied_rows"] == 32
    assert fake.batches == [32]


@pytest.mark.asyncio
async def test_source_close_cancellation_is_drained(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Closing(Source):
        async def aclose(self) -> None:
            entered.set()
            await release.wait()
            self.closed = True

    source = Closing(plan())
    task = asyncio.create_task(
        prepare_training_population(
            plan(),
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "closing",
        )
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed
    assert receipt(tmp_path / "closing")["status"] == "cancelled"


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": 1},
        {"seed": True},
        {"overflow": "drop"},
        {"family": "modernbert"},
        {"model": {"repo_type": "dataset", "ref": "fictional/model@" + "b" * 40}},
        {"tokenizer": {"repo_type": "dataset", "ref": "fictional/tokenizer@" + "c" * 40}},
        {
            "source": {
                "repo_type": "model",
                "ref": "fictional/source@" + "a" * 40,
                "config": "default",
                "split": "train",
                "expected_rows": 80,
            }
        },
        {"selection": {"mode": "range", "start": 10, "end": 10}},
        {"selection": {"mode": "range", "start": 0, "end": 81}},
        {"limits": {"serialized_bytes": 8 * 1024**3 + 1, "disk_bytes": 32 * 1024**2}},
        {"limits": {"serialized_bytes": 1, "disk_bytes": 64 * 1024**3 + 1}},
    ],
)
def test_plan_boundaries_do_not_widen_v1_or_accept_unbounded_options(
    updates: dict[str, Any],
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrainingPopulationPlanV2.model_validate({**plan().model_dump(), **updates})


@pytest.mark.parametrize(
    "ref",
    [
        "fictional/repo@main",
        "fictional/repo@" + "A" * 40,
        "fictional/a--b@" + "a" * 40,
        "fictional/a..b@" + "a" * 40,
        "fictional/repo.git@" + "a" * 40,
        "fictional/repo/extra@" + "a" * 40,
        "fictional/repo@" + "a" * 40 + "\n",
    ],
)
def test_ref_grammar_is_exact_and_immutable(ref: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PopulationIdentity(repo_type="model", ref=ref)


@pytest.mark.parametrize(
    "changes",
    [
        {"epochs": 2.0},
        {"precision": "bf16"},
        {"batch_size": 8},
        {"max_steps": 0},
        {"gradient_checkpointing": True},
        {"class_weights": (1.0, 1.0)},
        {"label_smoothing": 0.1},
        {"source_objective": "nli_binary_grouped"},
        {"warmup_steps": 4},
        {"seed": 14},
    ],
)
def test_unsupported_training_features_refuse_instead_of_choosing_semantics(
    changes: dict[str, Any],
) -> None:
    from pydantic import ValidationError

    value = plan().model_dump()
    value["task"]["config"].update(changes)
    with pytest.raises(ValidationError):
        TrainingPopulationPlanV2.model_validate(value)


@pytest.mark.asyncio
async def test_nonlexical_first_overflow_and_missing_source_identity(tmp_path: Path) -> None:
    p = plan(2)
    content = b"".join(
        json.dumps({"item_id": item_id, "text": "a " * 40, "label": "yes"}).encode() + b"\n"
        for item_id in ("z-first", "a-later")
    )
    with pytest.raises(PopulationError, match=r"^population_overflow$"):
        await prepare_training_population(
            p,
            source=BytesSource(p.source, content),
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "over",
        )
    assert receipt(tmp_path / "over")["first_overflow_item_id"] == "z-first"
    source = Source(p)
    source.identity = PopulationSourceIdentity(
        ref="fictional/other@" + "d" * 40, config="default", split="train", expected_rows=2
    )
    with pytest.raises(PopulationError):
        await prepare_training_population(
            p,
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "other",
        )
    assert source.reads == 0
    assert source.closed


@pytest.mark.asyncio
async def test_cancellation_during_final_cache_hash_rewrites_nonusable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from strata_forge.training import population_runner

    entered, release = threading.Event(), threading.Event()
    original = population_runner.hash_file

    def held(path: Path) -> str:
        entered.set()
        release.wait(timeout=5)
        return original(path)

    monkeypatch.setattr(population_runner, "hash_file", held)
    source = Source(plan())
    task = asyncio.create_task(
        prepare_training_population(
            plan(),
            source=source,
            tokenizer=resolved(Tokenizer()),
            authority=Authority(),
            attempt_dir=tmp_path / "final",
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert receipt(tmp_path / "final")["status"] == "cancelled"
    assert receipt(tmp_path / "final")["cache_sha256"] is None


def test_auxiliary_semantic_order_cannot_contradict_geometry_columns() -> None:
    from pydantic import ValidationError

    from strata_forge.training.population import AuxiliaryPopulationTask
    from tests.unit.training.test_verifier_geometry import geometry

    with pytest.raises(ValidationError):
        AuxiliaryPopulationTask(
            codec="auxiliary_verifier_v1",
            geometry=geometry(labels=("E", "C", "U")),
            class_order=("C", "E", "U"),
        )


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert-decoder"])
@pytest.mark.parametrize("limit", [63, 64, 65])
@pytest.mark.asyncio
async def test_aux_native_capacity_below_equal_and_above_before_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, limit: int
) -> None:
    from strata_forge.training import _population_native
    from strata_forge.training.population import AuxiliaryPopulationTask

    torch, _, model, tokenizer = await population_initialized(family)
    if torch is None:
        return
    task = AuxiliaryPopulationTask(
        codec="auxiliary_verifier_v1", geometry=model.geometry, class_order=("E", "C", "U")
    )
    p = TrainingPopulationPlanV2.model_validate(
        {**plan(1).model_dump(), "family": family, "task": task}
    )
    body = model.model.bert if family == "bert" else model.model.backbone
    if family == "bert":
        model.model.config.max_position_embeddings = limit
        body.embeddings.position_embeddings = torch.nn.Embedding(limit, 8)
    elif family == "gpt2":
        model.model.config.n_positions = limit
        body.wpe = torch.nn.Embedding(limit, 8)
    else:
        model.model.config.max_position_embeddings = limit
        body.rotary_emb.original_max_seq_len = limit
    # Validate only: no forward/optimizer, even for intentionally reduced synthetic contexts.
    if limit < 64:
        with pytest.raises(PopulationError, match=r"^population_training_unsupported$"):
            _population_native._validate_model(p, model, tokenizer)  # pyright: ignore[reportPrivateUsage] — fixed native compatibility boundary
    else:
        _population_native._validate_model(p, model, tokenizer)  # pyright: ignore[reportPrivateUsage] — equal/below capacity positive control


@pytest.mark.parametrize("change", ["actual_limit", "config_lie", "missing_limit", "vocab"])
@pytest.mark.asyncio
async def test_primary_native_context_and_vocab_fail_before_adamw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from strata_forge.training._classifier_heads import initialize_classifier
    from tests.unit.training.test_classifier import stack
    from tests.unit.training.test_verifier_classifier import tokenizer as native_tokenizer

    torch, transformers = stack()
    if torch is None:
        return
    torch.set_num_threads(1)
    base = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8 if change in ("actual_limit", "config_lie") else 64,
        )
    )
    model = await initialize_classifier(
        base, labels=("yes", "no"), initializer_id="encoder_random_head", seed=13
    )
    tokenizer = native_tokenizer(True)
    p = plan(1)
    prepared = await prepare_training_population(
        p,
        source=Source(p),
        tokenizer=resolved(tokenizer),
        authority=Authority(),
        attempt_dir=tmp_path / "native",
    )
    if change == "config_lie":
        model.model.config.max_position_embeddings = 64
    elif change == "missing_limit":
        object.__setattr__(model.model.config, "max_position_embeddings", None)
    elif change == "vocab":
        model.model.bert.embeddings.word_embeddings = torch.nn.Embedding(4, 8)
    calls = []

    def refuse(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("optimizer forbidden")

    monkeypatch.setattr(torch.optim, "AdamW", refuse)
    with pytest.raises(PopulationError, match=r"^population_training_unsupported$"):
        await train_prepared_population(
            prepared, initialized=model, tokenizer=resolved(tokenizer), authority=Authority()
        )
    assert calls == []
