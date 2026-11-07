"""Actual random CPU checkpoints/forward/Parquet through both native population workers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.pipelines import eval_runner, inference_runner
from strata_forge.pipelines._evaluation_reader import _parquet_rows
from strata_forge.pipelines.evaluation_population_spec import EvaluationPopulationSpecV2
from strata_forge.pipelines.scoring_population_spec import InferencePopulationSpecV2
from tests.unit.pipelines.test_population_workers import Policy, Reader, job
from tests.unit.training.test_classifier import tokenizer as tiny_tokenizer

if TYPE_CHECKING:
    from pathlib import Path

# Actual installed ML factories and serialized tokenizer/model kwargs are intentionally dynamic.


def native_checkpoint(path: Path, family: str) -> None:
    import torch
    import transformers

    torch.set_num_threads(1)
    torch.manual_seed(7)
    if family == "bert":
        model = transformers.BertForSequenceClassification(
            transformers.BertConfig(
                vocab_size=8,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=32,
                num_labels=3,
                id2label={0: "E", 1: "C", 2: "U"},
                label2id={"E": 0, "C": 1, "U": 2},
                pad_token_id=0,
            )
        )
    else:
        model = transformers.GPT2LMHeadModel(
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
    model.save_pretrained(path, safe_serialization=True)
    tokenizer = tiny_tokenizer()
    tokenizer.eos_token = "[PAD]"
    tokenizer.save_pretrained(path)


def route_native_factories(monkeypatch: pytest.MonkeyPatch, path: Path) -> dict[str, list[Any]]:
    import httpx
    import huggingface_hub
    import transformers
    from huggingface_hub.errors import RemoteEntryNotFoundError

    observations: dict[str, list[Any]] = {"factories": [], "encoded": [], "forwards": []}

    def absent(
        repo_id: str, filename: str, *, revision: str, token: object, dry_run: bool = False
    ) -> Any:
        assert (repo_id, filename, revision, token, dry_run) == (
            "example/synthetic-model",
            "strata_classifier.json",
            "b" * 40,
            False,
            True,
        )
        response = httpx.Response(
            404,
            request=httpx.Request(
                "HEAD",
                "https://huggingface.co/example/synthetic-model/resolve/"
                + "b" * 40
                + "/strata_classifier.json",
            ),
        )
        raise RemoteEntryNotFoundError("fictional sidecar absent", response=response)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", absent)
    tokenizer_factory = transformers.AutoTokenizer.from_pretrained

    def load_tokenizer(repo_id: str, **kwargs: Any) -> Any:
        assert repo_id == "example/synthetic-model"
        assert kwargs["revision"] == "b" * 40
        assert kwargs["token"] is False
        assert kwargs["trust_remote_code"] is False
        observations["factories"].append("tokenizer")
        result = tokenizer_factory(
            path, **kwargs, local_files_only=True, cache_dir=path / "owned-cache"
        )
        encode = result.encode

        def observed(text: str, **options: Any) -> Any:
            observations["encoded"].append(text)
            return encode(text, **options)

        monkeypatch.setattr(result, "encode", observed)
        return result

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    for factory in (
        transformers.AutoModelForSequenceClassification,
        transformers.AutoModelForCausalLM,
    ):
        original = factory.from_pretrained

        def load(repo_id: str, *, _factory: Any = original, **kwargs: Any) -> Any:
            assert repo_id == "example/synthetic-model"
            assert kwargs["revision"] == "b" * 40
            assert kwargs["token"] is False
            assert kwargs["trust_remote_code"] is False
            assert kwargs["use_safetensors"] is True
            assert kwargs["output_loading_info"] is True
            observations["factories"].append("weights")
            model, audit = _factory(
                path, **kwargs, local_files_only=True, cache_dir=path / "owned-cache"
            )

            def forward(module: object, args: object, options: Any) -> None:
                observations["forwards"].extend(options["attention_mask"].sum(-1).tolist())

            model.register_forward_pre_hook(forward, with_kwargs=True)
            return model, audit

        monkeypatch.setattr(factory, "from_pretrained", load)
    return observations


class ParquetReader(Reader):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    async def rows(
        self, ref: str, config: str, split: str, selection: Any, columns: tuple[str, ...]
    ) -> tuple[dict[str, object], ...]:
        assert ref == "example/synthetic-gold@" + "a" * 40
        return _parquet_rows((self.path,), selection, columns)


@pytest.mark.parametrize("family", ["bert", "gpt2"])
@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
async def test_actual_default_native_scorer_cpu_and_bounded_parquet(
    name: str, family: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    native = tmp_path / "model"
    native_checkpoint(native, family)
    observed = route_native_factories(monkeypatch, native)
    raw = job(name).wire_payload()
    if name == "inference":
        hp = raw["job"]["hyperparams"]
        hp["label_alias"] = None
        if family == "gpt2":
            hp.update(
                engine="hf_label_logits",
                pair_template=None,
                label_tokens={"E": "a", "C": "b", "U": "c"},
            )
            raw["job"]["template"] = "{p} {h}"
        value = InferencePopulationSpecV2.model_validate(raw)
    else:
        raw["job"]["label_aliases"] = {}
        if family == "gpt2":
            raw["job"].update(
                engine="hf_label_logits",
                label_tokens={"E": "a", "C": "b", "U": "c"},
                label_aliases=None,
            )
            raw["job"]["rendering"].update(template="{premises} {hypothesis}", pair_template=None)
        value = EvaluationPopulationSpecV2.model_validate(raw)
    data = []
    for item_id, p, premises, eligible in [
        ("short", "a", ["a"], True),
        ("exact", "a " * 6, ["a " * 4], True),
        ("overflow", "a " * 12, ["a " * 12], True),
        ("excluded", "EXCLUDED" + "a" * 200001, ["EXCLUDED" + "a" * 200001], False),
        ("invalid", "a" * 200001, ["a" * 200001], True),
    ]:
        data.append(
            {
                "item_id": item_id,
                "tier": "a",
                "within_limits": eligible,
                "premise_text": p,
                "premises": premises,
                "hypothesis": "b",
                "gold": "E" if eligible else "EXCLUDED_GOLD_IS_NOT_SCORED",
            }
        )
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(data), parquet)
    reader = ParquetReader(parquet)
    policy = Policy()
    if name == "inference":
        result = await inference_runner.execute_population_scoring(
            value, policy=policy, reader=reader, output_dir=tmp_path / "result"
        )
        table = pq.read_table(result.directory / "results.parquet").to_pylist()
    else:
        result = await eval_runner.run_population_evaluation(
            value, policy=policy, reader=reader, output_dir=tmp_path / "result"
        )
        table = pq.read_table(result.directory / "evaluations/threeway/items.parquet").to_pylist()
    assert observed["factories"] == ["tokenizer", "weights"]
    assert len(observed["forwards"]) == 2
    assert max(observed["forwards"]) == 7
    assert all(length <= 7 for length in observed["forwards"])
    assert not any("EXCLUDED" in text for text in observed["encoded"])
    assert [row["outcome"] for row in table] == [
        "scored",
        "scored",
        "overflow",
        "suite_excluded",
        "invalid",
    ]
    assert table[1]["token_count"] == 7
    assert table[2]["token_count"] > 7
    assert result.counts.rows_in == 5
    assert result.counts.rows_metric_population == 3
    assert result.counts.rows_scored_success == 2
    assert result.counts.rows_failed_or_invalid == 1
    assert {digest for _, _, digest in policy.calls} == {value.digest()}
    for row in table[:2]:
        assert sum(row[f"p_{label}"] for label in ("E", "C", "U")) == pytest.approx(1, abs=1e-6)
        assert row["truncated"] is False
        assert row["model_sent"] is True
    manifest = json.loads(result.report_path.read_text())
    if name == "native_evaluation":
        entry = manifest["evaluations"][0]
        assert entry["classification"]["accuracy"]["n"] == 3
        assert entry["metrics"]["brier_raw"]["result"]["n"] == 2
        assert entry["source"]["loading_info"]["status"] == "accepted"
    else:
        assert manifest["loading_info"]["status"] == "accepted"
        assert manifest["overflow_evidence"] == "native_computed"
