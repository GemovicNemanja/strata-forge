"""Tiny synthetic encoder and decoder checkpoints, built offline on CPU, for the classifier tests.

Every weight is random and every row is generated; nothing here comes from a real model or a real
dataset. The decoder copies the layout of a ModernBERT decoder checkpoint (``lm_head.dense``,
``lm_head.norm``, an untied ``decoder`` projection with a nonzero bias), so the head-init mapping
and the load guard are exercised on the key names a real conversion carries.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

WORDS = [f"w{i}" for i in range(40)]
VERBALIZERS = {"entailment": " True", "contradiction": " False", "neutral": " Unknown"}
LABELS = ("entailment", "contradiction", "neutral")
#: How a two-way row spells its positive and negative, declared explicitly by every config.
TWO_WAY = {"two_way_positive_label": "entailment", "two_way_labels": ("entailed", "not_entailed")}
_SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
_EXTRA = ["True", "False", "Unknown", ":", "p1", "p2", "p3", "1", "2", "3"]


def tokenizer(directory: Path) -> Any:
    """A WordLevel fast tokenizer with [CLS]/[SEP] framing and character offsets."""
    tokenizers: Any = __import__("tokenizers")
    transformers: Any = __import__("transformers")
    vocab = {tok: i for i, tok in enumerate([*_SPECIAL, *_EXTRA, *WORDS])}
    model = tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    tok = tokenizers.Tokenizer(model)
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    fast.save_pretrained(str(directory))
    return fast


def _common(vocab_size: int) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "cls_token_id": 2,
        "sep_token_id": 3,
        "global_attn_every_n_layers": 2,
        "local_attention": 16,
    }


def encoder(directory: Path, *, seed: int = 0, pooling: str = "cls") -> Path:
    """A ModernBERT masked-LM checkpoint: the shape of a classifier-less encoder base.

    ``pooling="mean"`` writes the ``classifier_pooling`` the Ettin encoder configs carry.
    """
    torch: Any = __import__("torch")
    transformers: Any = __import__("transformers")
    tok = tokenizer(directory)
    torch.manual_seed(seed)
    config = transformers.ModernBertConfig(**_common(len(tok)), classifier_pooling=pooling)
    model = transformers.ModernBertForMaskedLM(config)
    model.save_pretrained(str(directory), safe_serialization=True)
    return directory


def decoder(directory: Path, *, seed: int = 0, tied: bool = False) -> Path:
    """A ModernBERT-decoder causal-LM checkpoint with a nonzero output-projection bias."""
    torch: Any = __import__("torch")
    transformers: Any = __import__("transformers")
    tok = tokenizer(directory)
    torch.manual_seed(seed)
    config = transformers.ModernBertDecoderConfig(
        **{k: v for k, v in _common(len(tok)).items() if k not in ("global_attn_every_n_layers",)},
        tie_word_embeddings=tied,
    )
    model = transformers.ModernBertDecoderForCausalLM(config)
    with torch.no_grad():
        model.decoder.bias.normal_(0.0, 2.0)
        model.lm_head.norm.weight.normal_(1.0, 0.3)
    model.save_pretrained(str(directory), safe_serialization=True)
    return directory


def rows(n: int, *, seed: int = 0, two_way_every: int = 4) -> list[dict[str, Any]]:
    """Generated nli_pair rows with every role, nulls included; labels by a simple synthetic rule."""
    rng = random.Random(seed)  # noqa: S311 — synthetic fixture, not a security context.
    out: list[dict[str, Any]] = []
    for i in range(n):
        premises = [
            {"pid": f"p{j + 1}", "text": " ".join(rng.choices(WORDS, k=rng.randint(2, 5)))}
            for j in range(rng.randint(1, 3))
        ]
        hypothesis = " ".join(rng.choices(WORDS, k=rng.randint(2, 4)))
        label = LABELS[i % 3]
        two_way = i % two_way_every == 0
        out.append(
            {
                "premises": premises,
                "hypothesis": hypothesis,
                "label": ("entailed" if label == "entailment" else "not_entailed")
                if two_way
                else label,
                "label_space": "2way_entail_vs_not" if two_way else "3way",
                "inconsistent_premises": None if i % 5 == 0 else bool(i % 2),
                "support_set": None if label == "neutral" or two_way else [premises[0]["pid"]],
            }
        )
    return out
