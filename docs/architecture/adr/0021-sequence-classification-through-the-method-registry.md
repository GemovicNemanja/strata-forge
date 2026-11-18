# ADR 0021: Sequence classification enters through the method registry

**Status:** Accepted

## Context

Every registered fine-tuning method trained a causal-LM head through TRL. A classifier needs a
different head, a different objective and checks a causal-LM run does not: which checkpoint keys
may be freshly initialised, whether a decoder head built from the LM head reproduces the base
model's zero-shot label logits, and whether the longest real batch fits the device. It also needs
a population of hundreds of thousands of rows, which the TRL path materialises as a Python list.

A second training stack beside the registry (its own spec, worker and control-plane route) would
duplicate the pinning, secret delivery, progress protocol, push and record that the fine-tuning
runner already provides, and drift from them.

## Decision

Sequence classification is the registry entry `seq_cls` (`task_type="SEQ_CLS"`, dataset format
`nli_pair`), with its own config and runner (`SeqClsConfig`, `SeqClsRunner`) on
`transformers.Trainer`. The fine-tuning runner keeps one spec and one record and branches on the
task type only where the work differs: a classifier reads Arrow splits projected onto the roles,
loads a pinned local snapshot, and runs its checks before step 0.

The checks are part of the method, not options: a strict load guard driven by explicit
per-architecture tables, a decoder step-0 equivalence check, an overflow refusal (never
truncation), and a memory preflight at the longest real batch. Gradient accumulation is refused,
because the objective's means are over the declared batch. The artifact is a stock
sequence-classification checkpoint plus fixed-name auxiliary heads and a rendering config, so an
evaluator needs no repository code.

Every fine-tuning run, whatever its method, records a `strata.run.manifest/v1` manifest of kind
`finetune`, pushed with the artifact in one commit to a private repo.

## Consequences

The control plane validates a classifier spec with the engine's own registry and config at submit,
as it already does for the other methods. A new classifier architecture is a table entry in
`classifier_heads.py` plus a test on a tiny synthetic checkpoint, never a key-name heuristic.
Adapters for `seq_cls`, resuming from a checkpoint and multi-GPU training are not part of this
decision; each would change the method's contract and gets its own.
