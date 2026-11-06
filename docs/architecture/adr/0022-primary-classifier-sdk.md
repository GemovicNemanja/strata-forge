# ADR 0022: primary classifier training starts with resolved SDK objects

## Decision

Expose an async primary-classifier SDK independently of the causal-method registry. A caller
supplies already-resolved native models, tokenizer and rows. The SDK performs no Hub read,
write or remote launch. Its generic objective is ordinary cross-entropy. An explicit Brier
coefficient and named CE+0.1 Brier research preset record a different requested objective.
Undefined grouped NLI binary losses, weighting/smoothing and auxiliary roles refuse.

Fresh heads are an explicit trusted in-memory initialization, separate from complete-artifact
reload. Fixed BERT/GPT2/ModernBertDecoder adapters copy the complete body strictly and enumerate
exact head keys. Decoder initialization selects actual vocabulary projection rows, copies
prediction transforms and output bias, and checks step-zero probabilities at the same last
attended answer positions. A Forge-installed decoder wrapper preserves bias and pooling;
no repository code or wildcard key exemption is accepted. Receipts describe the transformation
of caller-resolved objects and do not attest Hub provenance or authorize access.

Final checkpoint export is the supported selection policy. Marked gate/test sources cannot
select checkpoints. Safe artifacts contain safetensors/config/tokenizer and bounded metadata;
local reload uses fixed installed classes and the strict completeness guard, including after
fresh initialization. Training/evaluation methods and worker authority remain separate.

## Consequences

The SDK is useful without pretending its worker integration exists. Decoder artifacts reload
with Forge `load_classifier`; generic AutoModel and the live HF scoring dispatch do not yet
recognize the wrapper. No seq_cls registry/launcher activation, custody-policy scope expansion,
LoRA merge, calibrated outputs, research data or real-model acceptance is implied. Those require
separate concrete increments and consumer review. Coroutine cancellation cooperatively requests
Trainer stop; supervising a process remains necessary to terminate a stuck native operation.
