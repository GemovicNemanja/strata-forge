"""Fine-tuning: SFT, preference tuning (DPO/ORPO/KTO/GRPO), and PEFT (LoRA/QLoRA).

The public surface is:

- :class:`ClassifierConfig`, :class:`ClassifierExample`, :class:`ClassifierRunner`,
  :class:`ClassifierRunResult`, :class:`InitializedClassifier`, :func:`initialize_classifier`
  and :func:`load_classifier` — primary classification over resolved objects, separate from
  worker registration.

- :class:`SFTConfig` / :class:`SFTRunner` / :class:`SFTRunResult`
- :class:`DPOConfig` / :class:`ORPOConfig` / :class:`KTOConfig` /
  :class:`GRPOConfig` / :class:`PreferenceRunner` /
  :class:`PreferenceRunResult`
- :class:`LoRAConfig` / :class:`QLoRAConfig`, and
  :func:`require_bitsandbytes` / :class:`MissingBitsAndBytesError`
  -- the up-front check that a QLoRA run can quantise its model.
- :func:`apply_chat_template` / :func:`conversation_to_dicts` /
  :func:`conversation_to_text`
- :func:`pack_sequences` / :class:`PackedSequence`
- :data:`METHODS` / :class:`MethodSpec` / :func:`pick_method` /
  :func:`enabled_methods` / :func:`check_format` /
  :class:`UnsupportedMethodError` — the method registry, for
  callers that drive training from a declaration rather than
  from Python.
- :data:`FORMATS` / :class:`FormatSpec` / :func:`pick_format` /
  :func:`validate_mapping` / :func:`build_training_rows` /
  :func:`sft_text_field` / :class:`DatasetFormatError` — the
  dataset-shape declaration those callers map columns with.

Heavy ML dependencies (``torch``, ``transformers``, ``trl``,
``peft``, ``datasets``, ``bitsandbytes``) ride behind the ``[finetuning]`` extra
and are lazy-imported inside the runners' ``train`` methods, so
``import strata_forge.training`` works without them.
"""

from strata_forge.training.chat_template import (
    apply_chat_template,
    conversation_to_dicts,
    conversation_to_text,
)
from strata_forge.training.classifier import (
    ClassifierConfig,
    ClassifierExample,
    ClassifierRunner,
    ClassifierRunResult,
    InitializedClassifier,
    initialize_classifier,
    load_classifier,
)
from strata_forge.training.dataset_format import (
    FORMATS,
    DatasetFormat,
    DatasetFormatError,
    FormatSpec,
    build_training_rows,
    pick_format,
    sft_text_field,
    validate_mapping,
)
from strata_forge.training.loading import HubToken, model_load_kwargs, tokenizer_load_kwargs
from strata_forge.training.methods import (
    METHODS,
    MethodName,
    MethodSpec,
    UnsupportedMethodError,
    check_format,
    enabled_methods,
    pick_method,
)
from strata_forge.training.packing import PackedSequence, pack_sequences
from strata_forge.training.peft import (
    LoRAConfig,
    MissingBitsAndBytesError,
    QLoRAConfig,
    require_bitsandbytes,
)
from strata_forge.training.population import (
    AuxiliaryPopulationTask,
    CompletePopulation,
    PopulationAuthority,
    PopulationEpochReceipt,
    PopulationError,
    PopulationIdentity,
    PopulationLimits,
    PopulationOptimizerSettings,
    PopulationPreparationReceipt,
    PopulationRange,
    PopulationSourceIdentity,
    PreparedTrainingPopulation,
    PrimaryPopulationTask,
    ResolvedPopulationTokenizer,
    TrainingPopulationPlanV2,
    TrustedPopulationSource,
)
from strata_forge.training.population_runner import (
    prepare_training_population,
    train_prepared_population,
)
from strata_forge.training.preference import (
    AnyPreferenceConfig,
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    ORPOConfig,
    PreferenceRunner,
    PreferenceRunResult,
)
from strata_forge.training.progress import (
    JsonlProgressWriter,
    ProgressEvent,
    trainer_callback,
)
from strata_forge.training.sft import SFTConfig, SFTRunner, SFTRunResult
from strata_forge.training.verifier_classifier import (
    AuxiliaryInitializationReceipt,
    InitializedVerifierClassifier,
    VerifierBatch,
    VerifierPrediction,
    encode_verifier_batch,
    export_verifier_classifier,
    initialize_verifier_classifier,
    load_verifier_classifier,
    score_verifier_batch,
)
from strata_forge.training.verifier_config import (
    RenderedVerifierInput,
    VerifierConfig,
    render_verifier_input,
)
from strata_forge.training.verifier_geometry import (
    RenderedVerifierGeometry,
    VerifierExample,
    VerifierGeometryConfig,
    VerifierGeometryError,
    VerifierPremise,
    render_verifier_geometry,
    resolve_pid_markers,
)

__all__ = [
    "FORMATS",
    "METHODS",
    "AnyPreferenceConfig",
    "AuxiliaryInitializationReceipt",
    "AuxiliaryPopulationTask",
    "ClassifierConfig",
    "ClassifierExample",
    "ClassifierRunResult",
    "ClassifierRunner",
    "CompletePopulation",
    "DPOConfig",
    "DatasetFormat",
    "DatasetFormatError",
    "FormatSpec",
    "GRPOConfig",
    "HubToken",
    "InitializedClassifier",
    "InitializedVerifierClassifier",
    "JsonlProgressWriter",
    "KTOConfig",
    "LoRAConfig",
    "MethodName",
    "MethodSpec",
    "MissingBitsAndBytesError",
    "ORPOConfig",
    "PackedSequence",
    "PopulationAuthority",
    "PopulationEpochReceipt",
    "PopulationError",
    "PopulationIdentity",
    "PopulationLimits",
    "PopulationOptimizerSettings",
    "PopulationPreparationReceipt",
    "PopulationRange",
    "PopulationSourceIdentity",
    "PreferenceRunResult",
    "PreferenceRunner",
    "PreparedTrainingPopulation",
    "PrimaryPopulationTask",
    "ProgressEvent",
    "QLoRAConfig",
    "RenderedVerifierGeometry",
    "RenderedVerifierInput",
    "ResolvedPopulationTokenizer",
    "SFTConfig",
    "SFTRunResult",
    "SFTRunner",
    "TrainingPopulationPlanV2",
    "TrustedPopulationSource",
    "UnsupportedMethodError",
    "VerifierBatch",
    "VerifierConfig",
    "VerifierExample",
    "VerifierGeometryConfig",
    "VerifierGeometryError",
    "VerifierPrediction",
    "VerifierPremise",
    "apply_chat_template",
    "build_training_rows",
    "check_format",
    "conversation_to_dicts",
    "conversation_to_text",
    "enabled_methods",
    "encode_verifier_batch",
    "export_verifier_classifier",
    "initialize_classifier",
    "initialize_verifier_classifier",
    "load_classifier",
    "load_verifier_classifier",
    "model_load_kwargs",
    "pack_sequences",
    "pick_format",
    "pick_method",
    "prepare_training_population",
    "render_verifier_geometry",
    "render_verifier_input",
    "require_bitsandbytes",
    "resolve_pid_markers",
    "score_verifier_batch",
    "sft_text_field",
    "tokenizer_load_kwargs",
    "train_prepared_population",
    "trainer_callback",
    "validate_mapping",
]
