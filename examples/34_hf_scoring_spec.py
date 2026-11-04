"""Build an inert scoring spec; this example does not download data or weights."""

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines.inference_runner import Hyperparams, RunSpec

spec = RunSpec(
    engine_version=SPEC_VERSION,
    model_id="cross-encoder/nli-deberta-v3-base",
    dataset_id="your-account/labelled-pairs",
    split="test",
    template="{premise}",
    column_mapping={"premise": "premise", "hypothesis": "hypothesis"},
    hyperparams=Hyperparams(
        engine="hf_classify",
        pair_template="{hypothesis}",
        max_model_len=512,
        scoring_batch_size=8,
    ),
)

if __name__ == "__main__":
    print(spec.model_dump_json(indent=2))
