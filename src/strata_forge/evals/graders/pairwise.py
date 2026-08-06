"""Pairwise comparison grader — judge candidate against the reference answer.

Different from :class:`LLMJudge`: instead of asking "is this response
good (0..1)?", :class:`PairwiseGrader` puts the candidate side-by-side
with the item's ``expected_output`` and asks the judge to pick a
winner (``A``, ``B``, or ``tie``). Position bias is mitigated by
randomizing which side carries the candidate — the verdict's letter
is decoded back to the underlying answer before scoring.

The grader returns a :class:`GraderResult` whose ``score`` reflects
the verdict:

- candidate wins → 1.0
- tie → 0.5
- reference wins → 0.0

Items without an ``expected_output`` produce a failed result with an
explanation — pairwise grading against an absent reference is
undefined.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from strata_forge.evals.experiment import GraderResult
from strata_forge.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from strata_forge.datasets.schema import DatasetItem
    from strata_forge.llm.client import LLMClient
    from strata_forge.llm.responses import LLMResponse

__all__ = [
    "DEFAULT_PAIRWISE_SYSTEM_PROMPT",
    "DEFAULT_PAIRWISE_USER_TEMPLATE",
    "PairwiseGrader",
    "PairwiseVerdict",
]


class PairwiseVerdict(BaseModel):
    """The judge's pick between two candidates plus reasoning."""

    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B", "tie"]
    reasoning: str


DEFAULT_PAIRWISE_SYSTEM_PROMPT = (
    "You are an impartial evaluator picking the better of two answers. "
    "Read the criteria, the input, and both candidate answers. Return "
    "the letter of the better answer (A, B, or 'tie') plus a brief "
    "reasoning. Judge solely on the criteria — position should not "
    "influence the verdict."
)

DEFAULT_PAIRWISE_USER_TEMPLATE = (
    "Criteria:\n{criteria}\n\nInput:\n{input}\n\nAnswer A:\n{answer_a}\n\nAnswer B:\n{answer_b}"
)


class PairwiseGrader:
    """Compare a candidate response against the item's reference answer.

    Args:
        client: The judge :class:`LLMClient` (typically a stronger
            model than the one whose output is being judged).
        criteria: Free-form description of what "better" means.
        name: Override the grader's ``.name``.
        randomize_positions: When ``True`` (default), the candidate
            and reference are randomly assigned to slots A/B before
            being sent to the judge. The verdict's letter is decoded
            back to ``candidate`` vs ``reference`` after the call,
            so a positional bias in the judge can't systematically
            favour the candidate or the reference.
        system_prompt: Override the judge's system instruction.
        user_template: Override the user-message template;
            ``{criteria}``, ``{input}``, ``{answer_a}``,
            ``{answer_b}`` placeholders are substituted via
            ``str.format``.

    The grader's verdict-to-score mapping:

    - candidate wins → ``1.0`` (passed)
    - tie → ``0.5`` (passed)
    - reference wins → ``0.0`` (failed)
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        criteria: str,
        name: str | None = None,
        randomize_positions: bool = True,
        system_prompt: str = DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        user_template: str = DEFAULT_PAIRWISE_USER_TEMPLATE,
    ) -> None:
        self._client = client
        self._criteria = criteria
        self._name = name or "pairwise"
        self._randomize_positions = randomize_positions
        self._system_prompt = system_prompt
        self._user_template = user_template

    @property
    def name(self) -> str:
        return self._name

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        if item.expected_output is None:
            return GraderResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                explanation="item has no expected_output to compare against",
            )

        candidate_text = response.text
        reference_text = str(item.expected_output)

        # Decide which slot the candidate occupies. By default we randomize
        # to mitigate position bias; tests can pin this by passing
        # randomize_positions=False (deterministic: candidate=A).
        if self._randomize_positions and secrets.randbelow(2) == 1:
            answer_a, answer_b = reference_text, candidate_text
            candidate_slot: Literal["A", "B"] = "B"
        else:
            answer_a, answer_b = candidate_text, reference_text
            candidate_slot = "A"

        user_msg = self._user_template.format(
            criteria=self._criteria,
            input=item.input,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        verdict_resp = await self._client.complete_structured(
            messages=[
                SystemMessage(content=self._system_prompt),
                UserMessage(content=user_msg),
            ],
            schema=PairwiseVerdict,
        )
        verdict: PairwiseVerdict = verdict_resp.parsed

        if verdict.winner == "tie":
            score, passed = 0.5, True
        elif verdict.winner == candidate_slot:
            score, passed = 1.0, True
        else:
            score, passed = 0.0, False

        return GraderResult(
            grader_name=self.name,
            score=score,
            passed=passed,
            explanation=verdict.reasoning,
            metadata={
                "candidate_slot": candidate_slot,
                "winner": verdict.winner,
                "judge_cost_usd": verdict_resp.cost_usd,
            },
        )
