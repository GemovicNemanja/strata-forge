"""LLM-judge grader — ask a model to grade a candidate response.

The judge runs :meth:`LLMClient.complete_structured` against
:class:`JudgeVerdict` (or a user-supplied schema) and converts the
verdict into a :class:`GraderResult`. ``pass_threshold`` decides the
score → passed projection so callers can tune strictness without
re-prompting.

The default system + user prompts present the criteria, the item's
input, the item's reference answer (when present), and the candidate
response, then ask the judge for a numeric score plus reasoning. Pass
``system_prompt`` / ``user_template`` to customize the wording — the
``{criteria}``, ``{input}``, ``{expected}``, ``{response}``
placeholders are substituted via ``str.format`` before the request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from forge.evals.experiment import GraderResult
from forge.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from forge.datasets.schema import DatasetItem
    from forge.llm.client import LLMClient
    from forge.llm.responses import LLMResponse

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_TEMPLATE",
    "JudgeVerdict",
    "LLMJudge",
]


class JudgeVerdict(BaseModel):
    """Default schema for :class:`LLMJudge`.

    ``score`` is constrained to ``[0, 1]`` so the judge's verdict
    composes with :func:`forge.evals.metrics.mean_score`. ``reasoning``
    is the judge's free-form explanation, surfaced in the
    :class:`GraderResult` explanation field.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


DEFAULT_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Read the criteria, the input, "
    "the reference answer (when present), and the candidate response. "
    "Return a score in [0, 1] reflecting how well the candidate "
    "satisfies the criteria, plus a one- or two-sentence reasoning. "
    "Be strict but fair — partial credit is allowed."
)

DEFAULT_USER_TEMPLATE = (
    "Criteria:\n{criteria}\n\n"
    "Input:\n{input}\n\n"
    "Reference answer:\n{expected}\n\n"
    "Candidate response:\n{response}"
)


class LLMJudge:
    """A grader that asks an LLM to score the response.

    Args:
        client: The :class:`LLMClient` the judge uses. Typically a
            different (often stronger) model than the one whose
            output is being graded.
        criteria: Free-form description of what "good" means for
            this evaluation. Inserted into the prompt verbatim.
        pass_threshold: Minimum score for a passing verdict. Default
            ``0.7``.
        name: Override the grader's ``.name`` (used in reports).
        system_prompt: Override the system instruction. The default
            asks for a score and short reasoning.
        user_template: Override the user-message template; ``str.format``
            substitutes ``{criteria}``, ``{input}``, ``{expected}``,
            and ``{response}`` placeholders.
        verdict_schema: The Pydantic schema the judge is asked to
            produce. Must have ``score: float`` and
            ``reasoning: str`` fields (extra fields are allowed).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        criteria: str,
        pass_threshold: float = 0.7,
        name: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
        verdict_schema: type[BaseModel] = JudgeVerdict,
    ) -> None:
        if not (0.0 <= pass_threshold <= 1.0):
            msg = f"pass_threshold must be in [0, 1]; got {pass_threshold}"
            raise ValueError(msg)
        self._client = client
        self._criteria = criteria
        self._pass_threshold = pass_threshold
        self._name = name or "llm_judge"
        self._system_prompt = system_prompt
        self._user_template = user_template
        self._verdict_schema = verdict_schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def pass_threshold(self) -> float:
        return self._pass_threshold

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        user_msg = self._user_template.format(
            criteria=self._criteria,
            input=item.input,
            expected="(none)" if item.expected_output is None else item.expected_output,
            response=response.text,
        )
        verdict_resp = await self._client.complete_structured(
            messages=[
                SystemMessage(content=self._system_prompt),
                UserMessage(content=user_msg),
            ],
            schema=self._verdict_schema,
        )
        verdict = verdict_resp.parsed
        score = float(verdict.score)  # type: ignore[attr-defined]
        reasoning = str(verdict.reasoning)  # type: ignore[attr-defined]
        return GraderResult(
            grader_name=self.name,
            score=score,
            passed=score >= self._pass_threshold,
            explanation=reasoning,
            metadata={
                "judge_cost_usd": verdict_resp.cost_usd,
                "judge_latency_ms": verdict_resp.latency_ms,
            },
        )
