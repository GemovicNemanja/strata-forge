"""Multi-agent patterns: hand-off and critic-refiner.

Both patterns are **pure compositions** over :class:`Agent`. No new
runtime, no new message types, no new tool plumbing — each function
just calls one or more agent ``.run()`` / ``.run_structured()``
methods in a specific shape.

- :func:`handoff` — a router agent decides which specialist should
  handle a query, then the chosen specialist is invoked with the
  user's input and its :class:`AgentResult` is returned.
- :func:`critic_refiner_run` — a "drafter" agent produces an
  answer, a "critic" agent reviews it; on rejection the drafter
  refines using the critic's feedback. Loops until approval or
  ``max_rounds`` is hit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from forge.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from forge.agents.agent import Agent, AgentResult
    from forge.llm.messages import AnyMessage

__all__ = [
    "CritiqueVerdict",
    "RouterChoice",
    "critic_refiner_run",
    "handoff",
]


class RouterChoice(BaseModel):
    """The router agent's structured verdict on which specialist to invoke."""

    model_config = ConfigDict(extra="forbid")

    specialist: str = Field(
        description="Name of the specialist agent that should handle this query."
    )
    reasoning: str = Field(
        default="",
        description="Why this specialist is the best fit for the query.",
    )


class CritiqueVerdict(BaseModel):
    """The critic agent's structured verdict on a drafted response."""

    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(
        description="True when the draft satisfies the criteria and no further refinement is needed."
    )
    feedback: str = Field(
        default="",
        description=(
            "Specific, actionable feedback for the drafter. Required "
            "when approved=False; ignored when approved=True."
        ),
    )


async def handoff(
    *,
    router: Agent,
    specialists: Mapping[str, Agent],
    user_input: str,
) -> AgentResult:
    """Route ``user_input`` to one of ``specialists`` via ``router``.

    The ``router`` is invoked via :meth:`Agent.run_structured` with
    :class:`RouterChoice` as the output schema. The
    ``specialists`` mapping is rendered into the router's user
    message so the model knows which choices exist. The chosen
    specialist is then invoked with the original ``user_input`` and
    its :class:`AgentResult` is returned.

    Args:
        router: The agent that decides which specialist to invoke.
            Typically a small / fast model.
        specialists: Map of specialist name → :class:`Agent`. The
            ``router`` is shown these names and must pick one.
        user_input: The original user query.

    Returns:
        The chosen specialist's :class:`AgentResult` — the router's
        intermediate decision is intentionally not exposed in the
        return; check Langfuse traces for full visibility.

    Raises:
        ValueError: When ``specialists`` is empty, or when the
            router picks a name that isn't in ``specialists``.
    """
    if not specialists:
        err = "handoff: specialists mapping must be non-empty"
        raise ValueError(err)

    specialist_names = sorted(specialists)
    catalog = "\n".join(f"- {name}" for name in specialist_names)
    routing_prompt = (
        f"Available specialists:\n{catalog}\n\n"
        f"User query:\n{user_input}\n\n"
        f"Choose the single best specialist for this query."
    )

    router_result = await router.run_structured(
        routing_prompt,
        output_schema=RouterChoice,
    )
    choice = router_result.parsed
    if not isinstance(choice, RouterChoice):  # defensive — parsed comes from Pydantic
        err = f"handoff: router didn't return a RouterChoice; got {type(choice).__name__}"
        raise ValueError(err)

    chosen = choice.specialist
    if chosen not in specialists:
        err = (
            f"handoff: router picked {chosen!r}, which isn't in the "
            f"specialists catalog {specialist_names!r}"
        )
        raise ValueError(err)

    return await specialists[chosen].run(user_input)


async def critic_refiner_run(
    *,
    drafter: Agent,
    critic: Agent,
    user_input: str,
    max_rounds: int = 3,
) -> AgentResult:
    """Iteratively draft and critique until approved or ``max_rounds`` is hit.

    Each round:

    1. The ``drafter`` produces an answer (informed by previous
       critique on rounds > 1).
    2. The ``critic`` reviews it via :meth:`Agent.run_structured`
       with :class:`CritiqueVerdict` as the output schema.
    3. If ``approved`` is True, the drafter's last result is
       returned.
    4. Otherwise, the critic's ``feedback`` is rolled into the
       drafter's next prompt and the loop continues.

    When ``max_rounds`` is exhausted without approval, the drafter's
    most recent result is returned — partial progress is preferable
    to raising.

    Args:
        drafter: The agent that produces drafts.
        critic: The agent that reviews drafts. Typically a stronger
            model than the drafter.
        user_input: The original user query.
        max_rounds: Maximum draft/critique cycles before giving up.
            Default 3; must be >= 1.

    Returns:
        The drafter's :class:`AgentResult` from the round when the
        critic approved, or from the final round when ``max_rounds``
        was hit.

    Raises:
        ValueError: When ``max_rounds < 1``.
    """
    if max_rounds < 1:
        err = f"critic_refiner_run: max_rounds must be >= 1; got {max_rounds}"
        raise ValueError(err)

    last_draft: AgentResult | None = None
    feedback_so_far: list[str] = []

    for _ in range(max_rounds):
        if not feedback_so_far:
            last_draft = await drafter.run(user_input)
        else:
            # Roll the previous draft + critique feedback into the next prompt.
            previous_text = last_draft.text if last_draft else ""
            critique_lines = "\n\n".join(
                f"Round {i + 1} critique:\n{fb}" for i, fb in enumerate(feedback_so_far)
            )
            messages: list[AnyMessage] = [
                SystemMessage(
                    content=(
                        "Revise the previous draft using the critic's "
                        "feedback. Keep what works; fix what's flagged."
                    )
                ),
                UserMessage(
                    content=(
                        f"Original request:\n{user_input}\n\n"
                        f"Previous draft:\n{previous_text}\n\n"
                        f"{critique_lines}"
                    )
                ),
            ]
            last_draft = await drafter.run(messages)

        critique_result = await critic.run_structured(
            (f"Original request:\n{user_input}\n\nDraft to evaluate:\n{last_draft.text}"),
            output_schema=CritiqueVerdict,
        )
        verdict = critique_result.parsed
        if not isinstance(verdict, CritiqueVerdict):  # defensive
            err = (
                f"critic_refiner_run: critic didn't return a CritiqueVerdict; "
                f"got {type(verdict).__name__}"
            )
            raise ValueError(err)

        if verdict.approved:
            return last_draft
        feedback_so_far.append(verdict.feedback or "(no specific feedback provided)")

    # max_rounds exhausted; return last draft.
    assert last_draft is not None  # noqa: S101 — invariant: loop ran at least once
    return last_draft
