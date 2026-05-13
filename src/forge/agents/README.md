# forge.agents

PydanticAI-based agent builder, built-in tools (`web_search`, `fs_read`, `fetch_url`, `calculator`), `ConversationMemory` plus vector-backed `EpisodicMemory`, and multi-agent patterns (Hand-Off, Critic-Refiner). Reuses the `Tool`, `@tool`, message types, and `run_tool_loop` primitives shipped by `forge.llm` — does not re-implement tool plumbing.

> Implementation pending. See `docs/roadmap.md` for current status.
