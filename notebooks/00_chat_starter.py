"""Chat starter — explore a single ``LLMClient`` call interactively.

Set the relevant provider env var (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, etc.) before launching with::

    uv run marimo edit notebooks/00_chat_starter.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        "# Chat starter\n\n"
        "Pick a model, write a prompt, inspect the full response: "
        "text, route, tokens, cost, latency."
    )
    return


@app.cell
def _(mo):
    model_input = mo.ui.dropdown(
        options=[
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "gpt-5.5",
            "gpt-5.5-instant",
            "gemini-3.1-pro",
            "gemini-3.1-flash",
        ],
        value="claude-haiku-4-5",
        label="model",
    )
    prompt_input = mo.ui.text_area(value="Explain Forge in one sentence.", label="prompt", rows=4)
    temperature_input = mo.ui.slider(0.0, 1.5, step=0.1, value=0.7, label="temp")
    return model_input, prompt_input, temperature_input


@app.cell
def _(mo, model_input, prompt_input, temperature_input):
    mo.hstack([model_input, temperature_input, prompt_input])
    return


@app.cell
async def _(model_input, prompt_input, temperature_input):
    from strata_forge.llm import LLMClient, Message

    client = LLMClient(model=model_input.value)
    response = await client.complete(
        messages=[Message.user(prompt_input.value)],
        temperature=temperature_input.value,
    )
    return (response,)


@app.cell
def _(mo, response):
    mo.md(
        f"### Response\n\n{response.text}\n\n"
        f"**route**: `{response.route.model}` via "
        f"`{response.route.provider}`  \n"
        f"**tokens**: in={response.usage.input_tokens} "
        f"out={response.usage.output_tokens}  \n"
        f"**cost**: ${response.cost_usd:.5f}  "
        f"**latency**: {response.latency_ms:.0f} ms"
    )
    return


if __name__ == "__main__":
    app.run()
