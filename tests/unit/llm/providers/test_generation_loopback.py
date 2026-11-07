"""Real installed SDK -> owned localhost HTTP, with fictional native observation only."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

from strata_forge.llm.generation import request_body
from strata_forge.llm.messages import UserMessage
from strata_forge.llm.providers import openai_compat as provider
from tests.unit.llm.providers.test_generation_transport import client
from tests.unit.llm.providers.test_openai_compat import (
    _loopback,  # pyright: ignore[reportPrivateUsage] — existing owned synthetic HTTP lifecycle.
)
from tests.unit.llm.test_generation_protocol import MODEL, request, response


@pytest.mark.parametrize(
    ("profile", "json_output"), [("plain", False), ("qwen3", True), ("gpt_oss", True)]
)
async def test_actual_sdk_localhost_body_channels_and_schema(
    monkeypatch: pytest.MonkeyPatch, profile: str, json_output: bool
) -> None:
    observed: list[tuple[str, dict[str, str], dict[str, Any]]] = []
    raw = response()
    message = raw["choices"][0]["message"]
    raw["system_fingerprint"] = "fp-fictional"
    if json_output:
        message["content"] = '{"answer":"yes","note":null}'
    if profile != "plain":
        message["reasoning"] = "Synthetic separate reasoning."
    payload = json.dumps(raw).encode()

    class Handler(BaseHTTPRequestHandler):
        timeout = 5

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            observed.append(
                (
                    self.path,
                    {key.lower(): value for key, value in self.headers.items()},
                    json.loads(self.rfile.read(length)),
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — inherited HTTP handler signature.
            pass

    with _loopback(Handler) as url:

        async def observe(_provider: object, value: Any) -> Any:
            return provider._ObservedGenerationEndpoint(  # pyright: ignore[reportPrivateUsage] — fixed synthetic observation only.
                url + "/v1", MODEL, value.model_ref, value.profile, "2.36.0", "0.24.0"
            )  # pyright: ignore[reportPrivateUsage] — fictional observation; no native engine is claimed.

        monkeypatch.setattr(provider, "_observe_generation_endpoint", observe)
        value = request(profile=profile, json_output=json_output)
        text = "Synthetic \r\n{literal}\u03b1"
        result = await client().generate_one([UserMessage(content=text)], request=value)
    assert len(observed) == 1
    path, headers, body = observed[0]
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer EMPTY"
    assert body == request_body(value, user_text=text, served_model=MODEL)
    assert result.status == "completed"
    assert result.final_text == message["content"]
    assert result.reasoning_text == message["reasoning"]
    assert result.system_fingerprint == "fp-fictional"
    assert result.response_id == "fictional-id"
    assert result.json_status == ("valid" if json_output else "not_requested")
