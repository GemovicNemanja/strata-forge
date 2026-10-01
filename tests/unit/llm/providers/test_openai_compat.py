"""Unit tests for `strata_forge.llm.providers.openai_compat`."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import (
    UNAUTHENTICATED_API_KEY,
    OpenAICompatProvider,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine, Generator


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("FORGE_OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("FORGE_OPENAI_COMPAT_API_KEY", raising=False)
    # The OpenAI client reads these itself, so auth_kwargs consults them — a developer who
    # happens to export one must not get a different result from CI.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert OpenAICompatProvider.name == "openai_compat"
        # Wire format identical to native OpenAI — discriminator is api_base.
        assert OpenAICompatProvider.litellm_prefix == "openai/"

    def test_default_config(self) -> None:
        client = OpenAICompatProvider()
        assert isinstance(client.config, OpenAICompatConfig)
        assert client.config.base_url is None
        assert client.config.api_key is None

    def test_explicit_config(self) -> None:
        cfg = OpenAICompatConfig()
        client = OpenAICompatProvider(cfg)
        assert client.config is cfg

    def test_explicit_base_url(self) -> None:
        cfg = OpenAICompatConfig(base_url="http://localhost:8000/v1")
        client = OpenAICompatProvider(cfg)
        assert client.config.base_url == "http://localhost:8000/v1"

    def test_litellm_model(self) -> None:
        # Uses the same prefix as the native OpenAI provider.
        client = OpenAICompatProvider()
        assert client.litellm_model("meta-llama/Llama-3.1-70B") == "openai/meta-llama/Llama-3.1-70B"
        assert client.litellm_model("custom-model") == "openai/custom-model"


class TestAuthKwargs:
    def test_empty_when_no_base_url(self) -> None:
        # Without a base_url, this provider has nothing to add — falls
        # back to native OpenAI behavior, which isn't useful but doesn't
        # break either.
        client = OpenAICompatProvider()
        assert client.auth_kwargs() == {}

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        client = OpenAICompatProvider()
        # The placeholder rides along with a configured base_url and no credential — see
        # test_unauthenticated_dev_deployment for why omitting the key is not an option.
        assert client.auth_kwargs() == {
            "api_base": "http://localhost:8000/v1",
            "api_key": UNAUTHENTICATED_API_KEY,
        }

    def test_base_url_and_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "dummy")
        client = OpenAICompatProvider()
        assert client.auth_kwargs() == {
            "api_base": "http://localhost:8000/v1",
            "api_key": "dummy",
        }

    def test_secret_value_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "extract-me")
        client = OpenAICompatProvider()
        assert client.auth_kwargs()["api_key"] == "extract-me"

    def test_unauthenticated_dev_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unauthenticated server gets a PLACEHOLDER key, not an omitted one.

        vLLM / TGI / SGLang commonly run without auth, but omitting the key does not produce an
        unauthenticated request — the OpenAI client refuses to build a request without one and
        fails before anything reaches the network ("Missing credentials. Please pass an
        `api_key` ..."). Because the failure is identical every time, it takes out an entire
        batch: a run against a local vLLM lost all 2098 rows without one reaching the server.
        """
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        client = OpenAICompatProvider()
        kwargs = client.auth_kwargs()
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        assert kwargs["api_key"] == UNAUTHENTICATED_API_KEY

    @pytest.mark.parametrize("var", ["OPENAI_API_KEY", "OPENAI_ADMIN_KEY"])
    def test_an_ambient_key_is_not_overridden(
        self, monkeypatch: pytest.MonkeyPatch, var: str
    ) -> None:
        # The OpenAI client reads these itself. A caller who exported one means it for an
        # authenticated deployment, and sending a placeholder would replace a real credential
        # with a string that cannot possibly work.
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv(var, "sk-a-real-credential")
        assert "api_key" not in OpenAICompatProvider().auth_kwargs()

    def test_a_configured_key_wins_over_the_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "configured")
        assert OpenAICompatProvider().auth_kwargs()["api_key"] == "configured"

    def test_no_placeholder_without_a_base_url(self) -> None:
        # With no base_url there is no self-hosted server to be unauthenticated against: the call
        # falls through to api.openai.com, where a placeholder would turn a plain "no credentials"
        # into a puzzling rejection of one.
        assert OpenAICompatProvider().auth_kwargs() == {}


class TestAcompletionWiring:
    async def test_routes_through_custom_base_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://vllm:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "dummy")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = OpenAICompatProvider()
        await client.acompletion(
            provider_model_id="meta-llama/Llama-3.1-70B-Instruct",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Uses the openai/ namespace — LiteLLM uses api_base to route
        # away from api.openai.com.
        assert captured["model"] == "openai/meta-llama/Llama-3.1-70B-Instruct"
        assert captured["api_base"] == "http://vllm:8000/v1"
        assert captured["api_key"] == "dummy"

    async def test_call_site_base_url_overrides_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://default:8000/v1")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = OpenAICompatProvider()
        await client.acompletion(
            provider_model_id="custom-model",
            messages=[],
            api_base="http://override:9000/v1",  # call-site override
        )
        assert captured["api_base"] == "http://override:9000/v1"


# The trust note on `base_url` (module docstring, docs/modules/llm.md) rests on how the transport
# treats a redirect. These run the real LiteLLM -> OpenAI client -> httpx stack against two
# loopback servers, so a dependency upgrade that changes either half of the claim fails here: the
# configured `api_key` must never reach another origin, and what DOES travel must match what the
# note says.

_SENTINEL_KEY = "sk-redirect-sentinel"
_PROMPT = "redirect-probe-prompt"
_REPLY = "reply-from-the-redirect-target"
_NON_COMPLETION = "<html>redirect-target-page</html>"
# Bounds every request in both directions: a transport change that hangs (a body promised by
# Content-Length but never sent, a reply never answered) fails the test instead of stalling CI.
_IO_TIMEOUT_S = 10


@dataclass(frozen=True)
class _Seen:
    """Everything one request carried to the server that answered it."""

    method: str
    path: str
    headers: dict[str, str]  # names lowercased: HTTP header names are case-insensitive
    body: str

    @property
    def authorization(self) -> str | None:
        return self.headers.get("authorization")

    @property
    def has_prompt(self) -> bool:
        return _PROMPT in self.body

    def carries(self, secret: str) -> bool:
        return (
            secret in self.path
            or secret in self.body
            or any(secret in f"{name}: {value}" for name, value in self.headers.items())
        )


def _completion(*, stream: bool) -> tuple[bytes, str]:
    if not stream:
        body = {
            "id": "chatcmpl-redirect",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": _REPLY},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return json.dumps(body).encode(), "application/json"
    chunks = [
        {"role": "assistant", "content": _REPLY},
        {},
    ]
    events = [
        {
            "id": "chatcmpl-redirect",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": "stop" if not delta else None}
            ],
        }
        for delta in chunks
    ]
    sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return sse.encode(), "text/event-stream"


def _handler(
    seen: list[_Seen],
    *,
    redirect: int | None = None,
    to_origin: str = "",
    completion: bool = True,
) -> type[BaseHTTPRequestHandler]:
    """Answer with a completion, or, given ``redirect``, send ``/v1/...`` to ``to_origin``.

    An empty ``to_origin`` is a relative ``Location``: a redirect within the same origin. With
    ``completion=False`` the answer is an HTML page instead, which no client can parse as one.
    """

    class _Handler(BaseHTTPRequestHandler):
        timeout = _IO_TIMEOUT_S

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode() if length else ""
            if redirect is not None and self.path.startswith("/v1/"):
                self.send_response(redirect)
                self.send_header("Location", f"{to_origin}/moved{self.path}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            seen.append(
                _Seen(
                    method=self.command,
                    path=self.path,
                    headers={name.lower(): value for name, value in self.headers.items()},
                    body=body,
                )
            )
            if completion:
                stream = bool(body) and json.loads(body).get("stream") is True
                payload, content_type = _completion(stream=stream)
            else:
                payload, content_type = _NON_COMPLETION.encode(), "text/html"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._serve()

        def do_POST(self) -> None:
            self._serve()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — base signature
            del format, args

    return _Handler


@contextmanager
def _loopback(handler: type[BaseHTTPRequestHandler]) -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def _complete(base_url: str, *, stream: bool = False, **kwargs: Any) -> str:
    provider = OpenAICompatProvider(
        OpenAICompatConfig(base_url=f"{base_url}/v1", api_key=SecretStr(_SENTINEL_KEY))
    )
    call: dict[str, Any] = {
        "provider_model_id": "m",
        "messages": [{"role": "user", "content": _PROMPT}],
        "num_retries": 0,
        "max_retries": 0,
        "timeout": _IO_TIMEOUT_S,
        **kwargs,
    }
    if stream:
        parts = [chunk.choices[0].delta.content async for chunk in provider.astream(**call)]
        return "".join(part for part in parts if part)
    response = await provider.acompletion(**call)
    return response.choices[0].message.content


class TestRedirects:
    @pytest.fixture(autouse=True)
    def _no_success_logging(self, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
        # LiteLLM hands every success callback to a worker bound to the running event loop. Each
        # test has its own loop, so the next call would drop the previous call's queued coroutine
        # unawaited. Nothing here is about logging: close the coroutine instead of queueing it.
        # The worker is LiteLLM internals; if a release moves it, the patch is skipped (at worst
        # a stray RuntimeWarning) rather than erroring tests that are about redirects.
        def _discard(async_coroutine: Coroutine[Any, Any, Any]) -> None:
            async_coroutine.close()

        try:
            from litellm.litellm_core_utils import logging_worker
        except ImportError:
            return
        worker = getattr(logging_worker, "GLOBAL_LOGGING_WORKER", None)
        if worker is not None and hasattr(worker, "ensure_initialized_and_enqueue"):
            monkeypatch.setattr(worker, "ensure_initialized_and_enqueue", _discard)

    @pytest.mark.parametrize(
        ("status", "method", "body_resent", "stream"),
        [
            (307, "POST", True, False),
            (307, "POST", True, True),
            (308, "POST", True, False),
            (301, "GET", False, False),
            (302, "GET", False, False),
            (303, "GET", False, False),
        ],
    )
    async def test_a_cross_origin_redirect_drops_the_key_but_not_the_reply(
        self, status: int, method: str, body_resent: bool, stream: bool
    ) -> None:
        # A different port is a different origin, exactly as a different host is.
        seen: list[_Seen] = []
        with (
            _loopback(_handler(seen)) as target,
            _loopback(_handler([], redirect=status, to_origin=target)) as base_url,
        ):
            reply = await _complete(base_url, stream=stream)

        assert len(seen) == 1
        # Not only the Authorization header: the key must not ride in any header, the path or
        # the body either.
        assert not seen[0].carries(_SENTINEL_KEY), "the key reached another origin"
        assert seen[0].authorization is None
        assert seen[0].method == method
        assert seen[0].has_prompt is body_resent
        assert reply == _REPLY

    async def test_a_non_completion_reply_comes_back_quoted_in_the_error(self) -> None:
        seen: list[_Seen] = []
        with (
            _loopback(_handler(seen, completion=False)) as target,
            _loopback(_handler([], redirect=302, to_origin=target)) as base_url,
            # Which exception class LiteLLM maps this to is not the claim; the quoted reply is.
            pytest.raises(Exception, match="redirect-target-page"),
        ):
            await _complete(base_url)

        assert len(seen) == 1
        assert not seen[0].carries(_SENTINEL_KEY)

    async def test_a_credential_in_extra_headers_follows_a_cross_origin_redirect(self) -> None:
        # Only `Authorization` is dropped. A credential a caller puts in another header goes
        # wherever the redirect points, which is why llm.md makes extra_headers caller-trusted.
        seen: list[_Seen] = []
        with (
            _loopback(_handler(seen)) as target,
            _loopback(_handler([], redirect=307, to_origin=target)) as base_url,
        ):
            reply = await _complete(base_url, extra_headers={"X-Api-Key": "sk-extra-sentinel"})

        assert len(seen) == 1
        assert seen[0].authorization is None
        assert seen[0].headers.get("x-api-key") == "sk-extra-sentinel"
        assert reply == _REPLY

    async def test_a_same_origin_redirect_keeps_the_key(self) -> None:
        seen: list[_Seen] = []
        with _loopback(_handler(seen, redirect=307)) as base_url:
            reply = await _complete(base_url)

        assert len(seen) == 1
        assert seen[0].method == "POST"
        assert seen[0].path == "/moved/v1/chat/completions"
        assert seen[0].authorization == f"Bearer {_SENTINEL_KEY}"
        assert seen[0].has_prompt
        assert reply == _REPLY
