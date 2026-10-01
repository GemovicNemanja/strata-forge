"""Unit tests for `strata_forge.llm.providers.openai_compat`."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, NamedTuple

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
# key must never reach another origin, and what DOES travel must match what the note says.

_SENTINEL_KEY = "sk-redirect-sentinel"
_PROMPT = "redirect-probe-prompt"
_REPLY = "reply-from-the-redirect-target"


class _Seen(NamedTuple):
    method: str
    path: str
    authorization: str | None
    has_prompt: bool


def _completion_body() -> bytes:
    return json.dumps(
        {
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
    ).encode()


def _handler(
    seen: list[_Seen], *, redirect: int | None = None, to_origin: str = ""
) -> type[BaseHTTPRequestHandler]:
    """Answer with a completion, or, given ``redirect``, send ``/v1/...`` to ``to_origin``.

    An empty ``to_origin`` is a relative ``Location``: a redirect within the same origin.
    """

    class _Handler(BaseHTTPRequestHandler):
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
                    authorization=self.headers.get("Authorization"),
                    has_prompt=_PROMPT in body,
                )
            )
            payload = _completion_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
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


async def _complete(base_url: str) -> str:
    provider = OpenAICompatProvider(
        OpenAICompatConfig(base_url=f"{base_url}/v1", api_key=SecretStr(_SENTINEL_KEY))
    )
    response = await provider.acompletion(
        provider_model_id="m",
        messages=[{"role": "user", "content": _PROMPT}],
        num_retries=0,
        max_retries=0,
    )
    return response.choices[0].message.content


class TestRedirects:
    @pytest.fixture(autouse=True)
    def _no_success_logging(self, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
        # LiteLLM hands every success callback to a worker bound to the running event loop. Each
        # test has its own loop, so the next call would drop the previous call's queued coroutine
        # unawaited. Nothing here is about logging: close the coroutine instead of queueing it.
        def _discard(async_coroutine: Coroutine[Any, Any, Any]) -> None:
            async_coroutine.close()

        monkeypatch.setattr(
            "litellm.litellm_core_utils.logging_worker.GLOBAL_LOGGING_WORKER"
            ".ensure_initialized_and_enqueue",
            _discard,
        )

    @pytest.mark.parametrize(
        ("status", "method", "body_resent"),
        [
            (307, "POST", True),
            (308, "POST", True),
            (301, "GET", False),
            (302, "GET", False),
            (303, "GET", False),
        ],
    )
    async def test_a_cross_origin_redirect_drops_the_key_but_not_the_reply(
        self, status: int, method: str, body_resent: bool
    ) -> None:
        # A different port is a different origin, exactly as a different host is.
        seen: list[_Seen] = []
        with (
            _loopback(_handler(seen)) as target,
            _loopback(_handler([], redirect=status, to_origin=target)) as base_url,
        ):
            reply = await _complete(base_url)

        assert [s.authorization for s in seen] == [None], "the key reached another origin"
        assert seen[0].method == method
        assert seen[0].has_prompt is body_resent
        assert reply == _REPLY

    async def test_a_same_origin_redirect_keeps_the_key(self) -> None:
        seen: list[_Seen] = []
        with _loopback(_handler(seen, redirect=307)) as base_url:
            reply = await _complete(base_url)

        assert seen == [
            _Seen("POST", "/moved/v1/chat/completions", f"Bearer {_SENTINEL_KEY}", True)
        ]
        assert reply == _REPLY
