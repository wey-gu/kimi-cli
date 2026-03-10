import asyncio
from typing import Any

import httpx
import pytest
import respx
from httpx import Response

from kosong.chat_provider import APIConnectionError, ChatProviderError, openai_common
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.message import Message


def test_create_openai_client_does_not_inject_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai_common, "AsyncOpenAI", FakeAsyncOpenAI)

    openai_common.create_openai_client(
        api_key="test-key",
        base_url="https://example.com/v1",
        client_kwargs={"timeout": 3},
    )

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["timeout"] == 3
    assert "max_retries" not in captured


@pytest.mark.asyncio
async def test_retry_recovery_does_not_close_shared_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OpenAILegacy(
        model="gpt-4.1",
        api_key="test-key",
        http_client=http_client,
    )

    provider.on_retryable_error(APIConnectionError("Connection error."))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert provider.client._client is http_client  # type: ignore[reportPrivateUsage]
    assert http_client.is_closed is False
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("choices", [None, []])
async def test_openai_legacy_invalid_non_stream_choices_raise_chat_provider_error(
    choices: object,
) -> None:
    payload = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4.1",
        "choices": choices,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(return_value=Response(200, json=payload))
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False)
        stream = await provider.generate(
            system_prompt="You are helpful.",
            tools=[],
            history=[Message(role="user", content="Hello!")],
        )

        with pytest.raises(ChatProviderError, match=r"missing choices\[0\]"):
            async for _ in stream:
                pass
