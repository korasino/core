"""Test STT platform of LiteLLM integration."""

from collections.abc import AsyncIterable
import io
from unittest.mock import AsyncMock, MagicMock, patch
import wave

import httpx
from openai import APIConnectionError, AuthenticationError, OpenAIError, PermissionDeniedError
import pytest

from homeassistant.components import stt
from homeassistant.components.litellm.const import DOMAIN
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant

from . import setup_integration
from .conftest import TEST_URL

from tests.common import MockConfigEntry


async def _audio_stream(*chunks: bytes) -> AsyncIterable[bytes]:
    """Yield audio chunks."""
    for chunk in chunks:
        yield chunk


async def _setup_stt(
    hass: HomeAssistant,
    mock_openai_client: AsyncMock,
    endpoints: list[str],
) -> stt.SpeechToTextEntity:
    """Set up a LiteLLM STT entity."""
    entry = MockConfigEntry(
        title="localhost:4000",
        domain=DOMAIN,
        data={CONF_URL: TEST_URL, CONF_API_KEY: "bla"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "home-stt"},
                subentry_id="STT",
                subentry_type="stt",
                title="home-stt",
                unique_id=None,
            )
        ],
    )
    with patch(
        "homeassistant.components.litellm.coordinator.async_get_model_groups",
        new_callable=AsyncMock,
        return_value=[
            {
                "model_group": "home-stt",
                "mode": "audio_transcription",
                "supported_endpoints": endpoints,
                "supported_openai_params": ["language"],
            }
        ],
    ):
        await setup_integration(hass, entry)

    return next(iter(hass.data[stt.DOMAIN].entities))


def _metadata() -> stt.SpeechMetadata:
    """Return the audio format supported by LiteLLM STT."""
    return stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )


async def test_stt_entity_properties(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test STT entity audio properties."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )

    assert "en-US" in entity.supported_languages
    assert "pl-PL" in entity.supported_languages
    assert len(entity.supported_languages) > 1
    assert entity.supported_formats == [stt.AudioFormats.WAV]
    assert entity.supported_codecs == [stt.AudioCodecs.PCM]
    assert entity.supported_bit_rates == [stt.AudioBitRates.BITRATE_16]
    assert entity.supported_sample_rates == list(stt.AudioSampleRates)
    assert entity.supported_channels == [stt.AudioChannels.CHANNEL_MONO]


async def test_batch_stt(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test batch transcription."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="Turn on the light")
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"first", b"second")
    )

    assert result == stt.SpeechResult(
        "Turn on the light", stt.SpeechResultState.SUCCESS
    )
    call = mock_openai_client.audio.transcriptions.create.call_args.kwargs
    assert call["model"] == "home-stt"
    assert call["language"] == "en"
    assert call["file"][0] == "audio.wav"
    with wave.open(io.BytesIO(call["file"][1]), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000


async def test_batch_stt_empty_response_keeps_entity_available(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test an empty batch response keeps the proxy available."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )
    coordinator = entity.entry.runtime_data
    coordinator.mark_connection_error()
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="")
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result == stt.SpeechResult(None, stt.SpeechResultState.ERROR)
    assert entity.available


async def test_batch_stt_connection_error_marks_entity_unavailable(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test batch connection errors mark the entity unavailable."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        side_effect=APIConnectionError(request=None)
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert not entity.available


@pytest.mark.parametrize("error_cls", [AuthenticationError, PermissionDeniedError])
async def test_batch_stt_auth_error_refreshes_coordinator(
    hass: HomeAssistant,
    mock_openai_client: AsyncMock,
    error_cls: type[AuthenticationError] | type[PermissionDeniedError],
) -> None:
    """Test batch authentication errors refresh coordinator state."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        side_effect=error_cls(
            message="invalid api key",
            response=httpx.Response(
                401, request=httpx.Request("POST", "http://localhost")
            ),
            body=None,
        )
    )
    coordinator = entity.entry.runtime_data
    with patch.object(
        coordinator, "async_request_refresh", new_callable=AsyncMock
    ) as refresh:
        result = await entity.async_process_audio_stream(
            _metadata(), _audio_stream(b"audio")
        )

    assert result.result is stt.SpeechResultState.ERROR
    refresh.assert_awaited_once()


async def test_batch_stt_provider_error_keeps_entity_available(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test provider errors do not mark the proxy unavailable."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        side_effect=OpenAIError("bad request")
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert entity.available
