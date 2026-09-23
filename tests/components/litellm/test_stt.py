"""Test STT platform of LiteLLM integration."""

from collections.abc import AsyncIterable
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import wave

import httpx
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
)
import pytest

from homeassistant.components import stt
from homeassistant.components.litellm.const import (
    CONF_AUDIO_CHANNELS,
    CONF_AUDIO_FORMAT_OVERRIDE,
    CONF_AUDIO_SAMPLE_RATE,
    CONF_VOCABULARY,
    DOMAIN,
)
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_PROMPT, CONF_URL
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
    supported_openai_params: list[str] | None = None,
    subentry_data: dict[str, bool | str] | None = None,
) -> stt.SpeechToTextEntity:
    """Set up a LiteLLM STT entity."""
    entry = MockConfigEntry(
        title="localhost:4000",
        domain=DOMAIN,
        data={CONF_URL: TEST_URL, CONF_API_KEY: "bla"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "home-stt", **(subentry_data or {})},
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
                "supported_openai_params": supported_openai_params or ["language"],
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
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/audio/transcriptions"])

    assert "en-US" in entity.supported_languages
    assert "pl-PL" in entity.supported_languages
    assert len(entity.supported_languages) > 1
    assert entity.supported_formats == list(stt.AudioFormats)
    assert entity.supported_codecs == list(stt.AudioCodecs)
    assert entity.supported_bit_rates == list(stt.AudioBitRates)
    assert entity.supported_sample_rates == [
        stt.AudioSampleRates.SAMPLERATE_8000,
        stt.AudioSampleRates.SAMPLERATE_11000,
        stt.AudioSampleRates.SAMPLERATE_16000,
        stt.AudioSampleRates.SAMPLERATE_18900,
        stt.AudioSampleRates.SAMPLERATE_22000,
        stt.AudioSampleRates.SAMPLERATE_32000,
        stt.AudioSampleRates.SAMPLERATE_37800,
        stt.AudioSampleRates.SAMPLERATE_44100,
        stt.AudioSampleRates.SAMPLERATE_48000,
    ]
    assert entity.supported_channels == list(stt.AudioChannels)


async def test_batch_stt(hass: HomeAssistant, mock_openai_client: AsyncMock) -> None:
    """Test batch transcription."""
    entity = await _setup_stt(
        hass,
        mock_openai_client,
        ["/v1/audio/transcriptions"],
        ["language", "prompt", "keywords"],
        {
            CONF_PROMPT: "Transcribe {{ states('sensor.context') }} commands.",
            CONF_VOCABULARY: (
                "Kitchen light, Hallway light, {{ states('sensor.term') }}"
            ),
        },
    )
    hass.states.async_set("sensor.context", "office")
    hass.states.async_set("sensor.term", "thermostat")
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
    assert call["prompt"] == "Transcribe office commands."
    assert call["keywords"] == ["Kitchen light", "Hallway light", "thermostat"]
    assert call["file"][0] == "audio.wav"
    with wave.open(io.BytesIO(call["file"][1]), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000


async def test_batch_stt_preserves_ogg_audio(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test batch transcription preserves non-WAV audio."""
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/audio/transcriptions"])
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="Turn on the light")
    )
    metadata = stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.OGG,
        codec=stt.AudioCodecs.OPUS,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )
    audio = b"ogg-audio"

    result = await entity.async_process_audio_stream(metadata, _audio_stream(audio))

    assert result == stt.SpeechResult(
        "Turn on the light", stt.SpeechResultState.SUCCESS
    )
    call = mock_openai_client.audio.transcriptions.create.call_args.kwargs
    assert call["file"] == ("audio.ogg", audio)


async def test_batch_stt_empty_response_keeps_entity_available(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test an empty batch response keeps the proxy available."""
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/audio/transcriptions"])
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
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/audio/transcriptions"])
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
    error_cls: type[AuthenticationError | PermissionDeniedError],
) -> None:
    """Test batch authentication errors refresh coordinator state."""
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/audio/transcriptions"])
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
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/audio/transcriptions"])
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        side_effect=OpenAIError("bad request")
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert entity.available


class _RealtimeConnection:
    """Minimal realtime connection used by STT tests."""

    def __init__(self) -> None:
        self.session_updates: list[dict] = []
        self.audio: list[str] = []
        self.session = self
        self.input_audio_buffer = self
        self.commits = 0

    async def update(self, *, session: dict) -> None:
        """Record a session update."""
        self.session_updates.append(session)

    async def append(self, *, audio: str) -> None:
        """Record an audio chunk."""
        self.audio.append(audio)

    async def commit(self) -> None:
        """Record the end of the HA audio stream."""
        self.commits += 1

    def __aiter__(self):
        """Iterate a completed transcription event."""

        async def events():
            yield SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Turn on the light",
            )

        return events()


class _RealtimeContext:
    """Context manager for a fake realtime connection."""

    def __init__(self, connection: _RealtimeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _RealtimeConnection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


async def test_realtime_stt_uses_input_audio_metadata(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test realtime STT declares the incoming audio format by default."""
    entity = await _setup_stt(hass, mock_openai_client, ["/v1/realtime"])
    connection = _RealtimeConnection()
    mock_openai_client.realtime.connect = MagicMock(
        return_value=_RealtimeContext(connection)
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result == stt.SpeechResult(
        "Turn on the light", stt.SpeechResultState.SUCCESS
    )
    assert connection.session_updates[0]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 16000,
        "channels": 1,
    }


async def test_realtime_stt_uses_configured_audio_format(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test realtime STT uses the configured audio format override."""
    entity = await _setup_stt(
        hass,
        mock_openai_client,
        ["/v1/realtime"],
        subentry_data={
            CONF_AUDIO_FORMAT_OVERRIDE: True,
            CONF_AUDIO_SAMPLE_RATE: "24000",
            CONF_AUDIO_CHANNELS: "2",
        },
    )
    connection = _RealtimeConnection()
    mock_openai_client.realtime.connect = MagicMock(
        return_value=_RealtimeContext(connection)
    )

    await entity.async_process_audio_stream(_metadata(), _audio_stream(b"audio"))

    assert connection.session_updates[0]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
        "channels": 2,
    }
