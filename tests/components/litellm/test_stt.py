"""Test STT platform of LiteLLM integration."""

from collections.abc import AsyncIterable
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import wave

from openai import APIConnectionError
from websockets.exceptions import WebSocketException

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
    supported_openai_params: list[str] | None = None,
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
    with (
        patch(
            "homeassistant.components.litellm.stt.async_get_model_groups",
            new_callable=AsyncMock,
            return_value=[
                {
                    "model_group": "home-stt",
                    "mode": "audio_transcription",
                    "supported_endpoints": endpoints,
                    "supported_openai_params": supported_openai_params or [],
                }
            ],
        ),
        patch(
            "homeassistant.components.litellm.stt.async_get_exposed_entities",
            return_value={
                "light.yellow_lamp": {
                    "names": "Yellow lamp, Desk lamp",
                    "domain": "light",
                    "areas": "Living room, Lounge",
                },
                "climate.bedroom": {
                    "names": "Bedroom climate",
                    "domain": "climate",
                    "areas": "Bedroom",
                },
            },
        ),
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

    assert entity.supported_languages == [hass.config.language]
    assert entity.supported_formats == [stt.AudioFormats.WAV]
    assert entity.supported_codecs == [stt.AudioCodecs.PCM]
    assert entity.supported_bit_rates == [stt.AudioBitRates.BITRATE_16]
    assert entity.supported_sample_rates == [stt.AudioSampleRates.SAMPLERATE_16000]
    assert entity.supported_channels == [stt.AudioChannels.CHANNEL_MONO]


async def test_batch_stt(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test batch transcription."""
    entity = await _setup_stt(
        hass,
        mock_openai_client,
        ["/v1/audio/transcriptions"],
        ["keywords"],
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
    assert call["keywords"] == [
        "Yellow lamp",
        "Desk lamp",
        "Living room",
        "Lounge",
        "Bedroom climate",
        "Bedroom",
    ]
    assert call["file"][0] == "audio.wav"
    with wave.open(io.BytesIO(call["file"][1]), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000


async def test_batch_stt_omits_unsupported_keywords(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test batch transcription omits unsupported optional parameters."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/audio/transcriptions"]
    )
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text="Turn on the light")
    )

    await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    call = mock_openai_client.audio.transcriptions.create.call_args.kwargs
    assert "keywords" not in call


async def test_batch_stt_error(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test batch transcription errors."""
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


class _RealtimeConnection:
    """Minimal realtime connection used by STT tests."""

    def __init__(
        self, events: list[SimpleNamespace], error: Exception | None = None
    ) -> None:
        self.events = events
        self.error = error
        self.session_updates: list[dict] = []
        self.audio: list[str] = []
        self.session = self
        self.commits = 0
        self.input_audio_buffer = self

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
        """Iterate server events."""

        async def events():
            for event in self.events:
                yield event
            if self.error is not None:
                raise self.error

        return events()


class _RealtimeContext:
    """Context manager for a fake realtime connection."""

    def __init__(self, connection: _RealtimeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _RealtimeConnection:
        return self.connection

    async def __aexit__(self, *args) -> None:
        return None


async def test_realtime_stt(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test realtime transcription streams chunks and ends once."""
    entity = await _setup_stt(
        hass,
        mock_openai_client,
        [
            "/v1/audio/transcriptions",
            "/v1/realtime/transcription_sessions",
        ],
        ["keywords"],
    )
    connection = _RealtimeConnection(
        [
            SimpleNamespace(type="session.created"),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Turn on the light",
            ),
        ]
    )
    mock_openai_client.realtime.connect = MagicMock(
        return_value=_RealtimeContext(connection)
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"first", b"second")
    )

    assert result == stt.SpeechResult(
        "Turn on the light", stt.SpeechResultState.SUCCESS
    )
    mock_openai_client.realtime.connect.assert_called_once_with(
        model="home-stt",
        extra_query={"intent": "transcription"},
        max_retries=0,
    )
    audio_input = connection.session_updates[0]["audio"]["input"]
    assert audio_input["format"] == {
        "type": "audio/pcm",
        "rate": 16000,
        "channels": 1,
    }
    assert audio_input["turn_detection"] is None
    assert audio_input["transcription"]["keywords"] == [
        "Yellow lamp",
        "Desk lamp",
        "Living room",
        "Lounge",
        "Bedroom climate",
        "Bedroom",
    ]
    assert len(connection.audio) == 2
    assert connection.commits == 1


async def test_realtime_stt_error_event(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test realtime transcription error event."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/realtime/transcription_sessions"]
    )
    connection = _RealtimeConnection([SimpleNamespace(type="error")])
    mock_openai_client.realtime.connect = MagicMock(
        return_value=_RealtimeContext(connection)
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert mock_openai_client.realtime.connect.call_count == 1


async def test_realtime_stt_closed_socket(
    hass: HomeAssistant, mock_openai_client: AsyncMock
) -> None:
    """Test realtime transcription returns an error when the socket closes."""
    entity = await _setup_stt(
        hass, mock_openai_client, ["/v1/realtime/transcription_sessions"]
    )
    connection = _RealtimeConnection([], WebSocketException("closed"))
    mock_openai_client.realtime.connect = MagicMock(
        return_value=_RealtimeContext(connection)
    )

    result = await entity.async_process_audio_stream(
        _metadata(), _audio_stream(b"audio")
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert mock_openai_client.realtime.connect.call_count == 1
