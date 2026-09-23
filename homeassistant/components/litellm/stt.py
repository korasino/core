"""Speech-to-text support for LiteLLM."""

import base64
from collections.abc import AsyncIterable
import io
from typing import Any, cast, override
import wave

from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
    WebSocketConnectionClosedError,
)
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_AUDIO_CHANNELS,
    CONF_AUDIO_FORMAT_OVERRIDE,
    CONF_AUDIO_SAMPLE_RATE,
    LOGGER,
    STT_BATCH_ENDPOINT,
    STT_REALTIME_ENDPOINT,
)
from .coordinator import LiteLLMConfigEntry
from .entity import LiteLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LiteLLMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up LiteLLM STT entities."""
    stt_subentries = [
        subentry
        for subentry in config_entry.subentries.values()
        if subentry.subentry_type == "stt"
    ]
    if not stt_subentries:
        return

    model_groups = await config_entry.runtime_data.async_get_model_groups()
    capabilities_by_model = {
        model["model_group"]: (
            model.get("supported_endpoints") or [],
            model.get("supported_openai_params") or [],
        )
        for model in model_groups
        if model.get("mode") == "audio_transcription"
    }

    for subentry in stt_subentries:
        async_add_entities(
            [
                LiteLLMSTTEntity(
                    config_entry,
                    subentry,
                    *capabilities_by_model.get(subentry.data[CONF_MODEL], ([], [])),
                )
            ],
            config_subentry_id=subentry.subentry_id,
        )


class LiteLLMSTTEntity(stt.SpeechToTextEntity, LiteLLMEntity):
    """LiteLLM speech-to-text entity."""

    def __init__(
        self,
        entry: LiteLLMConfigEntry,
        subentry: ConfigSubentry,
        supported_endpoints: list[str],
        supported_openai_params: list[str],
    ) -> None:
        """Initialize the STT entity."""
        super().__init__(entry, subentry)
        self._supported_endpoints = supported_endpoints
        self._supports_language = "language" in supported_openai_params
        self._audio_format_override = subentry.data.get(
            CONF_AUDIO_FORMAT_OVERRIDE, False
        )

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        # LiteLLM does not expose model-specific language capabilities. This
        # static OpenAI-compatible list is only an API-level advertisement; the
        # selected backend may reject languages it does not support.
        return [
            "af-ZA",
            "ar-SA",
            "hy-AM",
            "az-AZ",
            "be-BY",
            "bs-BA",
            "bg-BG",
            "ca-ES",
            "zh-CN",
            "hr-HR",
            "cs-CZ",
            "da-DK",
            "nl-NL",
            "en-US",
            "et-EE",
            "fi-FI",
            "fr-FR",
            "gl-ES",
            "de-DE",
            "el-GR",
            "he-IL",
            "hi-IN",
            "hu-HU",
            "is-IS",
            "id-ID",
            "it-IT",
            "ja-JP",
            "kn-IN",
            "kk-KZ",
            "ko-KR",
            "lv-LV",
            "lt-LT",
            "mk-MK",
            "ms-MY",
            "mr-IN",
            "mi-NZ",
            "ne-NP",
            "no-NO",
            "fa-IR",
            "pl-PL",
            "pt-PT",
            "ro-RO",
            "ru-RU",
            "sr-RS",
            "sk-SK",
            "sl-SI",
            "es-ES",
            "sw-KE",
            "sv-SE",
            "fil-PH",
            "ta-IN",
            "th-TH",
            "tr-TR",
            "uk-UA",
            "ur-PK",
            "vi-VN",
            "cy-GB",
        ]

    @property
    @override
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return supported formats."""
        # LiteLLM exposes the OpenAI transcription contract but does not
        # report model-specific audio capabilities, so mirror its API-level
        # input declarations here.
        return [stt.AudioFormats.WAV, stt.AudioFormats.OGG]

    @property
    @override
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported codecs."""
        return [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]

    @property
    @override
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported bit rates."""
        return [
            stt.AudioBitRates.BITRATE_8,
            stt.AudioBitRates.BITRATE_16,
            stt.AudioBitRates.BITRATE_24,
            stt.AudioBitRates.BITRATE_32,
        ]

    @property
    @override
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported sample rates."""
        return [
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

    @property
    @override
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO, stt.AudioChannels.CHANNEL_STEREO]

    @override
    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream."""
        if STT_REALTIME_ENDPOINT in self._supported_endpoints:
            LOGGER.debug("LiteLLM STT using realtime endpoint for model %s", self.model)
            return await self._async_process_realtime(metadata, stream)
        if STT_BATCH_ENDPOINT in self._supported_endpoints:
            LOGGER.debug("LiteLLM STT using batch endpoint for model %s", self.model)
            return await self._async_process_batch(metadata, stream)
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    async def _async_process_realtime(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Stream audio to LiteLLM's realtime transcription endpoint."""
        coordinator = self.entry.runtime_data
        try:
            async with coordinator.client.realtime.connect(
                model=self.model,
                extra_query={"intent": "transcription"},
                max_retries=0,
            ) as connection:
                coordinator.async_set_updated_data(None)
                await connection.session.update(
                    session=cast(
                        Any,
                        {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": self._realtime_audio_format(metadata),
                                    "transcription": {
                                        "model": self.model,
                                        **self._transcription_options(metadata),
                                    },
                                    "turn_detection": None,
                                }
                            },
                        },
                    ),
                )
                async for chunk in stream:
                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(chunk).decode()
                    )
                await connection.input_audio_buffer.commit()

                async for event in connection:
                    if (
                        event.type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        if event.transcript:
                            return stt.SpeechResult(
                                event.transcript, stt.SpeechResultState.SUCCESS
                            )
                        break
                    if (
                        event.type
                        == "conversation.item.input_audio_transcription.failed"
                    ):
                        LOGGER.error(
                            "Realtime STT transcription failed: %s",
                            event.error.message,
                        )
                        break
                    if event.type == "error":
                        LOGGER.error("Realtime STT error: %s", event)
                        break
        except (AuthenticationError, PermissionDeniedError) as err:
            await coordinator.async_request_refresh()
            LOGGER.error("Authentication error during realtime STT: %s", err)
        except APIConnectionError as err:
            coordinator.mark_connection_error()
            LOGGER.error("Connection error during realtime STT: %s", err)
        except InvalidStatus as err:
            if err.response.status_code in (401, 403):
                await coordinator.async_request_refresh()
                LOGGER.error("Authentication error during realtime STT: %s", err)
            else:
                coordinator.async_set_updated_data(None)
                LOGGER.error("Realtime STT websocket handshake failed: %s", err)
        except OSError as err:
            coordinator.mark_connection_error()
            LOGGER.error("Connection error during realtime STT: %s", err)
        except WebSocketConnectionClosedError as err:
            coordinator.mark_connection_error()
            LOGGER.error("Connection error during realtime STT: %s", err)
        except OpenAIError as err:
            coordinator.async_set_updated_data(None)
            LOGGER.error("Error during realtime STT: %s", err)
        except ConnectionClosed as err:
            coordinator.mark_connection_error()
            LOGGER.error("Connection error during realtime STT: %s", err)
        except WebSocketException as err:
            coordinator.async_set_updated_data(None)
            LOGGER.error("WebSocket error during realtime STT: %s", err)

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    def _realtime_audio_format(
        self, metadata: stt.SpeechMetadata
    ) -> dict[str, int | str]:
        """Return the PCM format declared for the realtime session."""
        if not self._audio_format_override:
            return {
                "type": "audio/pcm",
                "rate": metadata.sample_rate.value,
                "channels": metadata.channel.value,
            }
        return {
            "type": "audio/pcm",
            "rate": int(self.subentry.data[CONF_AUDIO_SAMPLE_RATE]),
            "channels": int(self.subentry.data[CONF_AUDIO_CHANNELS]),
        }

    def _transcription_options(self, metadata: stt.SpeechMetadata) -> dict[str, str]:
        """Return optional transcription parameters supported by the model."""
        if not self._supports_language:
            return {}
        return {"language": metadata.language.split("-")[0]}

    async def _async_process_batch(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process audio with the transcription endpoint."""
        audio_bytes = bytearray()
        async for chunk in stream:
            audio_bytes.extend(chunk)

        audio_data = bytes(audio_bytes)
        if metadata.format == stt.AudioFormats.WAV:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(metadata.channel.value)
                wav_file.setsampwidth(metadata.bit_rate.value // 8)
                wav_file.setframerate(metadata.sample_rate.value)
                wav_file.writeframes(audio_data)
            audio_data = wav_buffer.getvalue()

        coordinator = self.entry.runtime_data
        try:
            response = await coordinator.client.audio.transcriptions.create(
                model=self.model,
                file=(f"audio.{metadata.format.value}", audio_data),
                **cast(Any, self._transcription_options(metadata)),
            )
        except (AuthenticationError, PermissionDeniedError) as err:
            await coordinator.async_request_refresh()
            LOGGER.error("Authentication error during STT: %s", err)
        except APIConnectionError as err:
            coordinator.mark_connection_error()
            LOGGER.error("Connection error during STT: %s", err)
        except OpenAIError as err:
            coordinator.async_set_updated_data(None)
            LOGGER.error("Error during STT: %s", err)
        else:
            coordinator.async_set_updated_data(None)
            if response.text:
                return stt.SpeechResult(response.text, stt.SpeechResultState.SUCCESS)

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
