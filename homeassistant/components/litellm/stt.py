"""Speech-to-text support for LiteLLM."""

from collections.abc import AsyncIterable
import io
from typing import Any, cast, override
import wave

from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
)

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL, CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.template import Template

from .const import CONF_VOCABULARY, LOGGER, STT_BATCH_ENDPOINT
from .coordinator import LiteLLMConfigEntry
from .entity import LiteLLMEntity


def _render_template(hass: HomeAssistant, value: str) -> str:
    """Render an STT option template."""
    return cast(str, Template(value, hass).async_render(parse_result=False))


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
        return [stt.AudioFormats.WAV]

    @property
    @override
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    @override
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    @override
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported sample rates."""
        # LiteLLM accepts the OpenAI-compatible PCM sample-rate range. This is
        # a static API-level capability; model-specific audio capabilities are
        # not included in the model-group metadata.
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
        return [stt.AudioChannels.CHANNEL_MONO]

    @override
    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream."""
        if STT_BATCH_ENDPOINT in self._supported_endpoints:
            LOGGER.debug("LiteLLM STT using batch endpoint for model %s", self.model)
            return await self._async_process_batch(metadata, stream)
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    def _transcription_options(
        self, metadata: stt.SpeechMetadata
    ) -> dict[str, str | list[str]]:
        """Return optional transcription parameters supported by the model."""
        if not self._supports_language:
            options: dict[str, str | list[str]] = {}
        else:
            options = {"language": metadata.language.split("-")[0]}
        hass = self.entry.runtime_data.hass
        if prompt_template := self.subentry.data.get(CONF_PROMPT):
            if prompt := _render_template(hass, prompt_template):
                options["prompt"] = prompt
        if vocabulary_template := self.subentry.data.get(CONF_VOCABULARY):
            vocabulary = _render_template(hass, vocabulary_template)
            if keywords := [
                item.strip() for item in vocabulary.split(",") if item.strip()
            ]:
                options["keywords"] = keywords
        return options

    async def _async_process_batch(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process audio with the transcription endpoint."""
        audio_bytes = bytearray()
        async for chunk in stream:
            audio_bytes.extend(chunk)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(metadata.channel.value)
            wav_file.setsampwidth(metadata.bit_rate.value // 8)
            wav_file.setframerate(metadata.sample_rate.value)
            wav_file.writeframes(audio_bytes)

        coordinator = self.entry.runtime_data
        try:
            response = await coordinator.client.audio.transcriptions.create(
                model=self.model,
                file=("audio.wav", wav_buffer.getvalue()),
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
        except TemplateError as err:
            LOGGER.error("Error rendering STT template: %s", err)
        else:
            coordinator.async_set_updated_data(None)
            if response.text:
                return stt.SpeechResult(response.text, stt.SpeechResultState.SUCCESS)

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
