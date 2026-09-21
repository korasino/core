"""Speech-to-text support for LiteLLM."""

from collections.abc import AsyncIterable
import base64
import io
from typing import override
import wave

from openai import OpenAIError
from websockets.exceptions import WebSocketException

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_PROMPT, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.template import Template

from .const import CONF_VOCABULARY, LOGGER, STT_BATCH_ENDPOINT, STT_REALTIME_ENDPOINT
from .coordinator import LiteLLMConfigEntry, async_get_model_groups
from .entity import LiteLLMEntity


def _render_template(hass: HomeAssistant, value: str) -> str:
    """Render an STT option template."""
    return Template(value, hass).async_render(parse_result=False)



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

    model_groups = await async_get_model_groups(
        hass, config_entry.data[CONF_URL], config_entry.data.get(CONF_API_KEY)
    )
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
        self._supports_prompt = "prompt" in supported_openai_params
        self._supports_keywords = "keywords" in supported_openai_params

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return [
            "af-ZA",
            "am-ET",
            "ar-AE",
            "ar-BH",
            "ar-DZ",
            "ar-EG",
            "ar-IL",
            "ar-IQ",
            "ar-JO",
            "ar-KW",
            "ar-LB",
            "ar-MA",
            "ar-OM",
            "ar-PS",
            "ar-QA",
            "ar-SA",
            "ar-TN",
            "ar-YE",
            "az-AZ",
            "bg-BG",
            "bn-BD",
            "bn-IN",
            "bs-BA",
            "ca-ES",
            "cs-CZ",
            "da-DK",
            "de-AT",
            "de-CH",
            "de-DE",
            "el-GR",
            "en-AU",
            "en-CA",
            "en-GB",
            "en-GH",
            "en-HK",
            "en-IE",
            "en-IN",
            "en-KE",
            "en-NG",
            "en-NZ",
            "en-PH",
            "en-PK",
            "en-SG",
            "en-TZ",
            "en-US",
            "en-ZA",
            "es-AR",
            "es-BO",
            "es-CL",
            "es-CO",
            "es-CR",
            "es-DO",
            "es-EC",
            "es-ES",
            "es-GT",
            "es-HN",
            "es-MX",
            "es-NI",
            "es-PA",
            "es-PE",
            "es-PR",
            "es-PY",
            "es-SV",
            "es-US",
            "es-UY",
            "es-VE",
            "et-EE",
            "eu-ES",
            "fa-IR",
            "fi-FI",
            "fil-PH",
            "fr-BE",
            "fr-CA",
            "fr-CH",
            "fr-FR",
            "ga-IE",
            "gl-ES",
            "gu-IN",
            "he-IL",
            "hi-IN",
            "hr-HR",
            "hu-HU",
            "hy-AM",
            "id-ID",
            "is-IS",
            "it-CH",
            "it-IT",
            "iw-IL",
            "ja-JP",
            "jv-ID",
            "ka-GE",
            "kk-KZ",
            "km-KH",
            "kn-IN",
            "ko-KR",
            "lb-LU",
            "lo-LA",
            "lt-LT",
            "lv-LV",
            "mk-MK",
            "ml-IN",
            "mn-MN",
            "mr-IN",
            "ms-MY",
            "my-MM",
            "nb-NO",
            "ne-NP",
            "nl-BE",
            "nl-NL",
            "no-NO",
            "pl-PL",
            "pt-BR",
            "pt-PT",
            "ro-RO",
            "ru-RU",
            "si-LK",
            "sk-SK",
            "sl-SI",
            "sq-AL",
            "sr-RS",
            "su-ID",
            "sv-SE",
            "sw-KE",
            "sw-TZ",
            "ta-IN",
            "ta-LK",
            "ta-MY",
            "ta-SG",
            "te-IN",
            "th-TH",
            "tr-TR",
            "uk-UA",
            "ur-IN",
            "ur-PK",
            "uz-UZ",
            "vi-VN",
            "zh-CN",
            "zh-HK",
            "zh-TW",
            "zu-ZA",
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
        return [stt.AudioSampleRates.SAMPLERATE_16000]

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
        if STT_REALTIME_ENDPOINT in self._supported_endpoints:
            LOGGER.debug("LiteLLM STT using realtime endpoint for model %s", self.model)
            return await self._async_process_realtime(metadata, stream)
        if STT_BATCH_ENDPOINT in self._supported_endpoints:
            LOGGER.debug("LiteLLM STT using batch endpoint for model %s", self.model)
            return await self._async_process_batch(metadata, stream)
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    def _transcription_options(
        self, metadata: stt.SpeechMetadata
    ) -> dict[str, str | list[str]]:
        """Return optional transcription parameters supported by the model."""
        options: dict[str, str | list[str]] = {}
        if self._supports_language:
            # LiteLLM follows the OpenAI transcription contract, which uses
            # ISO-639-1 language codes rather than regional locale tags.
            options["language"] = metadata.language.split("-")[0]
        hass = self.entry.runtime_data.hass
        if self._supports_prompt and (prompt_template := self.subentry.data.get(CONF_PROMPT)):
            if prompt := _render_template(hass, prompt_template):
                options["prompt"] = prompt
        if self._supports_keywords and (
            vocabulary_template := self.subentry.data.get(CONF_VOCABULARY)
        ):
            if vocabulary := _render_template(hass, vocabulary_template):
                options["keywords"] = [vocabulary]
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

        try:
            response = await self.entry.runtime_data.client.audio.transcriptions.create(
                model=self.model,
                file=("audio.wav", wav_buffer.getvalue()),
                **self._transcription_options(metadata),
            )
        except (OpenAIError, TemplateError):
            LOGGER.exception("Error during STT")
        else:
            if response.text:
                return stt.SpeechResult(
                    response.text, stt.SpeechResultState.SUCCESS
                )

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

    async def _async_process_realtime(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Stream audio to LiteLLM's realtime transcription endpoint."""
        try:
            async with self.entry.runtime_data.client.realtime.connect(
                model=self.model,
                extra_query={"intent": "transcription"},
                max_retries=0,
            ) as connection:
                transcription: dict[str, str | list[str]] = {
                    "model": self.model,
                    **self._transcription_options(metadata),
                }
                LOGGER.debug(
                    "LiteLLM realtime STT session.update: model=%s language=%s keywords=%d sample_rate=%d",
                    self.model,
                    transcription.get("language"),
                    len(transcription.get("keywords", [])),
                    metadata.sample_rate.value,
                )
                await connection.session.update(
                    session={
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": 16000,
                                    "channels": 1,
                                },
                                "transcription": transcription,
                                "turn_detection": None,
                            }
                        },
                    }
                )
                async for chunk in stream:
                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(chunk).decode()
                    )
                LOGGER.debug("LiteLLM realtime STT committing input audio buffer")
                await connection.input_audio_buffer.commit()

                async for event in connection:
                    if (
                        event.type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        if event.transcript:
                            LOGGER.debug(
                                "LiteLLM realtime STT received final transcript (%d chars)",
                                len(event.transcript),
                            )
                            return stt.SpeechResult(
                                event.transcript, stt.SpeechResultState.SUCCESS
                            )
                        break
                    if event.type == "error":
                        LOGGER.error("Realtime STT error: %s", event)
                        break
        except (OpenAIError, TemplateError, WebSocketException, OSError):
            LOGGER.exception("Error during realtime STT")

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
