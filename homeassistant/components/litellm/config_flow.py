"""Config flow for LiteLLM integration."""

import logging
from typing import Any, cast, override

from openai import AsyncOpenAI, AuthenticationError, OpenAIError, PermissionDeniedError
import probatio
from yarl import URL

from homeassistant.components import stt
from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_AUDIO_CHANNELS,
    CONF_AUDIO_FORMAT_OVERRIDE,
    CONF_AUDIO_SAMPLE_RATE,
    CONF_PROMPT,
    DOMAIN,
    PLACEHOLDER_API_KEY,
    RECOMMENDED_CONVERSATION_OPTIONS,
    STT_BATCH_ENDPOINT,
    STT_REALTIME_ENDPOINT,
)
from .coordinator import ModelGroupInfo
from .url import normalize_url

_LOGGER = logging.getLogger(__name__)


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect to the proxy."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate the API key is invalid."""


async def _get_models(hass: HomeAssistant, url: str, api_key: str | None) -> list[str]:
    """Fetch the available model names from the LiteLLM proxy.

    Uses the OpenAI-compatible `/v1/models` endpoint, which a LiteLLM proxy
    serves with the configured model names.
    """
    client = AsyncOpenAI(
        base_url=url,
        api_key=api_key or PLACEHOLDER_API_KEY,
        # Legacy HTTPX clients are supported at runtime only.
        http_client=cast(Any, get_async_client(hass)),
    )
    try:
        return [
            model.id async for model in client.with_options(timeout=10.0).models.list()
        ]
    except (AuthenticationError, PermissionDeniedError) as err:
        raise InvalidAuth from err
    except OpenAIError as err:
        raise CannotConnect from err


class LiteLLMConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LiteLLM."""

    VERSION = 1

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            "conversation": ConversationFlowHandler,
            "stt": STTFlowHandler,
        }

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            url = normalize_url(user_input[CONF_URL])
            api_key = user_input.get(CONF_API_KEY)
            self._async_abort_entries_match({CONF_URL: url})
            try:
                await _get_models(self.hass, url, api_key)
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                data = {CONF_URL: url}
                if api_key:
                    data[CONF_API_KEY] = api_key
                return self.async_create_entry(
                    title=URL(url).host or url,
                    data=data,
                )
        return self.async_show_form(
            step_id="user",
            data_schema=probatio.Schema(
                {
                    probatio.Required(CONF_URL): str,
                    probatio.Optional(CONF_API_KEY): str,
                }
            ),
            errors=errors,
        )


class LiteLLMSubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for LiteLLM."""

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        self.models: list[str] = []

    async def _fetch_models(self) -> None:
        """Fetch models from the LiteLLM proxy."""
        entry = self._get_entry()
        self.models = await _get_models(
            self.hass, entry.data[CONF_URL], entry.data.get(CONF_API_KEY)
        )


class STTFlowHandler(ConfigSubentryFlow):
    """Handle STT subentry flow."""

    def __init__(self) -> None:
        """Initialize the STT subentry flow."""
        self.options: dict[str, Any] = {}
        self.model_groups: list[ModelGroupInfo] | None = None

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    def _supports_realtime(self, model_name: str) -> bool:
        """Return whether the selected model supports realtime transcription."""
        assert self.model_groups is not None
        return any(
            model["model_group"] == model_name
            and STT_REALTIME_ENDPOINT in (model.get("supported_endpoints") or [])
            for model in self.model_groups
        )

    def _finish(self) -> SubentryFlowResult:
        """Finish creating or updating the STT subentry."""
        entry = self._get_entry()
        model = self.options[CONF_MODEL]
        data = {CONF_MODEL: model}
        if self.options.get(CONF_AUDIO_FORMAT_OVERRIDE):
            data.update(
                {
                    CONF_AUDIO_FORMAT_OVERRIDE: True,
                    CONF_AUDIO_SAMPLE_RATE: self.options[CONF_AUDIO_SAMPLE_RATE],
                    CONF_AUDIO_CHANNELS: self.options[CONF_AUDIO_CHANNELS],
                }
            )
        if self._is_new:
            return self.async_create_entry(title=model, data=data)
        return self.async_update_and_abort(
            entry,
            self._get_reconfigure_subentry(),
            title=model,
            data=data,
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create an STT entity."""
        self.options = {}
        self.model_groups = None
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an STT entity."""
        self.options = self._get_reconfigure_subentry().data.copy()
        self.model_groups = None
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Select an STT model."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if self.model_groups is None:
            try:
                self.model_groups = await entry.runtime_data.async_get_model_groups()
            except (AuthenticationError, PermissionDeniedError):
                return self.async_abort(reason="invalid_auth")
            except OpenAIError:
                return self.async_abort(reason="cannot_connect")
            except Exception:
                _LOGGER.exception("Unexpected exception")
                return self.async_abort(reason="unknown")

        models = [
            model["model_group"]
            for model in self.model_groups
            if (
                model.get("mode") == "audio_transcription"
                and (endpoints := model.get("supported_endpoints"))
                and (
                    STT_BATCH_ENDPOINT in endpoints
                    or STT_REALTIME_ENDPOINT in endpoints
                )
            )
        ]

        if user_input is not None:
            self.options[CONF_MODEL] = user_input[CONF_MODEL]
            if self._supports_realtime(self.options[CONF_MODEL]):
                return await self.async_step_model()
            return self._finish()

        return self.async_show_form(
            step_id="init",
            data_schema=probatio.Schema(
                {
                    probatio.Required(
                        CONF_MODEL, default=self.options.get(CONF_MODEL)
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(value=model, label=model)
                                for model in models
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                            sort=True,
                        )
                    )
                }
            ),
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure realtime audio options."""
        if user_input is not None:
            self.options.update(user_input)
            if self.options.get(CONF_AUDIO_FORMAT_OVERRIDE):
                return await self.async_step_audio_format()
            return self._finish()

        return self.async_show_form(
            step_id="model",
            data_schema=probatio.Schema(
                {
                    probatio.Optional(
                        CONF_AUDIO_FORMAT_OVERRIDE,
                        default=self.options.get(CONF_AUDIO_FORMAT_OVERRIDE, False),
                    ): BooleanSelector(),
                }
            ),
            last_step=True,
        )

    async def async_step_audio_format(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure the realtime audio format override."""
        if user_input is not None:
            self.options.update(user_input)
            return self._finish()

        sample_rate_options = [
            SelectOptionDict(value=str(rate.value), label=f"{rate.value} Hz")
            for rate in (*stt.AudioSampleRates,)
        ]
        if "24000" not in {option["value"] for option in sample_rate_options}:
            sample_rate_options.append(
                SelectOptionDict(value="24000", label="24000 Hz")
            )
        return self.async_show_form(
            step_id="audio_format",
            data_schema=probatio.Schema(
                {
                    probatio.Required(
                        CONF_AUDIO_SAMPLE_RATE,
                        default=str(
                            self.options.get(CONF_AUDIO_SAMPLE_RATE, 24000)
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=sample_rate_options,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    probatio.Required(
                        CONF_AUDIO_CHANNELS,
                        default=str(self.options.get(CONF_AUDIO_CHANNELS, 1)),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(value="1", label="Mono"),
                                SelectOptionDict(value="2", label="Stereo"),
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
            last_step=True,
        )


class ConversationFlowHandler(LiteLLMSubentryFlowHandler):
    """Handle conversation subentry flow."""

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        super().__init__()
        self.options: dict[str, Any] = {}

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a conversation agent."""
        self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a conversation agent."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage conversation agent configuration."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            if user_input.get(CONF_LLM_HASS_API) is None:
                user_input.pop(CONF_LLM_HASS_API, None)
            if self._is_new:
                return self.async_create_entry(
                    title=user_input[CONF_MODEL], data=user_input
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                title=user_input[CONF_MODEL],
                data=user_input,
            )

        try:
            await self._fetch_models()
        except InvalidAuth:
            return self.async_abort(reason="invalid_auth")
        except CannotConnect:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        options = [SelectOptionDict(value=model, label=model) for model in self.models]

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        ]

        if suggested_llm_apis := self.options.get(CONF_LLM_HASS_API):
            valid_api_ids = {api["value"] for api in hass_apis}
            self.options[CONF_LLM_HASS_API] = [
                api for api in suggested_llm_apis if api in valid_api_ids
            ]

        return self.async_show_form(
            step_id="init",
            data_schema=probatio.Schema(
                {
                    probatio.Required(
                        CONF_MODEL, default=self.options.get(CONF_MODEL)
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=options, mode=SelectSelectorMode.DROPDOWN, sort=True
                        ),
                    ),
                    probatio.Optional(
                        CONF_PROMPT,
                        description={
                            "suggested_value": self.options.get(
                                CONF_PROMPT,
                                RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT],
                            )
                        },
                    ): TemplateSelector(),
                    probatio.Optional(
                        CONF_LLM_HASS_API,
                        default=self.options.get(
                            CONF_LLM_HASS_API,
                            RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API],
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True)
                    ),
                }
            ),
        )
