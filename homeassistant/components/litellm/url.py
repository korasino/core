"""URL helpers for the LiteLLM integration."""

from yarl import URL


def _with_api_path(url: str, *, include_api_path: bool) -> str:
    """Return a LiteLLM URL with or without the OpenAI-compatible API path."""
    parsed = URL(url.strip())
    path = parsed.path.rstrip("/")
    if include_api_path:
        if not path.endswith("/v1"):
            path = f"{path}/v1"
    elif path.endswith("/v1"):
        path = path.removesuffix("/v1")
    return str(parsed.with_path(path))


def normalize_url(url: str) -> str:
    """Return the OpenAI-compatible API URL with a `/v1` path."""
    return _with_api_path(url, include_api_path=True)


def denormalize_url(url: str) -> str:
    """Return the LiteLLM proxy URL without the OpenAI-compatible `/v1` path."""
    return _with_api_path(url, include_api_path=False)
