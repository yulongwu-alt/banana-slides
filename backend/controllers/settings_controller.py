"""Settings Controller - file/env-backed runtime settings endpoints."""

import logging

from flask import Blueprint, current_app

from config import Config
from utils import error_response, success_response

logger = logging.getLogger(__name__)

settings_bp = Blueprint("settings", __name__, url_prefix="/api/settings")


@settings_bp.route("/", methods=["GET"], strict_slashes=False)
def get_settings():
    """GET /api/settings - Return the effective file/env-backed settings."""
    try:
        return success_response(_config_snapshot())
    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}")
        return error_response(
            "GET_SETTINGS_ERROR",
            f"Failed to get settings: {str(e)}",
            500,
        )


@settings_bp.route("/", methods=["PUT"], strict_slashes=False)
def update_settings():
    """PUT /api/settings - Runtime config is read-only."""
    logger.warning("Settings update rejected because runtime config is file/env-backed only")
    return error_response(
        "SETTINGS_READ_ONLY",
        "Runtime settings are read-only. Update the config file or environment variables and restart the backend.",
        409,
    )


@settings_bp.route("/reset", methods=["POST"], strict_slashes=False)
def reset_settings():
    """POST /api/settings/reset - Runtime config is read-only."""
    logger.warning("Settings reset rejected because runtime config is file/env-backed only")
    return error_response(
        "SETTINGS_READ_ONLY",
        "Runtime settings are read-only. Update the config file or environment variables and restart the backend.",
        409,
    )


def _mask_secret(value):
    """Mask secret values before returning them to clients."""
    if not value:
        return None
    if len(value) <= 4:
        return "***"
    return f"***{value[-4:]}"


def _config_snapshot():
    """Build a settings payload from file/env-backed config only."""
    config = current_app.config
    ai_provider_format = config.get("AI_PROVIDER_FORMAT", Config.AI_PROVIDER_FORMAT)

    if (ai_provider_format or "").lower() == "openai":
        api_base_url = config.get("OPENAI_API_BASE", Config.OPENAI_API_BASE) or None
        api_key = config.get("OPENAI_API_KEY", Config.OPENAI_API_KEY) or None
    else:
        api_base_url = config.get("GOOGLE_API_BASE", Config.GOOGLE_API_BASE) or None
        api_key = config.get("GOOGLE_API_KEY", Config.GOOGLE_API_KEY) or None

    mineru_token = config.get("MINERU_TOKEN", Config.MINERU_TOKEN) or None

    return {
        "id": 1,
        "ai_provider_format": ai_provider_format,
        "api_base_url": api_base_url,
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_masked": _mask_secret(api_key),
        "image_resolution": config.get("DEFAULT_RESOLUTION", Config.DEFAULT_RESOLUTION),
        "image_aspect_ratio": config.get("DEFAULT_ASPECT_RATIO", Config.DEFAULT_ASPECT_RATIO),
        "max_description_workers": config.get("MAX_DESCRIPTION_WORKERS", Config.MAX_DESCRIPTION_WORKERS),
        "max_image_workers": config.get("MAX_IMAGE_WORKERS", Config.MAX_IMAGE_WORKERS),
        "text_model": config.get("TEXT_MODEL", Config.TEXT_MODEL),
        "image_model": config.get("IMAGE_MODEL", Config.IMAGE_MODEL),
        "mineru_api_base": config.get("MINERU_API_BASE", Config.MINERU_API_BASE),
        "mineru_token_length": len(mineru_token) if mineru_token else 0,
        "mineru_token_masked": _mask_secret(mineru_token),
        "image_caption_model": config.get("IMAGE_CAPTION_MODEL", Config.IMAGE_CAPTION_MODEL),
        "output_language": config.get("OUTPUT_LANGUAGE", Config.OUTPUT_LANGUAGE),
        "config_source": "env/config file",
    }
