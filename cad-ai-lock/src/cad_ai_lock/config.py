"""Environment / settings for providers and generation defaults."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderName(str, Enum):
    MOCK = "mock"
    REPLICATE = "replicate"
    FAL = "fal"
    LOCAL = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cad_ai_provider: ProviderName = Field(
        default=ProviderName.MOCK, alias="CAD_AI_PROVIDER"
    )

    replicate_api_token: str | None = Field(default=None, alias="REPLICATE_API_TOKEN")
    # Public ControlNet canny on Replicate (jagilley/controlnet-canny)
    # Prefer REPLICATE_MODEL_NAME (owner/name). Optional version hash for /v1/predictions.
    replicate_controlnet_model: str = Field(
        default="jagilley/controlnet-canny",
        alias="REPLICATE_CONTROLNET_MODEL",
    )
    # Prefer versionless model name; generate.py normalizes
    replicate_model_name: str = Field(
        default="jagilley/controlnet-canny",
        alias="REPLICATE_MODEL_NAME",
    )

    fal_key: str | None = Field(default=None, alias="FAL_KEY")
    fal_controlnet_model: str = Field(
        default="fal-ai/flux/dev/image-to-image",
        alias="FAL_CONTROLNET_MODEL",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_vision_model: str = Field(
        default="gpt-4o-mini", alias="OPENAI_VISION_MODEL"
    )

    control_strength: float = Field(default=0.85, alias="CONTROL_STRENGTH")
    guidance_scale: float = Field(default=7.5, alias="GUIDANCE_SCALE")
    num_inference_steps: int = Field(default=30, alias="NUM_INFERENCE_STEPS")

    # Soft layout policy defaults (product toggles)
    policy_daylight: bool = Field(default=True, alias="POLICY_DAYLIGHT")
    policy_fengshui: bool = Field(default=False, alias="POLICY_FENGSHUI")
    policy_common_sense: bool = Field(default=True, alias="POLICY_COMMON_SENSE")
    policy_threshold: float = Field(default=0.55, alias="POLICY_THRESHOLD")
    structure_threshold: float = Field(default=0.35, alias="STRUCTURE_THRESHOLD")

    def resolve_provider(self) -> ProviderName:
        """Auto-pick real provider when keys exist and CAD_AI_PROVIDER=mock was defaulted."""
        explicit = os.environ.get("CAD_AI_PROVIDER")
        if explicit:
            return ProviderName(explicit.lower())
        if self.replicate_api_token:
            return ProviderName.REPLICATE
        if self.fal_key:
            return ProviderName.FAL
        return self.cad_ai_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
