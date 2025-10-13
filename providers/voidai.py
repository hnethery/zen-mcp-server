"""VoidAI model provider implementation."""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .shared import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    TemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class VoidAIModelProvider(OpenAICompatibleProvider):
    """VoidAI API provider (api.voidai.app)."""

    FRIENDLY_NAME = "VoidAI"

    # Model configurations using ModelCapabilities objects - Curated list of best models
    MODEL_CAPABILITIES = {
        # === OpenAI Models ===
        "gpt-5": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="gpt-5",
            friendly_name="VoidAI (GPT-5)",
            context_window=400_000,
            max_output_tokens=128_000,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="GPT-5 via VoidAI - Latest OpenAI model with massive context and capabilities",
            aliases=["gpt5", "gpt-5"],
        ),
        "o1": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="o1",
            friendly_name="VoidAI (o1)",
            context_window=200_000,
            max_output_tokens=100_000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=False,
            temperature_constraint=TemperatureConstraint.create("fixed"),
            description="o1 via VoidAI - OpenAI's reasoning model with extended thinking",
            aliases=[],
        ),
        "o3": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="o3",
            friendly_name="VoidAI (o3)",
            context_window=200_000,
            max_output_tokens=100_000,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,
            temperature_constraint=TemperatureConstraint.create("fixed"),
            description="o3 via VoidAI - OpenAI's advanced reasoning model",
            aliases=[],
        ),
        "o3-mini": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="o3-mini",
            friendly_name="VoidAI (o3-mini)",
            context_window=200_000,
            max_output_tokens=100_000,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=False,
            temperature_constraint=TemperatureConstraint.create("fixed"),
            description="o3-mini via VoidAI - Efficient version of o3 reasoning model",
            aliases=["o3mini"],
        ),
        "gpt-4o": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="gpt-4o",
            friendly_name="VoidAI (GPT-4o)",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="GPT-4o via VoidAI - Advanced multimodal model with vision and function calling",
            aliases=["gpt4o", "4o"],
        ),
        "gpt-4o-mini": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="gpt-4o-mini",
            friendly_name="VoidAI (GPT-4o Mini)",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="GPT-4o Mini via VoidAI - Fast and efficient multimodal model",
            aliases=["gpt4o-mini", "4o-mini"],
        ),
        "gpt-4.1": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="gpt-4.1",
            friendly_name="VoidAI (GPT-4.1)",
            context_window=200_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="GPT-4.1 via VoidAI - Enhanced GPT-4 model with improved capabilities",
            aliases=["gpt4.1"],
        ),
        # === Anthropic Claude Models ===
        "claude-sonnet-4-5-20250929": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="claude-sonnet-4-20250514",
            friendly_name="VoidAI (Claude 4.5 Sonnet)",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Claude 4 Sonnet via VoidAI - Latest Claude model with enhanced capabilities",
            aliases=["claude-4.5-sonnet", "sonnet-4.5", "sonnet"],
        ),
        "claude-opus-4-1-20250805": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="claude-opus-4-1-20250805",
            friendly_name="VoidAI (Claude 4.1 Opus)",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=False,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Claude 4.1 Opus via VoidAI - Most capable Claude model with latest enhancements",
            aliases=["claude-4.1-opus", "opus-4.1", "opus"],
        ),
        # === Google Gemini Models ===
        "gemini-2.5-pro": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="gemini-2.5-pro",
            friendly_name="VoidAI (Gemini 2.5 Pro)",
            context_window=1_000_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Gemini 2.5 Pro via VoidAI - Most advanced Gemini model with ultra-large context",
            aliases=["gemini-pro", "pro"],
        ),
        "gemini-2.5-flash": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="gemini-2.5-flash",
            friendly_name="VoidAI (Gemini 2.5 Flash)",
            context_window=1_000_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=20.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Gemini 2.5 Flash via VoidAI - Ultra-large context window for extensive documents",
            aliases=["gemini-flash", "flash"],
        ),
        # === X.AI Grok Models ===
        "grok-4": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="grok-4",
            friendly_name="VoidAI (Grok 4)",
            context_window=131_072,
            max_output_tokens=131_072,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=False,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Grok 4 via VoidAI - Latest Grok model with enhanced reasoning",
            aliases=["grok4"],
        ),
        # === Other Advanced Models ===
        "mistral-large-latest": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="mistral-large-latest",
            friendly_name="VoidAI (Mistral Large)",
            context_window=128_000,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Mistral Large via VoidAI - Most capable Mistral model",
            aliases=["mistral-large", "mistral"],
        ),
        "sonar-reasoning-pro": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="sonar-reasoning-pro",
            friendly_name="VoidAI (Perplexity Sonar Reasoning Pro)",
            context_window=32_768,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_json_mode=False,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Perplexity Sonar Reasoning Pro via VoidAI - Advanced search with reasoning",
            aliases=["perplexity-reasoning"],
        ),
        "kimi-k2-instruct": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="kimi-k2-instruct",
            friendly_name="VoidAI (Kimi K2)",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Kimi K2 via VoidAI - Advanced Chinese model with strong reasoning",
            aliases=["kimi"],
        ),
        "deepseek-r1": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="deepseek-r1",
            friendly_name="VoidAI (DeepSeek R1)",
            context_window=64_000,
            max_output_tokens=8_192,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="DeepSeek R1 via VoidAI - Reasoning model with extended thinking",
            aliases=["deepseek"],
        ),
        "deepseek-v3.1:thinking": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="deepseek-v3.1:thinking",
            friendly_name="VoidAI (DeepSeek V3.1:thinking)",
            context_window=64_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="DeepSeek V3.1:terminus via VoidAI - Advanced coding and reasoning model",
            aliases=["deepseek:v3.1-terminus"],
        ),
        "deepseek-v3.1": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="deepseek-v3.1",
            friendly_name="VoidAI (DeepSeek V3.1)",
            context_window=64_000,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="DeepSeek V3.1 via VoidAI - Advanced coding and reasoning model",
            aliases=["deepseek:v3.1"],
        ),
        "qwen3-coder-480b-a35b-instruct": ModelCapabilities(
            provider=ProviderType.VOIDAI,
            model_name="qwen3-coder-480b-a35b-instruct",
            friendly_name="VoidAI (Qwen3 Coder 480B)",
            context_window=32_768,
            max_output_tokens=8_192,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=False,
            max_image_size_mb=0.0,
            supports_temperature=True,
            temperature_constraint=TemperatureConstraint.create("range"),
            description="Qwen3 Coder 480B via VoidAI - Specialized coding model with massive parameters",
            aliases=["qwen3-coder", "qwen-coder"],
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize VoidAI provider with API key."""
        # Set VoidAI base URL (use proxy if available)
        kwargs.setdefault("base_url", "http://localhost:8000/v1")
        super().__init__(api_key, **kwargs)

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific VoidAI model."""
        # Resolve shorthand
        resolved_name = self._resolve_model_name(model_name)

        if resolved_name not in self.MODEL_CAPABILITIES:
            raise ValueError(f"Unsupported VoidAI model: {model_name}")

        # Check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(
            ProviderType.VOIDAI, resolved_name, model_name
        ):
            raise ValueError(
                f"VoidAI model '{model_name}' is not allowed by restriction policy."
            )

        # Return the ModelCapabilities object directly from MODEL_CAPABILITIES
        return self.MODEL_CAPABILITIES[resolved_name]

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.VOIDAI

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported and allowed."""
        resolved_name = self._resolve_model_name(model_name)

        # First check if model is supported
        if resolved_name not in self.MODEL_CAPABILITIES:
            return False

        # Then check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(
            ProviderType.VOIDAI, resolved_name, model_name
        ):
            logger.debug(
                f"VoidAI model '{model_name}' -> '{resolved_name}' blocked by restrictions"
            )
            return False

        return True

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using VoidAI API with proper model name resolution."""
        # Resolve model alias before making API call
        resolved_model_name = self._resolve_model_name(model_name)

        # Call parent implementation with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        capabilities = self.MODEL_CAPABILITIES.get(resolved_name)
        if capabilities:
            return capabilities.supports_extended_thinking
        return False

    def get_preferred_model(
        self, category: "ToolModelCategory", allowed_models: list[str]
    ) -> Optional[str]:
        """Get VoidAI's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer Claude for advanced reasoning tasks
            if "claude-3.5-sonnet" in allowed_models:
                return "claude-3.5-sonnet"
            elif "gpt-4o" in allowed_models:
                return "gpt-4o"
            elif "grok-3" in allowed_models:
                return "grok-3"
            # Fall back to any available model
            return allowed_models[0]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer faster models for speed
            if "claude-3.5-haiku" in allowed_models:
                return "claude-3.5-haiku"
            elif "gpt-4o-mini" in allowed_models:
                return "gpt-4o-mini"
            elif "gemini-2.5-flash" in allowed_models:
                return "gemini-2.5-flash"
            # Fall back to any available model
            return allowed_models[0]

        else:  # BALANCED or default
            # Prefer Claude Sonnet for balanced use (excellent for coding/analysis)
            if "claude-3.5-sonnet" in allowed_models:
                return "claude-3.5-sonnet"
            elif "gpt-4o" in allowed_models:
                return "gpt-4o"
            elif "gemini-2.5-flash" in allowed_models:
                return "gemini-2.5-flash"
            # Fall back to any available model
            return allowed_models[0]
