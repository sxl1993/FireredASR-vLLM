# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FireRedASR model configuration."""

import os
from typing import Optional

from transformers import PretrainedConfig


class FireRedAsrConfig(PretrainedConfig):
    """
    Configuration for FireRedASR model.

    This is the official config class for FireRedASR models in vLLM.
    It handles the configuration for:
    - Audio encoder (from FireRedASR)
    - Adapter/Projector layer
    - Language model (Qwen2)
    - Feature extraction parameters

    Supports automatic path resolution for nested model structure.
    """

    model_type = "fireredasr"

    def __init__(
        self,
        # Model directory (for automatic path resolution)
        model_dir: Optional[str] = None,
        asr_type: str = "llm",

        # Encoder configuration
        encoder_path: Optional[str] = None,
        model_path: Optional[str] = None,  # Main model checkpoint
        encoder_dim: int = 512,
        freeze_encoder: bool = True,

        # Projector configuration
        encoder_downsample_rate: int = 4,
        adapter_dim: Optional[int] = None,  # If None, uses llm hidden_size

        # LLM configuration
        llm_config: Optional[dict] = None,
        llm_dir: Optional[str] = None,
        freeze_llm: bool = False,

        # Tokenizer configuration
        tokenizer_path: Optional[str] = None,
        tokenizer_padding_side: str = "left",  # "left" or "right"

        # Feature extractor configuration
        cmvn_path: Optional[str] = None,
        sampling_rate: int = 16000,

        # Special tokens
        default_speech_token: str = "<speech>",
        speech_token_id: int = 151659,  # Default for Qwen2, will be set by tokenizer

        # Text configuration (for compatibility)
        text_config: Optional[dict] = None,

        **kwargs
    ):
        """
        Initialize FireRedASR configuration.

        Args:
            model_dir: Root directory containing all FireRedASR components
            asr_type: Type of ASR model ("llm" or "aed")
            encoder_path: Path to the pre-trained ASR encoder checkpoint
            model_path: Path to the main model checkpoint
            encoder_dim: Dimension of encoder outputs
            freeze_encoder: Whether to freeze encoder during training
            encoder_downsample_rate: Downsample rate of encoder (typically 4)
            adapter_dim: Dimension of adapter output (if None, uses LLM hidden size)
            llm_config: Configuration dict for the language model
            llm_dir: Directory containing the LLM model
            freeze_llm: Whether to freeze LLM during training
            cmvn_path: Path to CMVN statistics file
            sampling_rate: Audio sampling rate in Hz
            default_speech_token: Token representing speech in prompts
            speech_token_id: Token ID for speech token
            text_config: Text model configuration (for compatibility with vLLM)
            **kwargs: Additional arguments passed to PretrainedConfig
        """
        super().__init__(**kwargs)

        # Model directory and type
        self.model_dir = model_dir
        self.asr_type = asr_type

        # Encoder settings
        self.encoder_path = encoder_path
        self.model_path = model_path
        self.encoder_dim = encoder_dim
        self.freeze_encoder = freeze_encoder
        self.encoder_downsample_rate = encoder_downsample_rate

        # Adapter/Projector settings
        self.adapter_dim = adapter_dim

        # LLM settings
        self.llm_config = llm_config
        self.llm_dir = llm_dir
        self.freeze_llm = freeze_llm

        # Tokenizer settings
        self.tokenizer_path = tokenizer_path
        self.tokenizer_padding_side = tokenizer_padding_side

        # Feature extractor settings
        self.cmvn_path = cmvn_path
        self.sampling_rate = sampling_rate

        # Token settings
        self.default_speech_token = default_speech_token
        self.speech_token_id = speech_token_id

        # For vLLM compatibility - text_config points to the LLM config
        if text_config is not None:
            self.text_config = text_config
        elif llm_config is not None:
            # Map llm_config to text_config for vLLM compatibility
            self.text_config = llm_config
        else:
            # Will be set when loading the actual LLM
            self.text_config = None

        # Auto-resolve paths if model_dir is provided
        if self.model_dir and os.path.isdir(self.model_dir):
            self._resolve_paths()
    
    def get_tokenizer_path(self):
        """Get the tokenizer path to use for this model."""
        # Use custom tokenizer_path if available, otherwise use llm_dir
        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            return self.tokenizer_path
        elif self.llm_dir and os.path.exists(self.llm_dir):
            return self.llm_dir
        return None

    @classmethod
    def from_pretrained(cls, model_path_or_id, **kwargs):
        """
        Load config from a pretrained model directory.
        Automatically resolves all sub-model paths.
        """
        # Check if this is a FireRedASR model directory
        is_fireredasr_dir = False
        if os.path.isdir(model_path_or_id):
            # Check for FireRedASR-specific files
            required_files = ["cmvn.ark", "asr_encoder.pth.tar", "model.pth.tar"]
            has_required_files = all(
                os.path.exists(os.path.join(model_path_or_id, f))
                for f in required_files
            )
            # Also check for config.yaml (FireRedASR specific)
            has_config_yaml = os.path.exists(os.path.join(model_path_or_id, "config.yaml"))
            is_fireredasr_dir = has_required_files or has_config_yaml

        # First try to load existing config.json
        config = None
        config_json_path = os.path.join(model_path_or_id, "config.json") if os.path.isdir(model_path_or_id) else None

        if config_json_path and os.path.exists(config_json_path):
            # config.json exists, load it
            try:
                config = super().from_pretrained(model_path_or_id, **kwargs)
            except:
                config = None

        # If no config loaded and this is a FireRedASR directory, create default config
        if config is None and is_fireredasr_dir:
            config = cls(model_dir=model_path_or_id, **kwargs)
            # Optionally create config.json for future use
            cls._maybe_create_config_json(model_path_or_id)
        elif config is None:
            # Try standard loading
            try:
                config = super().from_pretrained(model_path_or_id, **kwargs)
            except:
                # If no config.json, create default config
                config = cls(**kwargs)

        # Auto-resolve paths if model_path_or_id is a directory
        if os.path.isdir(model_path_or_id):
            config.model_dir = model_path_or_id
            config._resolve_paths()

        return config

    @classmethod
    def _maybe_create_config_json(cls, model_dir: str):
        """
        Create a minimal config.json for FireRedASR model if it doesn't exist.
        This helps vLLM recognize the model type.
        """
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            import json
            minimal_config = {
                "model_type": "fireredasr",
                "architectures": ["FireRedAsrForConditionalGeneration"],
                "asr_type": "llm",
            }
            try:
                with open(config_path, "w") as f:
                    json.dump(minimal_config, f, indent=2)
                print(f"Created config.json for FireRedASR model at {config_path}")
            except Exception as e:
                # Don't fail if we can't write the file
                import warnings
                warnings.warn(f"Could not create config.json: {e}")

    def _resolve_paths(self):
        """Automatically resolve all sub-model paths from model_dir."""
        if not self.model_dir:
            return

        # Set default paths if not explicitly provided
        if self.cmvn_path is None:
            cmvn_path = os.path.join(self.model_dir, "cmvn.ark")
            if os.path.exists(cmvn_path):
                self.cmvn_path = cmvn_path

        if self.model_path is None:
            model_path = os.path.join(self.model_dir, "model.pth.tar")
            if os.path.exists(model_path):
                self.model_path = model_path

        if self.encoder_path is None:
            encoder_path = os.path.join(self.model_dir, "asr_encoder.pth.tar")
            if os.path.exists(encoder_path):
                self.encoder_path = encoder_path

        if self.llm_dir is None:
            # Try to find Qwen model directory
            possible_dirs = [
                "Qwen2-7B-Instruct",
                "Qwen2-1.5B-Instruct",
                "Qwen2.5-7B-Instruct",
                "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-0.5B-Instruct",
            ]
            for dir_name in possible_dirs:
                llm_path = os.path.join(self.model_dir, dir_name)
                if os.path.exists(llm_path):
                    # Resolve symlinks to get the actual path
                    self.llm_dir = os.path.realpath(llm_path)
                    break

            if self.llm_dir is None:
                # Fallback: look for any directory containing "qwen"
                for item in os.listdir(self.model_dir):
                    item_path = os.path.join(self.model_dir, item)
                    if os.path.isdir(item_path) and "qwen" in item.lower():
                        # Resolve symlinks to get the actual path
                        self.llm_dir = os.path.realpath(item_path)
                        break

        # Set tokenizer path if not explicitly provided
        if self.tokenizer_path is None:
            tokenizer_path = os.path.join(self.model_dir, "modified_tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer_path = tokenizer_path

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required files exist and emit warnings for missing ones."""
        missing = []

        # Check critical paths for LLM mode
        if self.asr_type == "llm":
            if self.cmvn_path and not os.path.exists(self.cmvn_path):
                missing.append(f"CMVN file: {self.cmvn_path}")

            if self.encoder_path and not os.path.exists(self.encoder_path):
                missing.append(f"Encoder: {self.encoder_path}")

            if self.llm_dir and not os.path.exists(self.llm_dir):
                missing.append(f"LLM directory: {self.llm_dir}")

        if missing:
            import warnings
            warnings.warn(f"Missing model files: {', '.join(missing)}")

    def get_text_config(self, **kwargs) -> "PretrainedConfig":
        """
        Get the text configuration.

        This method is used by vLLM to access the underlying LLM config.
        For FireRedASR, the text config is the Qwen2 LLM config.
        """
        if self.text_config is not None:
            # If text_config is a dict, convert to PretrainedConfig
            if isinstance(self.text_config, dict):
                from transformers import AutoConfig
                # Create config from dict
                return PretrainedConfig(**self.text_config)
            return self.text_config

        # Try to load from llm_dir if available
        if self.llm_dir and os.path.exists(self.llm_dir):
            from transformers import AutoConfig
            try:
                return AutoConfig.from_pretrained(self.llm_dir)
            except:
                pass

        # Return self as fallback (vLLM expects some config)
        return self


# Register the config with transformers
try:
    from transformers import AutoConfig
    AutoConfig.register("fireredasr", FireRedAsrConfig)
except Exception:
    # Registration might fail in some contexts, that's okay
    pass

