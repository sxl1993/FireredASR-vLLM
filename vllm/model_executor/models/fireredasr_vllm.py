"""
FireRedASR LLM mode integration for vLLM.

This module implements the vLLM integration for FireRedASR's LLM-based ASR model,
which uses a speech encoder + projector + Qwen2 LLM architecture.
"""
import os
from typing import Any, Iterable, List, Optional, Tuple, TypedDict, Union
from collections.abc import Iterable, Mapping
import torch
import torch.nn as nn
from transformers import PretrainedConfig, BatchFeature

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
    flatten_bn,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP, SupportsLoRA

# Import FireRedASR components
try:
    from .fireredasr.data.asr_feat import ASRFeatExtractor
    from .fireredasr.models.fireredasr_aed import FireRedAsrAed
    from .fireredasr.models.module.adapter import Adapter
except ImportError:
    ASRFeatExtractor = None
    FireRedAsrAed = None
    Adapter = None

# Import FireRedASR config from transformers_utils
from vllm.transformers_utils.configs.fireredasr import FireRedAsrConfig


# ============= Data Structures =============

class FireRedAsrInputs(TypedDict):
    """Type definition for FireRedASR audio inputs."""
    speech_features: torch.Tensor  # Shape: (batch, time, feat_dim)
    speech_lengths: torch.Tensor   # Shape: (batch,)


# ============= Processing Components =============

class FireRedAsrProcessingInfo(BaseProcessingInfo):
    """Processing information for FireRedASR."""

    def get_hf_config(self) -> PretrainedConfig:
        return FireRedAsrConfig.from_pretrained(self.ctx.model_config.model)

    def get_supported_mm_limits(self) -> dict[str, Optional[int]]:
        return {"audio": None}  # No limit on number of audio inputs

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> dict[str, int]:
        """Estimate max tokens per audio item."""
        hf_config = self.get_hf_config()
        # Approximate: audio length / downsample_rate
        # This is a rough estimate, actual value depends on audio duration
        # (patchy) TODO: should be same with the one in FireRedAsrMultiModalProcessor._calc_speech_features_time_length
        max_audio_tokens = seq_len // hf_config.encoder_downsample_rate // 4
        return {"audio": max_audio_tokens}


class FireRedAsrDummyInputsBuilder(BaseDummyInputsBuilder[FireRedAsrProcessingInfo]):
    """Dummy inputs builder for FireRedASR memory profiling."""

    def get_dummy_text(self, mm_counts: dict[str, int]) -> str:
        # """Generate dummy text with speech placeholders."""
        # num_audios = mm_counts.get("audio", 1)

        hf_config = self.info.get_hf_config()
        speech_token = hf_config.default_speech_token
        
        # Return speech tokens for each audio item
        return speech_token

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: dict[str, int],
    ) -> MultiModalDataDict:
        """Generate dummy audio data for memory profiling."""
        # num_audios = mm_counts.get("audio", 1)

        # hf_config = self.info.get_hf_config()

        # # Calculate audio length to maximize tokens
        # # We want to generate the worst-case scenario for memory profiling
        # # The number of tokens after processing is:
        # # audio_frames / encoder_downsample_rate / projector_downsample_rate
        # #
        # # To maximize tokens, we use a long audio duration
        # # Assuming 100 fps (frames per second) for fbank features
        # # and seq_len as target output tokens
        # fps = 100  # typical fbank feature rate
        # # total_downsample = hf_config.encoder_downsample_rate
        # total_downsample = hf_config.encoder_downsample_rate

        # # Calculate audio length that would result in seq_len tokens
        # # But cap it at a reasonable maximum (e.g., 30 seconds)
        # max_audio_frames = min(seq_len * total_downsample, 30 * fps)

        # # Convert frames to audio samples (assuming 16kHz sample rate)
        # sample_rate = 16000
        # audio_len = int((max_audio_frames / fps) * sample_rate)

        # TODO: should be a link of local file path
        return {
            "audio": ""
        }


class FireRedAsrMultiModalProcessor(BaseMultiModalProcessor[FireRedAsrProcessingInfo]):
    """Multimodal processor for FireRedASR."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get CMVN path from config (auto-resolved) or mm_kwargs (user override)
        cmvn_path = self.info.get_hf_config().cmvn_path
        if cmvn_path is None:
            raise ValueError(
                "cmvn_path could not be resolved. Please ensure the model directory "
                "contains 'cmvn.ark' or provide cmvn_path explicitly."
            )

        if not os.path.exists(cmvn_path):
            raise FileNotFoundError(f"CMVN file not found at {cmvn_path}")

        self.feat_extractor = ASRFeatExtractor(cmvn_path)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False


    def _calc_speech_features_time_length(self, feat_frames: int) -> int:
        """
        hard-code the speech_features time frame downsample calculation
        
        Args:
            feat_frames: time frames of the input features
        Returns:
            time frames of the final speech_features
        """
        # Step 1: Encoder Conv2dSubsampling
        padded = feat_frames + 6  # context=7, padding=6
        after_conv1 = (padded - 3) // 2 + 1
        encoder_frames = (after_conv1 - 3) // 2 + 1

        # Step 2: Adapter downsample
        speech_frames = encoder_frames // 2

        return max(1, speech_frames)

    def _get_mm_fields_config(
        self,
        hf_inputs: dict[str, Any],
        hf_processor_mm_kwargs: dict[str, Any],
    ) -> dict[str, MultiModalFieldConfig]:
        """Configure multimodal fields.

        Only include fields that will be passed to the model's forward method.
        Derived metadata like 'projected_lengths' should not be included here.
        """
        return {
            "speech_features": MultiModalFieldConfig.batched("audio"),
            "speech_lengths": MultiModalFieldConfig.batched("audio"),
        }

    def _check_audio_files(self, audio_data: list[str]) -> bool:
        """Check if the audio files exist."""
        if audio_data is None or len(audio_data) == 0:
            return False
        for audio_file in audio_data:
            if audio_file == "":
                return False
            expanded_path = os.path.expanduser(audio_file)
            if not os.path.exists(expanded_path):
                return False
        return True
        
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, Any],
        mm_kwargs: dict[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process audio data using ASRFeatExtractor."""
        if ASRFeatExtractor is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )

        # directly use tokenizer to encode the prompt text
        tokenizer = self.info.get_tokenizer()
        # Get padding side from config, default to 'left'
        hf_config = self.info.get_hf_config()
        desired_padding_side = getattr(hf_config, 'tokenizer_padding_side', 'left')
        if tokenizer.padding_side != desired_padding_side:
            tokenizer.padding_side = desired_padding_side
        # (patchy): The tokenizer should be the same as the one in fireredasr.py
        encoded = tokenizer([prompt], padding="longest",truncation=True, max_length=128)
        prompt_ids = torch.tensor(encoded["input_ids"])
        # attention_mask = torch.tensor(encoded["attention_mask"])

        # Process audio data
        # from fpdb import ForkedPdb; ForkedPdb().set_trace()
        audio_data = mm_data.get("audios", [])
        is_valid = self._check_audio_files(audio_data)
        if not is_valid:
            return BatchFeature({
                "speech_features": torch.empty(1,512, 80),
                "speech_lengths": torch.tensor([512], dtype=torch.long),
                "input_ids": prompt_ids,
            })
        # Extract features
        # audio_data should be list of file paths or audio arrays
        if isinstance(audio_data, list):
            feats, lengths, _ = self.feat_extractor(audio_data)
        else:
            feats, lengths, _ = self.feat_extractor([audio_data])

        # # (patchy) DEBUG
        # print(f"Processor input: feats shape: {feats.shape}")
        # print(f"Processor input: lengths shape: {lengths.shape}")

        return BatchFeature({
            "speech_features": feats,
            "speech_lengths": lengths,
            "input_ids": prompt_ids,
        })
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: dict[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptUpdate]:
        """
        Get prompt updates for FireRedASR.

        This method creates prompt updates that expand <speech> tokens to match the
        projector output length. The projected lengths are retrieved from the cache
        or computed from speech_lengths in out_mm_kwargs.
        """
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        # Get speech token and token ID
        speech_token = hf_config.default_speech_token
        vocab = tokenizer.get_vocab()
        speech_token_id = vocab[speech_token]

        # 提前提取和处理 speech_lengths
        speech_lengths_list = []
        speech_lengths_data = out_mm_kwargs.get("speech_lengths")

        if speech_lengths_data is not None:
            if isinstance(speech_lengths_data, torch.Tensor):
                # 转换为 list
                if speech_lengths_data.dim() == 1:
                    speech_lengths_list = [int(l.item()) for l in speech_lengths_data]
                elif speech_lengths_data.dim() == 0:
                    speech_lengths_list = [int(speech_lengths_data.item())]
            elif isinstance(speech_lengths_data, (list, tuple)):
                speech_lengths_list = [
                    int(item.item()) if isinstance(item, torch.Tensor) else int(item)
                    for item in speech_lengths_data
                ]
        else:
            print(">>>>>>>>> speech_lengths is not provided, use 1 as fallback <<<<<<<<<<<")
            speech_lengths_list = [1]
        speech_lengths_list = [self._calc_speech_features_time_length(l) for l in speech_lengths_list]

        def get_replacement_fireredasr(item_idx: int) -> list[int]:
            """Get replacement tokens for a specific audio item."""
            if item_idx < len(speech_lengths_list):
                num_tokens = speech_lengths_list[item_idx]
            else:
                num_tokens = 1  # Fallback
            return [speech_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[speech_token_id],
                replacement=get_replacement_fireredasr,
            )
        ]


# ============= Model Components =============

class FireRedAsrEncoder(nn.Module):
    """
    Wrapper for FireRedASR's speech encoder.
    Loads the encoder from FireRedAsrAed model.
    """

    def __init__(self, encoder_path: str):
        super().__init__()

        if FireRedAsrAed is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )

        # Load encoder architecture from checkpoint (args only, no weights)
        package = torch.load(encoder_path, map_location="cpu", weights_only=False)
        model = FireRedAsrAed.from_args(package["args"])

        # Note: encoder weights will be loaded later in load_weights() from model.pth.tar
        self.encoder = model.encoder
        self.encoder_dim = self.encoder.odim

        # IMPORTANT: Move encoder to the correct device
        # The encoder is loaded on CPU by default, but vLLM expects it on GPU
        # This will be handled by vLLM's model loading mechanism which moves
        # the entire model to the target device after initialization
    
    def forward(
        self,
        speech_features: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            speech_features: (batch, time, feat_dim)
            speech_lengths: (batch,)
        
        Returns:
            encoder_outputs: (batch, time', encoder_dim)
            output_lengths: (batch,)
            encoder_mask: (batch, 1, time')
        """
        encoder_outs, enc_lengths, enc_mask = self.encoder(speech_features, speech_lengths)
        return encoder_outs, enc_lengths, enc_mask


class FireRedAsrProjector(nn.Module):
    """
    Adapter/Projector that maps encoder outputs to LLM embedding space.
    This is a wrapper around FireRedASR's Adapter module.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 4,
    ):
        super().__init__()
        
        if Adapter is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )
        
        self.adapter = Adapter(encoder_dim, llm_dim, downsample_rate)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, time, encoder_dim)
            output_lengths: (batch,)
        
        Returns:
            projected_features: (batch, time', llm_dim)
            projected_lengths: (batch,)
        """
        return self.adapter(encoder_outputs, output_lengths)


# ============= Main Model =============

@MULTIMODAL_REGISTRY.register_processor(
    FireRedAsrMultiModalProcessor,
    info=FireRedAsrProcessingInfo,
    dummy_inputs=FireRedAsrDummyInputsBuilder,
)
class FireRedAsrForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    """
    FireRedASR model for conditional generation in vLLM.

    Architecture:
        Audio -> Encoder -> Projector -> LLM (Qwen2)
    """

    supports_multimodal: bool = True

    # Weight mapping from original FireRedASR checkpoint to vLLM model structure
    # This handles the prefix differences between training and inference
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Encoder mapping
            "encoder.": "speech_encoder.encoder.",
            # Projector/Adapter mapping
            "encoder_projector.": "projector.adapter.",
            # LoRA mapping (MUST come before base LLM mapping!)
            # Format in checkpoint: llm.base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
            # Format in vLLM with LoRA: language_model.model.layers.X.self_attn.q_proj.lora_A.default.weight
            "llm.base_model.model.": "language_model.",
            # LLM base mapping (for full finetuning scenario)
            "llm.model.": "language_model.model.",
            "llm.lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        # Extract configurations from vllm_config
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config

        # Save model directory for weight loading
        self.model_dir = vllm_config.model_config.model

        # Verify config type
        if not hasattr(config, 'model_type') or config.model_type != 'fireredasr':
            from vllm.transformers_utils.configs.fireredasr import FireRedAsrConfig
            if not isinstance(config, FireRedAsrConfig):
                raise ValueError(f"Expected FireRedAsrConfig, got {type(config)}")

        # Initialize speech encoder
        if config.encoder_path is None:
            raise ValueError("encoder_path must be provided in config. "
                           "Please ensure the model directory contains 'asr_encoder.pth.tar'")

        if not os.path.exists(config.encoder_path):
            raise FileNotFoundError(f"Encoder not found at {config.encoder_path}")

        self.speech_encoder = FireRedAsrEncoder(config.encoder_path)

        # Freeze encoder if required
        if config.freeze_encoder:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False
            self.speech_encoder.eval()

        # Get actual encoder dimension from loaded encoder
        encoder_dim = self.speech_encoder.encoder_dim

        # Initialize LLM
        if config.llm_dir is None:
            raise ValueError("llm_dir must be provided in config. "
                           "Please ensure the model directory contains a Qwen2 subdirectory")

        if not os.path.exists(config.llm_dir):
            raise FileNotFoundError(f"LLM directory not found at {config.llm_dir}")

        # Create a modified vllm_config for the LLM
        llm_vllm_config = self._create_llm_vllm_config(vllm_config, config.llm_dir)

        self.language_model = init_vllm_registered_model(
            vllm_config=llm_vllm_config,
            hf_config=llm_vllm_config.model_config.hf_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        # Get LLM hidden size
        llm_config = llm_vllm_config.model_config.hf_config
        llm_dim = llm_config.hidden_size

        # Initialize projector
        self.projector = FireRedAsrProjector(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            downsample_rate=config.encoder_downsample_rate,
        )

        self.projector.float().to(self.device)
        self.speech_encoder.float().to(self.device)
        self.projector.eval()
        self.speech_encoder.eval()
        self.language_model.eval()

        # Initialize sampler (if needed for standalone use)
        self.sampler = get_sampler()

    def _create_llm_vllm_config(self, base_vllm_config: VllmConfig, llm_dir: str) -> VllmConfig:
        """Create a VllmConfig for the internal LLM.

        IMPORTANT: We must share the compilation_config.static_forward_context
        with the base config. This is because Attention layers register themselves
        to static_forward_context during initialization, and these registrations
        must be visible to the torch.compile system.

        If we don't share the context, attention layers will register to a different
        dict, causing KeyError during compilation when trying to look up layers.
        """
        from vllm.config import ModelConfig
        from transformers import AutoConfig

        # Load LLM config from directory
        llm_hf_config = AutoConfig.from_pretrained(llm_dir)

        # Create new ModelConfig for the LLM
        llm_model_config = ModelConfig(
            model=llm_dir,
            tokenizer=base_vllm_config.model_config.tokenizer,
            tokenizer_mode=base_vllm_config.model_config.tokenizer_mode,
            trust_remote_code=base_vllm_config.model_config.trust_remote_code,
            dtype=base_vllm_config.model_config.dtype,
            seed=base_vllm_config.model_config.seed,
            revision=base_vllm_config.model_config.revision,
            code_revision=base_vllm_config.model_config.code_revision,
            rope_scaling=base_vllm_config.model_config.rope_scaling,
            rope_theta=base_vllm_config.model_config.rope_theta,
            tokenizer_revision=base_vllm_config.model_config.tokenizer_revision,
            max_model_len=base_vllm_config.model_config.max_model_len,
            quantization=base_vllm_config.model_config.quantization,
            enforce_eager=base_vllm_config.model_config.enforce_eager,
            max_seq_len_to_capture=base_vllm_config.model_config.max_seq_len_to_capture,
            max_logprobs=base_vllm_config.model_config.max_logprobs,
            disable_sliding_window=base_vllm_config.model_config.disable_sliding_window,
            skip_tokenizer_init=base_vllm_config.model_config.skip_tokenizer_init,
            served_model_name=base_vllm_config.model_config.served_model_name,
            # Set the HF config
            hf_config=llm_hf_config,
        )

        # Create a shallow copy of base_vllm_config
        # CRITICAL: We use shallow copy instead of deepcopy to share the
        # compilation_config.static_forward_context dict reference.
        # This ensures that attention layers in the LLM register to the same
        # context that will be used during torch.compile.
        llm_vllm_config = VllmConfig(
            model_config=llm_model_config,
            cache_config=base_vllm_config.cache_config,
            parallel_config=base_vllm_config.parallel_config,
            scheduler_config=base_vllm_config.scheduler_config,
            device_config=base_vllm_config.device_config,
            load_config=base_vllm_config.load_config,
            lora_config=base_vllm_config.lora_config,
            speculative_config=base_vllm_config.speculative_config,
            decoding_config=base_vllm_config.decoding_config,
            observability_config=base_vllm_config.observability_config,
            quant_config=base_vllm_config.quant_config,
            compilation_config=base_vllm_config.compilation_config,  # Share the same reference!
            kv_transfer_config=base_vllm_config.kv_transfer_config,
        )

        return llm_vllm_config
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        """
        Get placeholder text for FireRedASR audio inputs.
        """
        if modality == "audio":
            return "<speech>"
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def get_language_model(self) -> torch.nn.Module:
        """
        Returns the underlying language model used for text generation.
        """
        return self.language_model
    
    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        name: str,
    ) -> torch.Tensor:
        """Validate and reshape multimodal tensor input."""
        if mm_input is None:
            return torch.empty(0, device=self.device)
        
        if isinstance(mm_input, list):
            if not mm_input:
                return torch.empty(0, device=self.device)
            mm_input = torch.stack(mm_input)
        
        return mm_input
    
    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[FireRedAsrInputs]:
        """Parse and validate audio inputs."""
        speech_features = kwargs.pop("speech_features", None)
        speech_lengths = kwargs.pop("speech_lengths", None)

        if speech_features is None:
            return None

        # Handle list of tensors with different lengths - pad before stacking
        if isinstance(speech_features, list) and speech_features:
            # Find max length across all tensors
            max_time_len = max(feat.shape[1] for feat in speech_features)

            # Pad each tensor to max length
            padded_features = []
            for feat in speech_features:
                if feat.shape[1] < max_time_len:
                    # Pad on time dimension (dimension 1)
                    pad_len = max_time_len - feat.shape[1]
                    # Pad format: (left, right, top, bottom, front, back)
                    # We pad on the right side of dimension 1
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_len), value=0.0)
                padded_features.append(feat)
            speech_features = padded_features

        speech_features = self._validate_and_reshape_mm_tensor(
            speech_features, "speech_features"
        )
        speech_lengths = self._validate_and_reshape_mm_tensor(
            speech_lengths, "speech_lengths"
        )

        if speech_features.numel() == 0:
            return None

        speech_features = flatten_bn(speech_features, concat=True)
        speech_lengths = flatten_bn(speech_lengths, concat=True)

        return FireRedAsrInputs(
            speech_features=speech_features,
            speech_lengths=speech_lengths,
        )
    
    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings:
        """
        Process audio inputs and generate embeddings.

        Returns:
            Tuple of tensors, one per audio item, each with shape (num_tokens, llm_dim)
        """

        audio_inputs = self._parse_and_validate_audio_input(**kwargs)
        if audio_inputs is None:
            return tuple()

        speech_features = audio_inputs["speech_features"]
        speech_lengths = audio_inputs["speech_lengths"]

        # Run encoder with inference mode if encoder is frozen
        # This saves memory by not storing intermediate gradients

        with torch.inference_mode():
            encoder_outputs, output_lengths, encoder_mask = self.speech_encoder(
                speech_features, speech_lengths
            )

        # Explicitly delete encoder mask to free memory immediately
        # del encoder_mask

        # Run projector
        with torch.inference_mode():
            projected_features, projected_lengths = self.projector(
                encoder_outputs, output_lengths
            )

        # Delete encoder outputs after projector to free memory
        # del encoder_outputs

        # Return as tuple of tensors, one per audio item
        # Use contiguous() and clone() to ensure we only keep the valid parts
        batch_size = projected_features.size(0)
        feat_dim = projected_features.size(2)  # Get feature dimension (llm_dim)
        audio_embeddings_list = []

        for i in range(batch_size):
            actual_len = int(projected_lengths[i].item())
            # Clone the valid slice to create a new contiguous tensor
            # This allows the original projected_features tensor to be freed
            if actual_len == 0:
                # For zero-length projections, create a single zero-filled token
                # This handles edge cases with very short audio inputs
                valid_embedding = torch.zeros(1, feat_dim, device=self.device, dtype=projected_features.dtype)
            else:
                valid_embedding = projected_features[i, :actual_len].clone()
            audio_embeddings_list.append(valid_embedding)

        return tuple(audio_embeddings_list)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        """
        Merge text and audio embeddings using vLLM's standard utilities.
        """
        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings(input_ids=input_ids)

        if multimodal_embeddings is None or not multimodal_embeddings:
            return inputs_embeds

        # Use vLLM's standard multimodal embedding merging
        # merge_multimodal_embeddings handles the merging by matching
        # placeholder token IDs in input_ids with the multimodal embeddings
        # TODO: Use merge_multimodal_embeddings_from_map instead or update _get_prompt_updates to return placeholder map
        inputs_embeds = merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            multimodal_embeddings,
            self.config.speech_token_id
        )

        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass through the model."""

        if intermediate_tensors is not None:
            inputs_embeds = None

        # Process multimodal inputs if needed
        # In V1 the inputs_embeds should always be generated at model runner.
        if inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)

            inputs_embeds = self.get_input_embeddings(
                input_ids, multimodal_embeddings
            )
            input_ids = None  # Don't use input_ids when using embeddings

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample from logits."""
        return self.sampler(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Custom weight loading for FireRedASR.

        This method loads weights from two sources:
        1. Encoder and Projector weights from model.pth.tar (if exists)
        2. LLM weights from the separate llm_dir specified in config
        """
        import logging
        logger = logging.getLogger(__name__)

        # Check if we should load LLM from separate directory
        load_llm_separately = (self.config.llm_dir is not None and
                              os.path.exists(self.config.llm_dir))

        # Get model.pth.tar path for encoder/projector weights
        model_pth_path = os.path.join(self.model_dir, "model.pth.tar")
        has_model_pth = os.path.exists(model_pth_path)

        # Initialize weight containers
        encoder_weights = {}
        projector_weights = {}
        llm_weights = []

        # Load encoder and projector weights from model.pth.tar if it exists
        if has_model_pth:
            logger.info(f"Loading encoder/projector weights from {model_pth_path}")

            # Load the checkpoint
            try:
                package = torch.load(model_pth_path, map_location=lambda storage, loc: storage, weights_only=False)
            except Exception as e:
                logger.error(f"Failed to load {model_pth_path}: {e}")
                raise

            if "model_state_dict" not in package:
                logger.error(f"'model_state_dict' not found in {model_pth_path}")
                logger.error(f"Available keys: {list(package.keys())}")
                raise RuntimeError("Invalid checkpoint format")

            state_dict = package["model_state_dict"]
            logger.info(f"Loaded checkpoint with {len(state_dict)} parameters")

            # Process weights from model.pth.tar
            for orig_name, weight in state_dict.items():
                # Apply weight mapping
                mapped_name = orig_name
                for old_prefix, new_prefix in self.hf_to_vllm_mapper.orig_to_new_prefix.items():
                    if mapped_name.startswith(old_prefix):
                        mapped_name = new_prefix + mapped_name[len(old_prefix):]
                        break

                # Categorize weights
                if mapped_name.startswith("speech_encoder."):
                    encoder_weights[mapped_name] = weight
                elif mapped_name.startswith("projector."):
                    projector_weights[mapped_name] = weight
                elif mapped_name.startswith("language_model.") and not load_llm_separately:
                    # Only load LLM weights from model.pth.tar if not loading separately
                    llm_internal_name = mapped_name.replace("language_model.", "", 1)
                    llm_weights.append((llm_internal_name, weight))
        else:
            logger.warning(f"model.pth.tar not found at {model_pth_path}")
            logger.info("Encoder and projector weights will need to be loaded separately")

        # Load LLM weights from separate directory if specified
        if load_llm_separately:
            logger.info(f"Loading LLM weights from separate directory: {self.config.llm_dir}")

            # Use vLLM's model loader to load LLM weights
            from vllm.model_executor.model_loader import get_model_loader
            from vllm.config import LoadConfig, ModelConfig

            # from fpdb import ForkedPdb; ForkedPdb().set_trace()

            # Create a temporary ModelConfig for the LLM
            # Use tokenizer_path if available, otherwise fall back to llm_dir
            tokenizer_path = getattr(self.config, 'tokenizer_path', None) or self.config.llm_dir
            llm_model_config = ModelConfig(
                model=self.config.llm_dir,
                tokenizer=tokenizer_path,
                tokenizer_mode="auto",
                trust_remote_code=True,
                dtype=self.vllm_config.model_config.dtype,
                seed=self.vllm_config.model_config.seed,
            )

            # Create LoadConfig
            llm_load_config = LoadConfig(load_format="auto")

            # Get the model loader
            llm_loader = get_model_loader(llm_load_config)

            # Load LLM weights using the loader
            logger.info("Loading LLM weights using vLLM model loader...")
            llm_loader.load_weights(self.language_model, llm_model_config)
            logger.info("✓ LLM weights loaded from separate directory")

        elif llm_weights:
            # Load LLM weights from model.pth.tar (old behavior)
            logger.info(f"Loading {len(llm_weights)} LLM weights from model.pth.tar...")
            if hasattr(self.language_model, 'load_weights'):
                loaded_llm_params = self.language_model.load_weights(llm_weights)
                logger.info(f"Loaded LLM parameters: {len(loaded_llm_params) if loaded_llm_params else 'unknown'}")
            else:
                logger.info("Using state_dict fallback for LLM weights...")
                llm_state_dict = dict(llm_weights)
                missing_llm, unexpected_llm = self.language_model.load_state_dict(
                    llm_state_dict, strict=False
                )
                if missing_llm:
                    logger.warning(f"Missing LLM keys: {len(missing_llm)}")

        # Log what we found
        logger.info(f"Weight loading summary:")
        logger.info(f"  - Encoder parameters: {len(encoder_weights)}")
        logger.info(f"  - Projector parameters: {len(projector_weights)}")
        if load_llm_separately:
            logger.info(f"  - LLM: Loaded from {self.config.llm_dir}")
        else:
            logger.info(f"  - LLM parameters from model.pth.tar: {len(llm_weights)}")

        # Load encoder and projector weights using load_state_dict
        encoder_projector_weights = {**encoder_weights, **projector_weights}
        if encoder_projector_weights:
            logger.info(f"Loading encoder and projector weights...")
            missing_keys, unexpected_keys = self.load_state_dict(
                encoder_projector_weights, strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys when loading encoder/projector: {missing_keys[:5]}")

        # Mark all parameters as loaded for vLLM
        loaded_params = set(self.state_dict().keys())

        # Final summary
        current_state = self.state_dict()
        total_llm = sum(1 for k in current_state.keys() if k.startswith("language_model."))
        total_encoder = sum(1 for k in current_state.keys() if k.startswith("speech_encoder."))
        total_projector = sum(1 for k in current_state.keys() if k.startswith("projector."))

        logger.info(f"\n✓ Successfully loaded FireRedASR model weights from {model_pth_path}")
        logger.info(f"  Total model parameters:")
        logger.info(f"    - Speech Encoder: {total_encoder}")
        logger.info(f"    - Projector: {total_projector}")
        logger.info(f"    - LLM (Qwen2): {total_llm}")
        logger.info(f"    - Total: {len(loaded_params)}")

        return loaded_params
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
