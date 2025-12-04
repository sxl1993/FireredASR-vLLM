#!/usr/bin/env python3
"""
LoRA weights merging script

This script merges LoRA weights from the FireRedASR model into the base LLM model,
generating a new model with complete weights, eliminating the need to load LoRA adapters during inference.

Usage:
    python merge_lora_weights.py \
        --model_path ~/deploy/fireredasr_models/FireRedASR-LLM-L/model.pth.tar \
        --llm_dir ~/deploy/fireredasr_models/FireRedASR-LLM-L/Qwen2-7B-Instruct \
        --output_dir ~/deploy/fireredasr_models/FireRedASR-LLM-L/Qwen2-7B-Instruct-Merged

Output:
    - The merged model will be saved in output_dir
    - Contains all original LLM configuration files and the merged weights
"""

import argparse
import logging
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_lora_weights_from_checkpoint(checkpoint_path):
    """
    Extract LoRA weights from model.pth.tar

    Args:
        checkpoint_path: Path to model.pth.tar file

    Returns:
        lora_state_dict: state_dict containing only LoRA weights
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    package = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' not in package:
        raise KeyError("'model_state_dict' not found in checkpoint")

    full_state_dict = package['model_state_dict']

    # Extract LoRA weights (keys starting with 'llm.base_model.model')
    lora_state_dict = {}
    lora_count = 0

    for key, value in full_state_dict.items():
        if key.startswith('llm.'):
            # Remove 'llm.' prefix, as the PEFT model will automatically add this prefix
            new_key = key[4:]  # Remove 'llm.'
            lora_state_dict[new_key] = value
            lora_count += 1

    logger.info(f"Extracted {lora_count} LoRA weight tensors from checkpoint")

    if lora_count == 0:
        logger.warning("No LoRA weights found in checkpoint!")

    return lora_state_dict


def create_peft_model_with_lora(base_model, lora_config):
    """
    Create a PEFT model with LoRA configuration

    Args:
        base_model: Base LLM model
        lora_config: LoRA configuration

    Returns:
        peft_model: Model with LoRA adapters
    """
    logger.info("Creating PEFT model with LoRA configuration")
    peft_model = get_peft_model(base_model, lora_config)
    logger.info("PEFT model created successfully")
    peft_model.print_trainable_parameters()

    return peft_model


def merge_lora_to_base_model(
    model_path,
    llm_dir,
    output_dir,
    use_flash_attn=True,
    torch_dtype='auto'
):
    """
    Merge LoRA weights into the base model

    Args:
        model_path: Path to model.pth.tar file
        llm_dir: Base LLM model directory
        output_dir: Output directory
        use_flash_attn: Whether to use flash attention
        torch_dtype: Data type ('auto', 'bfloat16', 'float16', 'float32')
    """

    # 1. Set data type and attention implementation
    if use_flash_attn:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    # Handle torch_dtype
    if torch_dtype == 'auto':
        # Use dtype from model config
        dtype = 'auto'
        logger.info("Using dtype: auto (from model config)")
    elif torch_dtype == 'bfloat16':
        dtype = torch.bfloat16
        logger.info("Using dtype: bfloat16")
    elif torch_dtype == 'float16':
        dtype = torch.float16
        logger.info("Using dtype: float16")
    elif torch_dtype == 'float32':
        dtype = torch.float32
        logger.info("Using dtype: float32")
    else:
        dtype = 'auto'
        logger.warning(f"Unknown dtype '{torch_dtype}', using 'auto'")

    logger.info(f"Using attention implementation: {attn_implementation}")

    # 2. Load base LLM model
    logger.info(f"Loading base LLM from {llm_dir}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            llm_dir,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        logger.info("Base LLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load base LLM: {e}")
        logger.info("Retrying without flash attention...")
        base_model = AutoModelForCausalLM.from_pretrained(
            llm_dir,
            torch_dtype=dtype,
            trust_remote_code=True
        )

    # 3. Create LoRA configuration (must match the training configuration)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # 4. Apply LoRA configuration to base model
    peft_model = create_peft_model_with_lora(base_model, lora_config)

    # 5. Load LoRA weights
    lora_state_dict = load_lora_weights_from_checkpoint(model_path)

    logger.info("Loading LoRA weights into PEFT model")
    try:
        # Load weights, allowing partial matches
        missing_keys, unexpected_keys = peft_model.load_state_dict(
            lora_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)}")
            logger.warning(f"Missing keys: {missing_keys[:10]}...")  # Show only the first 10

        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
            logger.warning(f"Unexpected keys: {unexpected_keys[:10]}...")

        logger.info("LoRA weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LoRA weights: {e}")
        raise

    # 6. Merge LoRA weights into base model
    logger.info("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    logger.info("LoRA weights merged successfully")

    # 7. Save merged model
    logger.info(f"Saving merged model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True  # Use safetensors format
    )

    # 8. Copy tokenizer and configuration files
    logger.info("Copying tokenizer and configuration files")
    tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Copy other necessary files
    files_to_copy = [
        'tokenizer_config.json',
        'special_tokens_map.json',
        'generation_config.json',
    ]

    for filename in files_to_copy:
        src = os.path.join(llm_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            logger.debug(f"Copied {filename}")

    logger.info("="*50)
    logger.info("LoRA weight merging completed successfully!")
    logger.info(f"Merged model saved to: {output_dir}")
    logger.info("="*50)

    return merged_model


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base LLM model"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model.pth.tar file containing LoRA weights'
    )

    parser.add_argument(
        '--llm_dir',
        type=str,
        required=True,
        help='Path to base LLM directory (e.g., Qwen2-7B-Instruct)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for merged model'
    )

    parser.add_argument(
        '--no_flash_attn',
        action='store_true',
        help='Disable flash attention'
    )

    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'bfloat16', 'float16', 'float32'],
        help='Data type for model weights (default: auto - use model config)'
    )

    args = parser.parse_args()

    # Expand ~ in paths
    model_path = os.path.expanduser(args.model_path)
    llm_dir = os.path.expanduser(args.llm_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # Validate input paths
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return 1

    if not os.path.exists(llm_dir):
        logger.error(f"LLM directory not found: {llm_dir}")
        return 1

    # Execute merge
    try:
        merge_lora_to_base_model(
            model_path=model_path,
            llm_dir=llm_dir,
            output_dir=output_dir,
            use_flash_attn=not args.no_flash_attn,
            torch_dtype=args.dtype
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to merge LoRA weights: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
