"""
Script to save a modified LLM tokenizer with FireRedASR-specific changes.

This creates a standalone tokenizer folder that can be loaded directly
without needing to apply modifications at runtime.
"""

import os
import argparse
from transformers import AutoTokenizer

DEFAULT_SPEECH_TOKEN = "<speech>"


def save_modified_tokenizer(
    llm_path: str,
    output_path: str,
    use_flash_attn: bool = True,
):
    """
    Load an LLM tokenizer, apply FireRedASR modifications, and save it.

    Args:
        llm_path: Path to the original LLM tokenizer (e.g., Qwen2.5 model path)
        output_path: Path to save the modified tokenizer
        use_flash_attn: If True, set padding_side to "left"; otherwise "right"
    """
    # Load original tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    # Apply modifications
    # 1. Set padding side
    if use_flash_attn:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    # 2. Add special tokens
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    # Save the modified tokenizer
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)

    print(f"Modified tokenizer saved to: {output_path}")
    print(f"  - padding_side: {tokenizer.padding_side}")
    print(f"  - Added {num_added} special token(s): {DEFAULT_SPEECH_TOKEN}")
    print(f"  - Vocabulary size: {len(tokenizer)}")
    print(f"\nYou can now load this tokenizer directly with:")
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_path}")')

    return tokenizer


def verify_tokenizer(output_path: str):
    """
    Verify that the saved tokenizer can be loaded and has the expected modifications.
    """
    tokenizer = AutoTokenizer.from_pretrained(output_path)

    print(f"\nVerification of saved tokenizer at: {output_path}")
    print(f"  - padding_side: {tokenizer.padding_side}")
    print(f"  - Vocabulary size: {len(tokenizer)}")

    # Check if special token exists
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    if speech_token_id != tokenizer.unk_token_id:
        print(f"  - {DEFAULT_SPEECH_TOKEN} token ID: {speech_token_id}")
    else:
        print(f"  - WARNING: {DEFAULT_SPEECH_TOKEN} token not found!")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a modified LLM tokenizer with FireRedASR-specific changes"
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        required=True,
        help="Path to the original LLM (e.g., Qwen2.5 model path)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the modified tokenizer",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=True,
        help="Use flash attention (sets padding_side to 'left')",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the saved tokenizer after saving",
    )

    args = parser.parse_args()

    save_modified_tokenizer(
        llm_path=args.llm_path,
        output_path=args.output_path,
        use_flash_attn=args.use_flash_attn,
    )

    if args.verify:
        verify_tokenizer(args.output_path)
