"""
Example usage of FireRedASR with vLLM integration.

This example demonstrates how to use the FireRedASR model with vLLM
for efficient speech-to-text transcription.

IMPORTANT: Before running this example, ensure your model directory has a config.json file.
You can create one using:
    python fireredasr_setup_config.py /path/to/your/model/directory/

The model directory should contain:
- config.json (FireRedASR configuration)
- asr_encoder.pth.tar (ASR encoder checkpoint)
- cmvn.ark (CMVN statistics)
- Qwen2-7B-Instruct/ (LLM directory)
- model.pth.tar (optional, main model checkpoint)
"""

from vllm import LLM, SamplingParams

import os
import glob


def main():
    """Example of using FireRedASR with vLLM."""

    # Model configuration
    model_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/"  # Update with your model path
    tokenizer_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/modified_tokenizer"
    # Initialize LLM with FireRedASR
    # The model directory should contain:
    # 1. config.json with FireRedASR configuration
    # 2. asr_encoder.pth.tar
    # 3. cmvn.ark
    # 4. Qwen2-7B-Instruct/ subdirectory

    # Note: The FireRedAsrConfig should be saved as config.json in the model directory
    # with all necessary paths configured

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        disable_log_stats=True,  # Disable logging for cleaner output
        tokenizer=tokenizer_dir,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        dtype="float16",
        mm_processor_cache_gb=0, # disable mm_processor cache 
        max_num_seqs=64, # set max_num_seqs to 1 to avoid batching
    )
    
    # Sampling parameters for ASR
    sampling_params = SamplingParams(
        temperature=1.0,  # Greedy decoding for ASR
        max_tokens=4096,    # Adjust based on expected transcription length
        repetition_penalty=1.0,
        top_p=1.0,
    )
    
    # Prepare audio inputs
    # Option 1: Audio file paths
    audio_paths = [
        "~/deploy/vad_segments/video3511_1e49a7d7a4ffd445041c2a4e8ec350ad/seg_0000_speaker_7_12.99_42.55.wav",
    ]
    
    prompts = [
        {
            "prompt": "<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",  # Speech token placeholder
            "multi_modal_data": {
                "audio": audio_path
            }
        }
        for audio_path in audio_paths
    ]
    # Generate transcriptions
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for idx, output in enumerate(outputs):
        print(f"Audio {idx + 1}: {audio_paths[idx]}")
        print(f"Transcription: {output.outputs[0].text}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")
        print("-" * 50)


# def example_with_raw_audio():
#     """Example using raw audio tensors instead of file paths."""
#     import torch
#     from vllm import LLM, SamplingParams

#     model_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/"

#     llm = LLM(
#         model=model_dir,
#         trust_remote_code=True,
#         disable_log_stats=True,
#     )
    
#     sampling_params = SamplingParams(
#         temperature=0.0,
#         max_tokens=100,
#     )
    
#     # Create dummy audio tensor (batch_size, time_steps)
#     # In practice, load this from your audio processing pipeline
#     dummy_audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
    
#     prompts = [
#         {
#             "prompt": "<|SPEECH|>",
#             "multi_modal_data": {
#                 "audio": dummy_audio
#             }
#         }
#     ]
    
#     outputs = llm.generate(prompts, sampling_params)
    
#     for output in outputs:
#         print(f"Transcription: {output.outputs[0].text}")


def batch_transcription_example():
    """Example of batch transcription for efficiency."""
    import glob

    model_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/"
    tokenizer_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/modified_tokenizer"

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        disable_log_stats=True,  # Disable logging for cleaner output
        tokenizer=tokenizer_dir,
        gpu_memory_utilization=0.90,
        # enable_lora=True,
        # max_lora_rank=64,
    )

    
    # ASR-optimized sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,           # Greedy for accuracy
        max_tokens=4096,            # Max transcription length
        repetition_penalty=3.0,    # Can adjust if needed
        top_p=1.0,
        # For beam search:
        # best_of=5,
        # use_beam_search=True,
    )
    
    # Load all audio files from directory
    audio_dir = "/home/ray/deploy/vad_segments/video3511_1e49a7d7a4ffd445041c2a4e8ec350ad/"
    audio_files = glob.glob(f"{audio_dir}/*.wav")
    audio_files.sort()
    audio_files = audio_files[:16]
    
    print(f"Processing {len(audio_files)} audio files...")
    
    # Create prompts for batch processing
    prompts = [
        {
            "prompt": "<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {"audio": audio_file}
        }
        for audio_file in audio_files
    ]
    
    # Process in batch - vLLM will handle batching automatically
    outputs = llm.generate(prompts, sampling_params)
    
    # Save results
    results = []
    for audio_file, output in zip(audio_files, outputs):
        transcription = output.outputs[0].text
        results.append({
            "file": audio_file,
            "transcription": transcription,
            "num_tokens": len(output.outputs[0].token_ids),
        })
        print(f"{audio_file}: {transcription}")
    
    return results


def batch_transcription_with_chunking():
    """
    Memory-optimized batch transcription using chunking.

    This approach processes audio files in smaller batches to avoid OOM errors
    when dealing with large datasets (e.g., 200+ audio files).
    """
    import glob
    import torch

    model_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/"
    tokenizer_dir = "/home/ray/deploy/fireredasr_models/FireRedASR-LLM-L/modified_tokenizer"

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        disable_log_stats=True,
        tokenizer=tokenizer_dir,
        gpu_memory_utilization=0.9,  # Reduced from 0.95 to leave more headroom
        max_model_len=8192,
        mm_processor_cache_gb=1, # disable mm_processor cache
        max_num_seqs=64, # set max_num_seqs to 1 to avoid batching
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4096,
        repetition_penalty=1.0,
        top_p=1.0,
    )

    # Load all audio files
    audio_dir = "/home/ray/deploy/vad_segments/"  # Change this to your audio directory

    # Get all .wav files from the directory
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    audio_files.sort()
    audio_files = audio_files[2045:2046]

    print(f"Total audio files: {len(audio_files)}")

    print(f"audio_file: {audio_files[0]}")

    # Process in chunks to avoid OOM
    BATCH_SIZE = 32  # Adjust based on your GPU memory and audio length
    all_results = []


    import time
    start_time = time.perf_counter()
    for chunk_idx in range(0, len(audio_files), BATCH_SIZE):
        chunk_files = audio_files[chunk_idx:chunk_idx + BATCH_SIZE]

        chunk_num = chunk_idx // BATCH_SIZE + 1
        total_chunks = (len(audio_files) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nProcessing chunk {chunk_num}/{total_chunks}")
        print(f"Files {chunk_idx + 1} to {min(chunk_idx + BATCH_SIZE, len(audio_files))} of {len(audio_files)}")

        # Create prompts for this chunk
        chunk_prompts = [
            {
                "prompt": "<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"audio": audio_file}
            }
            for audio_file in chunk_files
        ]

        # Process chunk
        chunk_outputs = llm.generate(chunk_prompts, sampling_params)

        # Collect results
        for audio_file, output in zip(chunk_files, chunk_outputs):
            transcription = output.outputs[0].text
            all_results.append({
                "file": audio_file,
                "transcription": transcription,
                "num_tokens": len(output.outputs[0].token_ids),
            })
            print(f"[{len(all_results)}/{len(audio_files)}] {audio_file.split('/')[-1]}: {transcription}")

        # Optional: Clear CUDA cache between chunks to reduce fragmentation
        if chunk_idx + BATCH_SIZE < len(audio_files):
            torch.cuda.empty_cache()
            print(f"Memory cleared for next chunk")

    print(f"\n✓ Successfully processed all {len(all_results)} audio files")
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Throughput: {len(audio_files) / (end_time - start_time):.2f} qps")
    return all_results


if __name__ == "__main__":
    print("FireRedASR vLLM Integration Examples")
    print("=" * 60)

    # print("\n1. Basic Usage")
    # print("-" * 60)
    # Uncomment to run:
    # main()

    # print("\n2. Raw Audio Tensor Usage")
    # print("-" * 60)
    # # Uncomment to run:
    # example_with_raw_audio()

    # print("\n2. Batch Transcription (All at once)")
    # print("-" * 60)
    # Uncomment to run:
    # batch_transcription_example()

    print("\n3. Batch Transcription with Chunking (Memory-optimized)")
    print("-" * 60)
    print("This is the RECOMMENDED approach for large datasets (200+ files)")
    # Uncomment to run:
    batch_transcription_with_chunking()

    print("\nNote: Update model paths and uncomment examples to run.")