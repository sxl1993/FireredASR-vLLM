<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Summary

The current repo is a specialized adaptation tailored to the original FireredASR-LLM model architecture and input parameters, containing extensive hard-coded elements. Significant work remains to be done before it can be merged into the main vLLM branch:

- [ ] Modify the FireredASR-LLM model files to match the standard loading procedure in vLLM
- [ ] Modify the input format to support raw features data
- [ ] Remove the separate fireredasr directory in `vllm/model_executor/models`


## Getting Started

1. Run `merge_lora_weights.py` under the directory of `FireRedASR-LLM-L` to get the complete Qwen2-7B LLM model with LoRA weights.

2. Run `save_tokenizer.py` to get the specific tokenizer of Qwen2-7B model.

3. Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

    Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

  - [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
  - [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
  - [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Simple Example

See files `examples/fireredasr_vllm_example.py`

## vLLM Engine Configuration

### Environment Variables

Set the following environment variables before starting the vLLM engine:

```bash
export RAY_MODEL_ENV_VAR="/path/to/FireRedASR-LLM-L"  # Path to merged model
export VLLM_MAX_MODEL_LEN=8192                        # Maximum model length
export VLLM_MAX_NUM_SEQS=16                           # Maximum number of sequences
export VLLM_GPU_MEMORY_UTILIZATION=0.85               # GPU memory utilization
export VLLM_QUANTIZATION=None                         # Quantization method (optional)
export DECODE_MAX_LEN=4096                            # Maximum decode length
export DECODE_MIN_LEN=0                               # Minimum decode length
export TEMPERATURE=1.0                                # Sampling temperature
export REPETITION_PENALTY=1.0                         # Repetition penalty
```

### Engine Parameters

The vLLM engine is configured with the following parameters:

```python
engine_kwargs = {
    "model": model_path,                    # Path to merged model
    "max_model_len": 8192,                  # Maximum sequence length
    "max_num_seqs": 16,                     # Maximum concurrent sequences
    "dtype": "auto",                        # auto (bf16 if supported, else fp16)
    "enforce_eager": False,                 # Use CUDA graphs
    "gpu_memory_utilization": 0.85,         # GPU memory fraction
    "trust_remote_code": True,              # Required for custom model
    "quantization": None,                   # Quantization method
    "tokenizer": tokenizer_path,            # Path to modified_tokenizer
    "mm_processor_cache_gb": 0,             # Multimodal processor cache
}
```

## Usage with Ray Data

### Example: Batch ASR Inference

```python
import os
import sys
import glob
import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Prepare audio data
audio_dir = "/path/to/audio/files"
wav_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))

data = [
    {"uttid": f"utt{i}_{os.path.basename(wav_path)}", "wav_path": wav_path}
    for i, wav_path in enumerate(wav_files)
]
ds = ray.data.from_items(data)

# Define preprocessing function
DEFAULT_SPEECH_TOKEN = "<speech>"

def vllm_preprocess_function(row):
    from datetime import datetime
    wav_array = row.get("wav_path")
    if wav_array is None:
        raise KeyError("Input batch must contain 'wav_path'")

    message = [
        {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
        {"role": "assistant", "content": ""},
    ]

    sampling_params = {
        "max_tokens": int(os.getenv("DECODE_MAX_LEN", 4096)),
        "min_tokens": int(os.getenv("DECODE_MIN_LEN", "0")),
        "temperature": max(1e-5, float(os.getenv("TEMPERATURE", "1.0"))),
        "top_p": 1.0,
        "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1.0")),
    }

    return {
        "uttid": str(row.get("uttid", "")),
        "wav_path": str(row.get("wav_path", "")),
        "duration": float(row.get("duration", 0.0)),
        "starttime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": "<|im_start|>user\n<speech>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",
        "messages": message,
        "multi_modal_data": {"audio": wav_array},
        "sampling_params": sampling_params,
    }

# Define postprocessing function
def vllm_postprocess_function(row):
    from datetime import datetime
    return {
        "uttid": row["uttid"],
        "text": row.get("generated_text", ""),
        "wav_path": row["wav_path"],
        "duration": row["duration"],
        "starttime": row["starttime"],
        "finishtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# Configure vLLM processor
model_path = os.getenv("RAY_MODEL_ENV_VAR", "")
tokenizer_path = os.path.join(model_path, "modified_tokenizer")

engine_kwargs = {
    "model": model_path,
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "16")),
    "dtype": "auto",
    "enforce_eager": False,
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
    "trust_remote_code": True,
    "quantization": os.getenv("VLLM_QUANTIZATION", None),
    "tokenizer": tokenizer_path,
    "mm_processor_cache_gb": 0,
}

processor_config = vLLMEngineProcessorConfig(
    model_source=model_path,
    engine_kwargs=engine_kwargs,
    resources_per_bundle={'GPU': 1.0},
    concurrency=1,
    batch_size=1,
)

# Build and run processor
processor = build_llm_processor(
    processor_config,
    preprocess=vllm_preprocess_function,
    postprocess=vllm_postprocess_function,
)

result_ds = processor(ds)
result_ds = result_ds.materialize()

# Print results
for result in result_ds.iter_rows():
    print(f"{result['uttid']}:\t{result['text']}")

ray.shutdown()
```

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 4096 | Maximum number of tokens to generate |
| `min_tokens` | 0 | Minimum number of tokens to generate |
| `temperature` | 1.0 | Sampling temperature (higher = more random) |
| `top_p` | 1.0 | Top-p (nucleus) sampling |
| `repetition_penalty` | 1.0 | Penalty for repeating tokens |
