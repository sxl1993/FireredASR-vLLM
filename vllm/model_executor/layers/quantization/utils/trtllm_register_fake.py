# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from functools import lru_cache

import torch
from loguru import logger

# Add a fake kernel for `quantize_e4m3_per_tensor``, so as to support
# `torch.compile`.
logger.warning(
    "Registering a fake kernel for "
    "`tensorrt_llm::quantize_e4m3_per_tensor` to support "
    "`torch.compile` ...", )


@torch.library.register_fake("tensorrt_llm::quantize_e4m3_per_tensor")
def _(input: torch.Tensor):
    return torch.empty_like(input).to(torch.float8_e4m3fn), input.new_empty(
        [1 for _ in range(input.dim())])


@torch.library.register_fake("trtllm::cublas_scaled_mm")
def _(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias,
    out_dtype,
    userbuffers_id=False,
):
    shape = [i for i in mat_a.shape]
    shape[-1] = mat_b.shape[-1]
    ret = mat_a.new_empty(shape, dtype=out_dtype)
    return ret


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


@lru_cache(maxsize=1)
def get_sm_version():
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


@torch.library.register_fake("trtllm::fp8_quantize_1x128")
def _(input: torch.Tensor):
    pad_m = pad_up(input.shape[0], 4)
    blocked_n = (input.shape[1] + 127) // 128
    if get_sm_version() >= 100:
        sz = (blocked_n, input.shape[0])
    else:
        sz = (pad_up(pad_m * blocked_n * 4, 128) // 4, )
    return torch.empty_like(input, dtype=torch.float8_e4m3fn), input.new_empty(
        sz, dtype=torch.float)


@torch.library.register_fake("trtllm::fp8_block_scaling_gemm")
def _(a, b, a_scale, b_scale):
    m = a.shape[0]
    n = b.shape[0]
    return a.new_empty((m, n), dtype=torch.bfloat16)
