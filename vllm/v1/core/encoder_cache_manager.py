# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

# <abs> Encoder Cache Sharing by MM Hash
#
import os
from collections import defaultdict

logger = init_logger(__name__)

# <abs> Encoder Cache Sharing by MM Hash
#
# Note that the same multi-modal item may be shared by multiple requests.
# Before the hit_cnt reaches `VLLM_EXPECT_ENCODER_CACHE_HITS` (or request
# ended), the encoder cache keeps retaining the items for future reuse at the
# cost of extra memory consumption.
#
# To decide the value, for example, we know in advance that each image will be
# used for Chinese and English prompts, so we expect each encoder cache item to
# be hit twice, i.e., `VLLM_EXPECT_ENCODER_CACHE_HITS=2`.
VLLM_EXPECT_ENCODER_CACHE_HITS = int(
    os.getenv("VLLM_EXPECT_ENCODER_CACHE_HITS", "1"))
if VLLM_EXPECT_ENCODER_CACHE_HITS != 1:
    if VLLM_EXPECT_ENCODER_CACHE_HITS <= 0:
        VLLM_EXPECT_ENCODER_CACHE_HITS = 1
    else:
        logger.warning(
            "Enabling encoder cache with expected hit count=%d! Note that upon "
            "enabling this feature, the encoder cache keeps retaining the "
            "multi-modal items for future resue at the cost of extra memory "
            "consumption.", VLLM_EXPECT_ENCODER_CACHE_HITS)
else:
    logger.warning("Using the default encoder cache!")

# </abs>


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Note that no caching is shared between requests at this time. If the same
    input is used across multiple requests, it will be reprocessed for each
    request.
    
    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    tokens from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder tokens
        num_free_slots: Current available cache capacity in encoder tokens
        cached: Mapping from request_id to set of cached input_ids for that
                request
        freed: List of (request_id, input_id) pairs that were recently freed.
               This is cleared after every call to get_freed_ids().
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        # req_id -> cached input ids
        self.cached: dict[str, set[int]] = {}

        # <abs> Encoder Cache Sharing by MM Hash
        #
        # Difference of `ref_cnt` and `hit_cnt`:
        #
        # - `ref_cnt` is increased when allocating for certain mm inputs, and
        #   decreased when freeing. It means the **sharing count** between
        #   running requests.
        # - `hit_cnt` is increased when allocating for certain mm inputs, but
        #   never decreased. However, this counter is cleared upon reaching
        #   `VLLM_EXPECT_ENCODER_CACHE_HITS` or being forced to be cleared.
        #
        # mm_hash -> cached ref_cnt
        self.mm_hash_ref: dict[str, int] = defaultdict(int)
        # mm_hash -> cached hit_cnt
        self.mm_hash_hit: dict[str, int] = defaultdict(int)
        # </abs>

        # <abs> Encoder Cache Sharing by MM Hash
        #
        # self.freed: list[tuple[str, int]] = []
        #
        # Originally, `freed` refers to a list of [req_id, input_id]. When
        # `hit_cnt` is cleared, put its mm_hash into `freed`, which will be sent
        # to model_runner to clear the cache.
        self.freed: list[str] = []

    def has_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        req_id = request.request_id

        # <abs> Encoder Cache Sharing by MM Hash
        #
        # return req_id in self.cached and input_id in self.cached[req_id]
        return (req_id in self.cached and input_id in self.cached[req_id]
                ) or self.mm_hash_hit[request.mm_hashes[input_id]] > 0

    def can_allocate(self, request: Request, input_id: int) -> bool:
        """Check if there's sufficient cache space for a multimodal input.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if there's enough free cache space to store the encoder output
            for this multimodal input
        """
        num_tokens = request.get_num_encoder_tokens(input_id)
        return num_tokens <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This method reserves cache space for storing the encoder output of
        the specified multimodal input. The actual encoder output storage
        happens in the model runner, but this method ensures the cache
        manager tracks the allocation.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Note:
            This method assumes can_allocate() returned True for the same
            request and input_id. It will reduce available cache space.
        """
        req_id = request.request_id
        if req_id not in self.cached:
            self.cached[req_id] = set()
        self.cached[req_id].add(input_id)

        # <abs> Encoder Cache Sharing by MM Hash
        #
        # self.num_free_slots -= request.get_num_encoder_tokens(input_id)
        if self.mm_hash_ref[request.mm_hashes[input_id]] == 0:
            self.num_free_slots -= request.get_num_encoder_tokens(input_id)
        self.mm_hash_ref[request.mm_hashes[input_id]] += 1
        self.mm_hash_hit[request.mm_hashes[input_id]] += 1
        # </abs>

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Args:
            request: The request to query

        Returns:
            Set of input_ids that have cached encoder outputs for this request.
            Returns empty set if no inputs are cached for this request.
        """
        return self.cached.get(request.request_id, set())

    def free_encoder_input(
        self,
        request: Request,
        input_id: int,

        # <abs> Encoder Cache Sharing by MM Hash
        #
        # ) -> None:
        #
        # "force" to free all associated cache items when the request is
        # exiting.
        force: bool = False,
    ) -> None:
        """Free cache space for a single multimodal input's encoder output.

        This method is called when:
        - The encoder output has been fully consumed by the decoder and is
          no longer needed (e.g., in vision-language models after image
          tokens are processed)
        - A request is being cancelled or aborted

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input to free from cache
        """
        req_id = request.request_id
        if req_id not in self.cached:
            return

        self.cached[req_id].discard(input_id)
        if len(self.cached[req_id]) == 0:
            del self.cached[req_id]

        # <abs> Encoder Cache Sharing by MM Hash
        #
        # self.num_free_slots += request.get_num_encoder_tokens(input_id)
        # self.freed.append((req_id, input_id))
        self.mm_hash_ref[request.mm_hashes[input_id]] -= 1
        if self.mm_hash_ref[request.mm_hashes[input_id]] == 0:
            self.num_free_slots += request.get_num_encoder_tokens(input_id)
        if force or self.mm_hash_hit[
                request.mm_hashes[input_id]] == VLLM_EXPECT_ENCODER_CACHE_HITS:
            self.freed.append(request.mm_hashes[input_id])
            del self.mm_hash_hit[request.mm_hashes[input_id]]
        # </abs>

    def free(self, request: Request) -> None:
        """Free all cached encoder outputs for a request.

        This method is typically called when a request is finished, cancelled,
        or aborted, and all its encoder outputs should be freed from cache.

        Args:
            request: The request whose encoder outputs should be freed
        """
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:

            # <abs> Encoder Cache Sharing by MM Hash
            # self.free_encoder_input(request, input_id)
            self.free_encoder_input(request, input_id, force=True)

    # <abs> Encoder Cache Sharing by MM Hash
    # def get_freed_ids(self) -> list[tuple[str, int]]:
    def get_freed_ids(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        This method returns all encoder cache entries that were freed since
        the last call to this method. It's used by the scheduler to notify
        workers about which encoder outputs can be removed from their caches.

        Returns:
            List of (request_id, input_id) tuples that were freed since the
            last call. The internal freed list is cleared after this call.
        """
        freed = self.freed
        self.freed = []
        return freed


def compute_encoder_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations.

    Args:
        model_config: Model configuration.
        scheduler_config: Scheduler configuration.
        mm_registry: Provides information about the token cost.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens 
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens 
            in the input sequence.
    """

    if not mm_registry.supports_multimodal_inputs(model_config):
        return 0, 0

    # TODO: handle encoder-decoder models once we support them.
    (
        encoder_compute_budget,
        encoder_cache_size,
    ) = _compute_encoder_budget_multimodal(
        model_config,
        scheduler_config,
        mm_registry,
    )

    return encoder_compute_budget, encoder_cache_size


def _compute_encoder_budget_multimodal(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a multimodal model.

    Args:
        model_config: Model configuration.
        scheduler_config: Scheduler configuration.
        mm_registry: Provides information about the token cost.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens 
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens 
            in the input sequence.
    """

    max_tokens_by_modality_dict = mm_registry \
        .get_max_tokens_per_item_by_nonzero_modality(model_config)

    if not max_tokens_by_modality_dict:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized.")
        return 0, 0

    _, max_tokens_per_mm_item = max(max_tokens_by_modality_dict.items(),
                                    key=lambda item: item[1])

    if (scheduler_config.disable_chunked_mm_input and max_tokens_per_mm_item
            > scheduler_config.max_num_batched_tokens):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens.")

    encoder_compute_budget = max(scheduler_config.max_num_encoder_input_tokens,
                                 max_tokens_per_mm_item)
    encoder_cache_size = max(scheduler_config.encoder_cache_size,
                             max_tokens_per_mm_item)

    return encoder_compute_budget, encoder_cache_size
