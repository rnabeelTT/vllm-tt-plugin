# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Prefix caching for models with recurrent state (Gated DeltaNet / Mamba hybrids).

vLLM's prefix cache reuses paged KV blocks. A hybrid model also needs its
recurrent state at the position where the reused prefix ends, and no KV block
holds that. The model keeps a small device pool of recurrent-state snapshots
("slots"); this module is the scheduler's index over that pool:

* A snapshot is keyed by the vLLM block hash of the block that ends at the
  snapshot position. The hash chains the whole prefix (plus cache salt, LoRA
  and multimodal keys), exactly as the KV blocks are keyed.
* A KV cache hit is capped to the longest position that has a valid snapshot,
  so the model restores its state at exactly the position vLLM resumes from.
* Every scheduled prefill chunk asks the model to save snapshots at the
  positions it declares cheap: the segment boundaries its prefill crosses
  anyway and, unless the model opts out (``prompt_end``), the last block
  boundary of the prompt (the point a follow-up turn resumes from; it costs
  the model an extra prefill pass).
* A snapshot becomes usable once the step that writes it has been processed.
  Snapshots that a scheduled restore reads are pinned until that step has been
  processed. Eviction is LRU over unpinned snapshots.

A model opts in with ``model_capabilities["prefix_state_cache"] = {"slots": N,
"segment": S, "min_tokens": M, "prompt_end": bool}``. ``prompt_end`` False saves at segment boundaries only (the
model's prefill then runs exactly the passes it runs without caching; a hit resumes at a segment boundary). It then receives the prefill kwarg
``prefix_state``: one dict per row, ``{"req_id": str, "restore": (slot, pos) |
None, "saves": [(pos, slot), ...]}``. ``restore`` means "load slot ``slot``
before running from ``pos`` (== this row's start_pos)". ``saves`` means "after
the first ``pos`` tokens, copy the recurrent state into slot ``slot``".
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_tt_logger(__name__)

PREFIX_STATE_CAPABILITY = "prefix_state_cache"
_CONFIG_KEY = "_tt_prefix_state_cache"
# Directives ride on the SchedulerOutput object, like the forced-reset discard
# counts in scheduler.py, so they reach the runner with the step they belong to.
_DIRECTIVES_ATTR = "_tt_prefix_state_directives"
_STEP_ATTR = "_tt_prefix_state_step"


def store_prefix_state_policy(vllm_config: "VllmConfig", policy: dict) -> None:
    slots = int(policy.get("slots", 0))
    if slots < 1:
        raise ValueError(f"{PREFIX_STATE_CAPABILITY} needs slots >= 1, got {policy!r}")
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_CONFIG_KEY] = {
        "slots": slots,
        "segment": int(policy.get("segment", 0)),
        "min_tokens": int(policy.get("min_tokens", 0)),
        "prompt_end": bool(policy.get("prompt_end", True)),
    }


def get_prefix_state_policy(vllm_config: "VllmConfig") -> dict | None:
    additional = getattr(vllm_config, "additional_config", None) or {}
    return additional.get(_CONFIG_KEY)


def get_prefix_state_directives(
    scheduler_output: "SchedulerOutput",
) -> dict[str, dict] | None:
    return getattr(scheduler_output, _DIRECTIVES_ATTR, None)


@dataclass
class _Snapshot:
    slot: int
    pos: int
    valid: bool = False
    pins: int = 0


class PrefixStateIndex:
    """Scheduler-side index of the model's recurrent-state snapshot slots."""

    def __init__(self, policy: dict, block_size: int, hash_block_size: int):
        self.num_slots = int(policy["slots"])
        self.segment = int(policy.get("segment", 0))
        self.min_tokens = int(policy.get("min_tokens", 0))
        self.prompt_end = bool(policy.get("prompt_end", True))
        self.block_size = int(block_size)
        self.hash_block_size = int(hash_block_size)
        assert self.block_size % self.hash_block_size == 0
        self._free = list(range(self.num_slots - 1, -1, -1))
        # key -> snapshot, in LRU order (most recently used last)
        self._snaps: OrderedDict[Any, _Snapshot] = OrderedDict()
        # req_id -> key pinned for a restore planned by get_computed_blocks
        self._planned: dict[str, Any] = {}
        self._step = 0
        # step -> (keys written by the step, keys pinned by the step's restores)
        self._inflight: dict[int, tuple[list, list]] = {}
        self.stats = {"lookups": 0, "hits": 0, "hit_tokens": 0, "capped_tokens": 0, "saves": 0, "evictions": 0}

    # -------------------------------------------------------------- lookups
    def _key_at(self, block_hashes, pos: int):
        i = pos // self.hash_block_size - 1
        return block_hashes[i] if 0 <= i < len(block_hashes) else None

    def longest(self, block_hashes, max_tokens: int) -> int:
        """Longest block-aligned position <= ``max_tokens`` with a valid snapshot."""
        pos = max_tokens // self.block_size * self.block_size
        while pos > 0:
            snap = self._snaps.get(self._key_at(block_hashes, pos))
            if snap is not None and snap.valid and snap.pos == pos:
                return pos
            pos -= self.block_size
        return 0

    def cap_hit(self, block_hashes, blocks, num_tokens: int, num_uncached: int):
        """Wraps ``KVCacheCoordinator.find_longest_cache_hit``: cap a KV hit to a snapshot."""
        if num_tokens <= 0:
            return blocks, num_tokens, num_uncached
        self.stats["lookups"] += 1
        pos = self.longest(block_hashes, num_tokens)
        if pos < num_tokens:
            self.stats["capped_tokens"] += num_tokens - pos
            n_blocks = pos // self.block_size
            blocks = tuple(list(group)[:n_blocks] for group in blocks)
            return blocks, pos, 0
        return blocks, num_tokens, num_uncached

    def plan_restore(self, request: "Request", num_tokens: int) -> None:
        """After ``get_computed_blocks``: pin the snapshot the request resumes from."""
        self.unplan(request.request_id)
        if num_tokens <= 0:
            return
        key = self._key_at(request.block_hashes, num_tokens)
        snap = self._snaps.get(key)
        assert snap is not None and snap.valid and snap.pos == num_tokens, (num_tokens, snap)
        snap.pins += 1
        self._planned[request.request_id] = key

    def unplan(self, req_id: str) -> None:
        key = self._planned.pop(req_id, None)
        if key is not None:
            snap = self._snaps.get(key)
            if snap is not None:
                snap.pins -= 1

    # -------------------------------------------------------------- scheduling
    def _alloc_slot(self) -> int | None:
        if self._free:
            return self._free.pop()
        for key, snap in self._snaps.items():  # LRU first
            if snap.valid and snap.pins == 0:
                del self._snaps[key]
                self.stats["evictions"] += 1
                return snap.slot
        return None

    def _save_positions(self, start: int, end: int, prompt_len: int) -> list[int]:
        bs = self.block_size
        out = []
        if self.segment > 0:
            out += list(range(start + self.segment, end + 1, self.segment))
        if self.prompt_end and end >= prompt_len:  # the chunk that finishes the prompt: its last block boundary
            out.append(end // bs * bs)
        return sorted({q for q in out if start < q <= end and q % bs == 0 and q >= max(self.min_tokens, bs)})

    def plan_step(self, scheduler_output: "SchedulerOutput", requests: dict[str, "Request"]) -> None:
        """Attach restore / save directives for every prefill row of a scheduled step."""
        self._step += 1
        written, pinned = [], []
        directives: dict[str, dict] = {}
        for req_id, n_sched in scheduler_output.num_scheduled_tokens.items():
            request = requests.get(req_id)
            if request is None:
                continue
            end = request.num_computed_tokens  # already advanced by _update_after_schedule
            start = end - int(n_sched)
            if start >= request.num_prompt_tokens and start >= request.num_tokens - 1:
                continue  # a decode row
            restore = None
            key = self._planned.pop(req_id, None)
            if key is not None:
                snap = self._snaps.get(key)
                if snap is not None and snap.pos == start:
                    restore = (snap.slot, snap.pos)
                    self._snaps.move_to_end(key)
                    pinned.append(key)  # stays pinned until this step is processed
                    self.stats["hits"] += 1
                    self.stats["hit_tokens"] += start
                elif snap is not None:
                    snap.pins -= 1
            saves = []
            # a row that starts at or past the prompt end is a decode step, also when it schedules several tokens (a
            # speculative / multi-token step fails the decode test above): the model saves no state there, so a save
            # planned on it would be marked valid unwritten and a later hit would restore stale state
            if not request.mm_features and start < request.num_prompt_tokens:
                # latest first: when slots run short, the prompt-end snapshot (where a follow-up turn resumes) wins
                for q in reversed(self._save_positions(start, end, request.num_prompt_tokens)):
                    k = self._key_at(request.block_hashes, q)
                    if k is None or k in self._snaps:
                        continue
                    slot = self._alloc_slot()
                    if slot is None:
                        break
                    self._snaps[k] = _Snapshot(slot=slot, pos=q)
                    written.append(k)
                    saves.append((q, slot))
                    self.stats["saves"] += 1
                saves.sort()
            directives[req_id] = {"req_id": req_id, "restore": restore, "saves": saves}
        if directives:
            setattr(scheduler_output, _DIRECTIVES_ATTR, directives)
            setattr(scheduler_output, _STEP_ATTR, self._step)
            self._inflight[self._step] = (written, pinned)
            if written or pinned:
                logger.info(
                    "prefix state: step restores %s, saves %s; totals %s, %d/%d slots used",
                    [d["restore"] for d in directives.values() if d["restore"]],
                    [q for d in directives.values() for q, _ in d["saves"]],
                    self.stats,
                    len(self._snaps),
                    self.num_slots,
                )

    def step_processed(self, scheduler_output: "SchedulerOutput") -> None:
        step = getattr(scheduler_output, _STEP_ATTR, None)
        if step is None:
            return
        written, pinned = self._inflight.pop(step, ([], []))
        for key in written:
            snap = self._snaps.get(key)
            if snap is not None:
                snap.valid = True
        for key in pinned:
            snap = self._snaps.get(key)
            if snap is not None:
                snap.pins -= 1

    def reset(self) -> None:
        """``reset_prefix_cache``: drop every snapshot no in-flight step still uses."""
        for key, snap in list(self._snaps.items()):
            if snap.pins == 0 and snap.valid:
                del self._snaps[key]
                self._free.append(snap.slot)


def install_prefix_state_index(scheduler) -> PrefixStateIndex | None:
    """Build the index for a TT scheduler and hook it into the KV cache manager."""
    policy = get_prefix_state_policy(scheduler.vllm_config)
    manager = scheduler.kv_cache_manager
    if policy is None or not manager.enable_caching:
        return None
    coordinator = manager.coordinator
    if len(coordinator.single_type_managers) != 1:
        raise ValueError("prefix_state_cache supports one KV cache group (uniform full attention)")
    index = PrefixStateIndex(policy, scheduler.block_size, scheduler.hash_block_size)

    find_longest = coordinator.find_longest_cache_hit

    def find_longest_cache_hit(block_hashes, max_cache_hit_length):
        blocks, n, uncached = find_longest(block_hashes, max_cache_hit_length)
        return index.cap_hit(block_hashes, blocks, n, uncached)

    coordinator.find_longest_cache_hit = find_longest_cache_hit

    get_computed = manager.get_computed_blocks

    def get_computed_blocks(request):
        if request.mm_features:  # image prompts are prefilled whole
            index.unplan(request.request_id)
            return manager.empty_kv_cache_blocks, 0, 0
        blocks, n, boundary = get_computed(request)
        index.plan_restore(request, n)
        return blocks, n, boundary

    manager.get_computed_blocks = get_computed_blocks
    logger.info(
        "Recurrent-state prefix caching: %d snapshot slots, segment %d, min %d tokens, block %d, prompt end %s",
        index.num_slots,
        index.segment,
        index.min_tokens,
        index.block_size,
        index.prompt_end,
    )
    return index
