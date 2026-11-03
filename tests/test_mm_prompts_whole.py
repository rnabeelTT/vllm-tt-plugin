# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Multimodal prompts stay whole on a chunked-prefill model that declares
``model_capabilities['mm_prompts_whole']``."""

from types import SimpleNamespace

import pytest

import vllm  # noqa: F401  (let vLLM finish its own bootstrap first)
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_tt_plugin.config import get_tt_mm_prompts_whole_budget
from vllm_tt_plugin.platform import TTPlatform, _apply_chunked_prefill_policy
from vllm_tt_plugin.scheduler import TTScheduler


class _Model:
    pass


def _config(budget=4096, threshold=0, chunked=True):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=chunked,
            max_num_batched_tokens=budget,
            long_prefill_token_threshold=threshold,
            disable_chunked_mm_input=False,
        ),
        model_config=SimpleNamespace(max_model_len=262144),
        additional_config={},
    )


CAPS = {"supports_chunked_prefill": True, "mm_prompts_whole": True}


def test_policy_records_the_step_budget():
    cfg = _config(budget=4096)
    _apply_chunked_prefill_policy(cfg, CAPS, _Model)
    assert cfg.scheduler_config.disable_chunked_mm_input is True
    assert get_tt_mm_prompts_whole_budget(cfg) == 4096


def test_policy_budget_is_capped_by_the_long_prefill_threshold():
    cfg = _config(budget=4096, threshold=2048)
    _apply_chunked_prefill_policy(cfg, CAPS, _Model)
    assert get_tt_mm_prompts_whole_budget(cfg) == 2048


def test_policy_without_the_capability_records_nothing():
    cfg = _config()
    _apply_chunked_prefill_policy(cfg, {"supports_chunked_prefill": True}, _Model)
    assert get_tt_mm_prompts_whole_budget(cfg) is None


def _scheduler(budget, chunked=True):
    s = TTScheduler.__new__(TTScheduler)
    s.scheduler_config = SimpleNamespace(enable_chunked_prefill=chunked)
    s._mm_prompts_whole_budget = budget
    return s


def _request(prompt, total=None):
    return SimpleNamespace(num_prompt_tokens=prompt, num_tokens=total or prompt)


@pytest.fixture
def base_schedules(monkeypatch):
    """The base scheduler schedules `cut` of the offered tokens (None = all)."""
    state = {"cut": None}

    def fake(self, request, num_computed_tokens, num_new_tokens, budget, shift=0):
        n = num_new_tokens if state["cut"] is None else state["cut"]
        return [0], n, budget - 1, []

    monkeypatch.setattr(Scheduler, "_try_schedule_encoder_inputs", fake)
    return state


def test_prompt_cut_by_the_step_budget_waits(base_schedules):
    # 3000-token image prompt, 1600 tokens left in this step
    out = _scheduler(4096)._try_schedule_encoder_inputs(_request(3000), 0, 1600, 10)
    assert out == ([], 0, 10, [])


def test_prompt_cut_before_its_image_waits(base_schedules):
    base_schedules["cut"] = 4  # text before the image, image does not fit
    out = _scheduler(4096)._try_schedule_encoder_inputs(_request(3000), 0, 3000, 10)
    assert out == ([], 0, 10, [])


def test_whole_prompt_is_scheduled(base_schedules):
    out = _scheduler(4096)._try_schedule_encoder_inputs(_request(3000), 0, 3000, 10)
    assert out == ([0], 3000, 9, [])


def test_prompt_larger_than_the_budget_is_not_parked(base_schedules):
    out = _scheduler(4096)._try_schedule_encoder_inputs(_request(5000), 0, 4096, 10)
    assert out == ([0], 4096, 9, [])


def test_decode_rows_are_untouched(base_schedules):
    out = _scheduler(4096)._try_schedule_encoder_inputs(_request(3000, 3010), 3009, 1, 10)
    assert out == ([0], 1, 9, [])


def test_model_without_the_capability_keeps_chunking(base_schedules):
    out = _scheduler(None)._try_schedule_encoder_inputs(_request(3000), 0, 1600, 10)
    assert out == ([0], 1600, 9, [])


def _mm_input(n):
    return {"type": "multimodal", "prompt_token_ids": list(range(n)), "mm_placeholders": {"image": [object()]}}


def test_frontend_rejects_a_multimodal_prompt_over_the_budget(monkeypatch):
    cfg = _config()
    _apply_chunked_prefill_policy(cfg, CAPS, _Model)
    monkeypatch.setattr(TTPlatform, "_resolve_tt_admission_handle", classmethod(lambda cls: cfg))
    TTPlatform._validate_mm_prompt_whole(_mm_input(4096))
    with pytest.raises(ValueError, match="4097 tokens, more than the 4096"):
        TTPlatform._validate_mm_prompt_whole(_mm_input(4097))
    TTPlatform._validate_mm_prompt_whole({"type": "token", "prompt_token_ids": list(range(9000))})


def test_frontend_accepts_everything_without_the_capability(monkeypatch):
    cfg = _config()
    monkeypatch.setattr(TTPlatform, "_resolve_tt_admission_handle", classmethod(lambda cls: cfg))
    TTPlatform._validate_mm_prompt_whole(_mm_input(9000))
