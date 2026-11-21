# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""PrefixStateIndex: snapshot planning, hit capping, pinning and LRU eviction (no device)."""

import types

from vllm_tt_plugin.prefix_state import PrefixStateIndex, get_prefix_state_directives

B = 64


def req(rid, hashes, prompt, computed, mm=None):
    return types.SimpleNamespace(
        request_id=rid,
        block_hashes=hashes,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        num_computed_tokens=computed,
        mm_features=mm or [],
    )


def out(sched):
    return types.SimpleNamespace(num_scheduled_tokens=sched)


def blocks(n):
    return (list(range(n)),)


def test_prefix_state_index():
    idx = PrefixStateIndex({"slots": 2, "segment": 2048, "min_tokens": 1024}, B, B)
    hA = [("A", i) for i in range(5000 // B)]
    A = req("A", hA, 5000, 5000)
    so = out({"A": 5000})
    idx.plan_step(so, {"A": A})
    d = get_prefix_state_directives(so)["A"]
    assert [q for q, _ in d["saves"]] == [4096, 4992], d  # 2 slots: the two latest, prompt end included
    # before processing: no hit
    assert idx.longest(hA, 4992) == 0
    idx.step_processed(so)
    assert idx.longest(hA, 4992) == 4992 and idx.longest(hA, 4500) == 4096 and idx.longest(hA, 4000) == 0
    # B shares A's first 78 blocks, then diverges
    hB = hA[:78] + [("B", i) for i in range(78, 6000 // B)]
    Bq = req("B", hB, 6000, 0)
    bl, n, unc = idx.cap_hit(hB, blocks(80), 80 * B, 0)  # pretend the KV hit was 80 blocks (5120)
    assert n == 4992 and len(bl[0]) == 78, (n, len(bl[0]))
    idx.plan_restore(Bq, n)
    Bq.num_computed_tokens = 6000
    so = out({"B": 6000 - 4992})
    idx.plan_step(so, {"B": Bq})
    d = get_prefix_state_directives(so)["B"]
    assert d["restore"][1] == 4992
    # B's saves: segment boundary 4992+2048 > end, prompt end 5952 -> needs a slot: both slots valid, A@4992 pinned -> evict A@4096
    assert [q for q, _ in d["saves"]] == [5952], d
    assert idx.longest(hA, 4500) == 0  # A@4096 evicted
    idx.step_processed(so)
    assert idx.longest(hB, 5999) == 5952
    assert all(s.pins == 0 for s in idx._snaps.values())
    # image request: no saves
    C = req("C", [("C", i) for i in range(40)], 2600, 2600, mm=[1])
    so = out({"C": 2600})
    idx.plan_step(so, {"C": C})
    assert get_prefix_state_directives(so)["C"]["saves"] == []
    # decode rows produce no directives
    A.num_tokens = 5001; A.num_computed_tokens = 5001
    so = out({"A": 1})
    idx.plan_step(so, {"A": A})
    assert get_prefix_state_directives(so) is None
    # short prompt below min_tokens: no saves
    D = req("D", [("D", i) for i in range(10)], 700, 700)
    so = out({"D": 700}); idx.plan_step(so, {"D": D})
    assert get_prefix_state_directives(so)["D"]["saves"] == []
    # unplan on abort releases the pin
    idx.plan_restore(req("E", hB, 6000, 0), 4992)
    assert sum(s.pins for s in idx._snaps.values()) == 1
    idx.unplan("E")
    assert sum(s.pins for s in idx._snaps.values()) == 0


def test_prefix_state_segment_only():
    idx = PrefixStateIndex({"slots": 8, "segment": 2048, "min_tokens": 1024, "prompt_end": False}, B, B)
    a = req("A", [("A", i) for i in range(5000 // B)], 5000, 5000)
    so = out({"A": 5000})
    idx.plan_step(so, {"A": a})
    assert [q for q, _ in get_prefix_state_directives(so)["A"]["saves"]] == [2048, 4096]
    b = req("B", [("B", i) for i in range(4096 // B)], 4096, 4096)  # ends on a segment boundary: saved, no extra pass
    so = out({"B": 4096})
    idx.plan_step(so, {"B": b})
    assert [q for q, _ in get_prefix_state_directives(so)["B"]["saves"]] == [2048, 4096]


def test_prefix_state_prompt_end_aligned_before_last():
    # Inkling: one snapshot per prompt at its last chunk (2048) boundary before the last prompt token
    pol = {"slots": 4, "segment": 0, "min_tokens": 2048, "prompt_end": True, "prompt_end_align": 2048,
           "prompt_end_before_last": True}
    idx = PrefixStateIndex(pol, B, B)
    a = req("A", [("A", i) for i in range(5000 // B)], 5000, 5000)
    so = out({"A": 5000})
    idx.plan_step(so, {"A": a})
    assert [q for q, _ in get_prefix_state_directives(so)["A"]["saves"]] == [4096]
    b = req("B", [("B", i) for i in range(4097 // B)], 4097, 4097)  # last token at 4096: the boundary before it is 4096
    so = out({"B": 4097})
    idx.plan_step(so, {"B": b})
    assert [q for q, _ in get_prefix_state_directives(so)["B"]["saves"]] == [4096]
    c = req("C", [("C", i) for i in range(4096 // B)], 4096, 4096)  # last token at 4095: 2048, not 4096
    so = out({"C": 4096})
    idx.plan_step(so, {"C": c})
    assert [q for q, _ in get_prefix_state_directives(so)["C"]["saves"]] == [2048]
    d = req("D", [("D", i) for i in range(2048 // B)], 2048, 2048)  # 2047 tokens before the last: below min_tokens
    so = out({"D": 2048})
    idx.plan_step(so, {"D": d})
    assert get_prefix_state_directives(so)["D"]["saves"] == []
    try:
        PrefixStateIndex({**pol, "prompt_end_align": 100}, B, B)
        raise AssertionError("an alignment that is not a block multiple must be refused")
    except ValueError:
        pass


def test_prefix_state_no_saves_past_the_prompt():
    # Inkling decode (MTP shim): after an n-token step the scheduler schedules those n emitted tokens (n <= 3) while
    # num_tokens has already moved on, so the decode-row test alone does not catch the row. A save planned there was never written
    # by the model (it snapshots in prefill only); the next turn restored the empty slot and the engine died
    # (multi-turn chat whose answer crossed a 2048 boundary: 14105-token prompt, save at 14336, next turn 14435).
    pol = {"slots": 4, "segment": 0, "min_tokens": 2048, "prompt_end": True, "prompt_end_align": 2048,
           "prompt_end_before_last": True}
    idx = PrefixStateIndex(pol, B, B)
    a = req("A", [("A", i) for i in range(14400 // B)], 14105, 14105)
    so = out({"A": 14105})
    idx.plan_step(so, {"A": a})
    assert [q for q, _ in get_prefix_state_directives(so)["A"]["saves"]] == [12288]
    idx.step_processed(so)
    for n, computed, ntok in ((3, 14200, 14203), (3, 14337, 14340), (2, 14338, 14341), (1, 14339, 14342)):
        a.num_computed_tokens, a.num_tokens = computed, ntok  # rows [computed - n, computed): 14334..14337 straddles 14336
        so = out({"A": n})
        idx.plan_step(so, {"A": a})
        d = get_prefix_state_directives(so)
        assert d is None or (d["A"]["saves"] == [] and d["A"]["restore"] is None), (computed, d)
        idx.step_processed(so)
    assert idx.longest(a.block_hashes, 14336) == 12288  # nothing claims a snapshot at 14336
