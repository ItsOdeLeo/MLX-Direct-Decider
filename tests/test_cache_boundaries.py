"""No Metal: exercise reuse boundaries against an adversarial nearest-cache API."""

import copy
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from jev_mlx.backends import mlx_lm as backend_module
from jev_mlx.backends.mlx_lm import MLXLMBackend
from jev_mlx.prompt import PreparedPrompt


class NearestSnapshotDouble:
    """Simulate relevant public LRU behavior, including trimming and prefix removal.

    No model computation is represented. This test double deliberately offers
    partial snapshots that the adapter must reject instead of trusting them.
    """

    def __init__(self, max_size=8, max_bytes=100):
        self.max_size, self.max_bytes = max_size, max_bytes
        self.entries = OrderedDict()
        self.fetches = []
        self.partial_offers = 0

    def __len__(self):
        return len(self.entries)

    @property
    def nbytes(self):
        return sum(len(cache[0].tokens) for cache in self.entries.values())

    def fetch_nearest_cache(self, namespace, tokens):
        self.fetches.append((namespace, tuple(tokens)))
        key = namespace, tuple(tokens)
        if key in self.entries:
            return copy.deepcopy(self.entries[key]), []
        best, common = None, 0
        for (saved_namespace, saved_tokens), cache in self.entries.items():
            if saved_namespace != namespace:
                continue
            prefix = 0
            for old, new in zip(saved_tokens, tokens):
                if old != new:
                    break
                prefix += 1
            if prefix > common:
                best, common = cache, prefix
        if best is not None and common:
            common = min(common, len(tokens) - 1)
            offered = copy.deepcopy(best)
            offered[0].tokens = offered[0].tokens[:common]
            self.partial_offers += 1
            return offered, tokens[common:]
        return None, list(tokens)

    def insert_cache(self, namespace, tokens, cache, *, cache_type):
        key = namespace, tuple(tokens)
        self.entries[key] = cache
        # Official trimmable caches remove saved shorter prefixes in the same
        # namespace; the adapter must protect its system snapshot from this.
        for old_namespace, old_tokens in list(self.entries):
            if (
                old_namespace == namespace
                and len(old_tokens) < len(tokens)
                and tuple(tokens[: len(old_tokens)]) == old_tokens
            ):
                del self.entries[(old_namespace, old_tokens)]
        while len(self.entries) > self.max_size or self.nbytes > self.max_bytes:
            self.entries.popitem(last=False)


def backend(monkeypatch, *, max_size=8, max_bytes=100):
    instance = MLXLMBackend.__new__(MLXLMBackend)
    instance.max_prompt_tokens = 4096
    instance.prompt_layout = "question_first"
    instance.model = instance.tokenizer = None
    instance._cache = NearestSnapshotDouble(max_size=max_size, max_bytes=max_bytes)
    instance._make_cache = lambda model: [SimpleNamespace(tokens=[])]
    instance.prefill_calls = []

    def prefill(tokens, cache):
        if tokens:
            instance.prefill_calls.append(tuple(tokens))
            cache[0].tokens.extend(tokens)

    instance._prefill = prefill
    monkeypatch.setattr(backend_module, "prepare_prompt", lambda tokenizer, request, **_: request)
    return instance


def prompt(page=(3, 4, 5), system=(1, 2), shared=()):
    stable = (*system, *shared, *page)
    return PreparedPrompt(
        tokens=(*stable, 9),
        system_tokens=system,
        state_tokens=stable,
        choices=(),
        shared_tokens=(*system, *shared) if shared else system,
    )


def test_changed_page_rejects_trimmed_middle_and_reuses_complete_system(monkeypatch):
    instance = backend(monkeypatch)
    instance._prepare(prompt(), "code", "session", True)
    instance.prefill_calls.clear()
    changed = prompt(page=(3, 7, 8))
    _, cache, suffix, info = instance._prepare(changed, "code", "session", True)
    assert instance._cache.partial_offers == 1
    assert info["scope"] == "prefix"
    assert info["reused_tokens"] == len(changed.system_tokens) == 2
    assert instance.prefill_calls == [(3, 7, 8)]
    assert cache[0].tokens == list(changed.state_tokens)
    assert suffix == [9]
    assert any(namespace[-1] == "system" for namespace, _ in instance._cache.entries)


def test_exact_page_hit_avoids_system_lookup_and_prefill(monkeypatch):
    instance = backend(monkeypatch)
    value = prompt()
    instance._prepare(value, "code", "session", True)
    instance.prefill_calls.clear()
    instance._cache.fetches.clear()
    _, cache, suffix, info = instance._prepare(value, "code", "session", True)
    assert info["scope"] == "state"
    assert info["reused_tokens"] == len(value.state_tokens)
    assert instance.prefill_calls == []
    assert instance._cache.fetches == [(("session", "code", "page"), value.state_tokens)]
    assert cache[0].tokens == list(value.state_tokens)
    assert suffix == [9]


def test_partial_system_snapshot_is_also_rejected(monkeypatch):
    instance = backend(monkeypatch)
    instance._prepare(prompt(), "code", "session", True)
    instance.prefill_calls.clear()
    changed = prompt(system=(1, 6))
    _, cache, _, info = instance._prepare(changed, "code", "session", True)
    assert instance._cache.partial_offers == 2
    assert info["scope"] == "cold"
    assert info["reused_tokens"] == 0
    assert instance.prefill_calls == [(1, 6), (3, 4, 5)]
    assert cache[0].tokens == list(changed.state_tokens)


def test_shorter_page_does_not_accept_a_trimmed_longer_snapshot(monkeypatch):
    instance = backend(monkeypatch)
    instance._prepare(prompt(), "code", "session", True)
    instance.prefill_calls.clear()
    shorter = prompt(page=(3, 4))
    _, cache, _, info = instance._prepare(shorter, "code", "session", True)
    # Official 0.31.3 trim leaves at least one suffix token, even when the
    # requested page is a complete prefix of an existing longer snapshot.
    assert instance._cache.partial_offers == 1
    assert info["scope"] == "prefix"
    assert info["reused_tokens"] == len(shorter.system_tokens)
    assert instance.prefill_calls == [(3, 4)]
    assert cache[0].tokens == list(shorter.state_tokens)


def test_missing_system_after_page_mismatch_recomputes_from_cold(monkeypatch):
    instance = backend(monkeypatch)
    instance._prepare(prompt(), "code", "session", True)
    instance._cache.entries = OrderedDict(
        (key, cache) for key, cache in instance._cache.entries.items() if key[0][-1] == "page"
    )
    instance.prefill_calls.clear()
    _, _, _, info = instance._prepare(prompt(page=(3, 7, 8)), "code", "session", True)
    assert info["scope"] == "cold"
    assert instance.prefill_calls == [(1, 2), (3, 7, 8)]


def test_namespaces_share_one_bounded_store_and_cold_calls_bypass_it(monkeypatch):
    instance = backend(monkeypatch, max_size=2, max_bytes=7)
    original_store = instance._cache
    for index in range(10):
        instance._prepare(prompt(page=(3, index + 10, 5)), "code", f"session-{index}", True)
        assert instance._cache is original_store
        assert len(original_store) <= 2
        assert original_store.nbytes <= 7
    original_store.fetches.clear()
    before = copy.deepcopy(original_store.entries)
    _, _, _, info = instance._prepare(prompt(), "code", "uncached", False)
    assert info["scope"] == "cold"
    assert not original_store.fetches
    assert list(original_store.entries) == list(before)


def test_state_first_question_reuses_shared_state_snapshot(monkeypatch):
    instance = backend(monkeypatch, max_bytes=1000)
    first = prompt(shared=(20, 21, 22), page=(30, 31))
    instance._prepare(first, "code", "session", True)
    assert instance.prefill_calls == [(1, 2), (20, 21, 22), (30, 31)]
    instance.prefill_calls.clear()
    second = prompt(shared=(20, 21, 22), page=(40, 41))  # new question, same state
    _, cache, suffix, info = instance._prepare(second, "code", "session", True)
    assert info["scope"] == "shared_state"
    assert info["reused_tokens"] == len(second.shared_tokens) == 5
    assert instance.prefill_calls == [(40, 41)]
    assert cache[0].tokens == list(second.state_tokens)
    assert suffix == [9]
    # A changed state falls back to the system snapshot, not a trimmed state.
    instance.prefill_calls.clear()
    changed = prompt(shared=(20, 99, 22), page=(40, 41))
    _, _, _, info = instance._prepare(changed, "code", "session", True)
    assert info["scope"] == "prefix"
    assert instance.prefill_calls == [(20, 99, 22), (40, 41)]


def test_state_first_uncached_uses_the_same_prefill_segments(monkeypatch):
    instance = backend(monkeypatch)
    value = prompt(shared=(20, 21, 22), page=(30, 31))
    _, _, _, info = instance._prepare(value, "code", "session", False)
    assert info["scope"] == "cold"
    assert instance.prefill_calls == [(1, 2), (20, 21, 22), (30, 31)]


class SizedSnapshot:
    """Minimal cache layer for the official LRU: byte size plus prefilled tokens."""

    def __init__(self):
        self.tokens = []

    @property
    def nbytes(self):
        return 10 * len(self.tokens)

    def is_trimmable(self):
        return False


def test_official_lru_keeps_shared_state_while_questions_churn(monkeypatch):
    cache_module = pytest.importorskip("mlx_lm.models.cache")
    instance = backend(monkeypatch)
    # Room for the system, shared-state and about two question snapshots only.
    instance._cache = cache_module.LRUPromptCache(max_size=16, max_bytes=10 * 40)
    instance._make_cache = lambda model: [SizedSnapshot()]
    shared = tuple(range(20, 30))
    scopes = []
    for question in range(8):
        value = prompt(shared=shared, page=(100 + question, 200 + question))
        instance.prefill_calls.clear()
        _, _, _, info = instance._prepare(value, "code", "session", True)
        scopes.append(info["scope"])
        if question:
            assert instance.prefill_calls == [(100 + question, 200 + question)]
    assert scopes == ["cold"] + ["shared_state"] * 7
