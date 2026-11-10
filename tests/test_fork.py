"""Forked batch scoring must match per-branch serial scoring (tiny random models)."""

import copy

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models import qwen3, qwen3_5  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from jev_mlx.fork import ForkedPrompts, _prefill, fork_cache, score_branches  # noqa: E402
from jev_mlx.prompt import CompiledChoice  # noqa: E402

VOCAB = 97


def _attention_model():
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
        max_position_embeddings=4096, rope_theta=10000.0, head_dim=16,
        tie_word_embeddings=True,
    )
    return qwen3.Model(args)


def _hybrid_model():
    """Gated DeltaNet + attention layers, like Qwen3.5 (3 linear : 1 full)."""
    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, head_dim=16, full_attention_interval=4,
        tie_word_embeddings=False,
    )
    return qwen3_5.TextModel(args)


def _prompts():
    choices = tuple(CompiledChoice(f"id{t}", str(t), t, "") for t in (3, 5, 7, 11))
    branches = ((20, 21, 22, 23, 24, 25, 26), (30, 31), (40,), (50, 51, 52, 53))
    return ForkedPrompts(tuple(range(12, 12 + 40)), branches, (choices,) * len(branches))


def _serial(model, prepared):
    trunk = make_prompt_cache(model)
    _prefill(model, prepared.trunk, trunk, 16)
    out = []
    for branch, choices in zip(prepared.branches, prepared.choices):
        cache = copy.deepcopy(trunk)
        logits = model(mx.array([list(branch)]), cache=cache)[0, -1]
        out.append([logits[c.token_id].item() for c in choices])
    return trunk, out


@pytest.mark.parametrize("build", [_attention_model, _hybrid_model], ids=["attention", "hybrid"])
def test_forked_matches_serial(build):
    mx.random.seed(0)
    model = build()
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    prepared = _prompts()
    trunk, expected = _serial(model, prepared)
    forked = score_branches(model, trunk, prepared)
    for want, got, choices in zip(expected, forked, prepared.choices):
        assert [got[c.id] for c in choices] == pytest.approx(want, abs=1e-4)


def test_fork_cache_rows_are_independent():
    mx.random.seed(1)
    model = _hybrid_model()
    mx.eval(model.parameters())
    trunk = make_prompt_cache(model)
    _prefill(model, list(range(10)), trunk, 8)
    before = [mx.array(a) for c in trunk for a in (c.state if isinstance(c.state, (list, tuple))
                                                    else [c.state]) if a is not None]
    forked = fork_cache(trunk, 3)
    model(mx.array([[1, 2], [3, 4], [5, 6]]), cache=forked)
    after = [a for c in trunk for a in (c.state if isinstance(c.state, (list, tuple))
                                         else [c.state]) if a is not None]
    assert all(mx.array_equal(x, y).item() for x, y in zip(before, after))
    assert all(c.state[0].shape[0] == 3 for c in forked)
