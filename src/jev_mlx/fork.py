"""Experimental: score several questions about one state in a single batched pass.

Idea borrowed from Flide's "exact state forking" (arXiv 2610.42148), adapted to
MLX without custom kernels:

1. Prefill the shared trunk (system prompt + state) once at batch size 1.
2. Repeat every layer cache along the batch axis, one row per question. This
   works for attention KV caches and for recurrent (Gated DeltaNet) state alike.
3. Right-pad the question branches to one length and run them as one batch.
   Every layer is causal, so padding after a branch never changes the hidden
   state at that branch's last real token. RoPE offsets equal the trunk length
   for every row, matching the standalone prompt.
4. Read the final hidden state of each branch's last real token and apply the
   output head only to those rows (selective readout).

Each branch is tokenized as part of its full standalone prompt, so trunk plus
branch is token-for-token the same sequence a single-question call would see.
Results agree with serial scoring up to floating-point differences from batched
matrix shapes; ``benchmarks/fork_bench.py`` measures that difference.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from .prompt import POLICY, compile_choices
from .token_codes import continuation_choices
from .types import DecisionRequest

_OUTPUT = "Reply with only the option code, with no whitespace, explanation or punctuation."


@dataclass(frozen=True)
class ForkedPrompts:
    trunk: tuple[int, ...]
    branches: tuple[tuple[int, ...], ...]
    choices: tuple[tuple, ...]  # per branch: CompiledChoice tuple


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _content(request: DecisionRequest, choices) -> tuple[str, str]:
    """State-first layout: the shared state precedes everything question-specific."""
    options = [{"code": c.code, "id": c.id, "meaning": c.description} for c in choices]
    if request.kind == "boolean":
        values = {c.id: c.value for c in request.candidates}
        for option in options:
            if option["id"] in values:
                option["value"] = values[option["id"]]
    shared = "CURRENT_STATE (visible arrays are in displayed order):\n" + _json(request.state)
    content = (
        shared
        + "\nQUESTION:\n"
        + _json({"kind": request.kind, "question": request.question})
        + "\nCHOICES (codes are labels, not visible positions):\n"
        + _json(options)
        + "\nUSER_UTTERANCE:\n"
        + _json(request.utterance)
    )
    return shared, content


def prepare_forked(tokenizer, requests: list[DecisionRequest]) -> ForkedPrompts:
    """Tokenize each question as a full standalone prompt, then split off the trunk."""
    if not requests:
        raise ValueError("at least one question is required")
    state = requests[0].state
    if any(r.state != state for r in requests):
        raise ValueError("all forked questions must share the same state")
    full, choices, shared_prefixes = [], [], []
    for request in requests:
        compiled = compile_choices(tokenizer, request)
        shared, content = _content(request, compiled)
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": POLICY + "\n" + _OUTPUT},
             {"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(text, str) or text.count(content) != 1:
            raise ValueError("Unsupported chat template")
        tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
        choices.append(continuation_choices(tokenizer, text, tokens, compiled))
        boundary = text.index(content) + len(shared)
        shared_prefixes.append(len(tokenizer.encode(text[:boundary], add_special_tokens=False)))
        full.append(tokens)
    # Trunk: longest common token prefix, never past the shared-state boundary
    # (a BPE merge may cross it) and always leaving at least one branch token.
    trunk_len = min(min(shared_prefixes), min(len(t) for t in full) - 1)
    for i in range(trunk_len):
        if any(t[i] != full[0][i] for t in full):
            trunk_len = i
            break
    trunk = full[0][:trunk_len]
    return ForkedPrompts(trunk, tuple(t[trunk_len:] for t in full), tuple(choices))


def _text_parts(model):
    """Return (backbone, head) for MLX-LM text models, or the text part of a wrapper."""
    lm = getattr(model, "language_model", model)
    backbone = lm.model
    head = getattr(lm, "lm_head", None) or backbone.embed_tokens.as_linear
    return backbone, head


def _prefill(model, tokens, cache, step):
    import mlx.core as mx

    for start in range(0, len(tokens), step):
        model(mx.array([tokens[start : start + step]]), cache=cache)
        mx.eval([c.state for c in cache])


def fork_cache(cache, n: int):
    """Copy a batch-1 cache into n independent rows (KV and recurrent state)."""
    import mlx.core as mx

    forked = []
    for layer in cache:
        clone = copy.copy(layer)
        state = layer.state
        if isinstance(state, (list, tuple)):
            clone.state = [None if a is None else mx.repeat(a, n, axis=0) for a in state]
        else:
            clone.state = mx.repeat(state, n, axis=0)
        forked.append(clone)
    return forked


def score_branches(model, trunk_cache, prepared: ForkedPrompts, *, pad_id: int = 0):
    """Score all branches in one batched forward pass; returns per-branch raw scores."""
    import mlx.core as mx

    n = len(prepared.branches)
    lengths = [len(b) for b in prepared.branches]
    width = max(lengths)
    batch = mx.array([list(b) + [pad_id] * (width - len(b)) for b in prepared.branches])
    cache = fork_cache(trunk_cache, n)
    backbone, head = _text_parts(model)
    hidden = backbone(batch, cache=cache)  # (n, width, d)
    last = hidden[mx.arange(n), mx.array(lengths) - 1]  # (n, d)
    logits = head(last)  # (n, vocab): head runs on n rows only
    picked = [
        logits[i, mx.array([c.token_id for c in choices])].astype(mx.float32)
        for i, choices in enumerate(prepared.choices)
    ]
    mx.eval(picked)
    return [
        {c.id: float(v) for c, v in zip(choices, row.tolist())}
        for choices, row in zip(prepared.choices, picked)
    ]


def score_forked(backend, requests: list[DecisionRequest], *, trunk_cache=None, prepared=None):
    """Prefill the shared state once (unless given) and score every question in one pass.

    Returns ``(scores, trunk_cache)`` so callers can keep the trunk for later questions.
    """
    from mlx_lm.models.cache import make_prompt_cache

    prepared = prepared or prepare_forked(backend.tokenizer, requests)
    if trunk_cache is None:
        trunk_cache = make_prompt_cache(backend.model)
        _prefill(backend.model, prepared.trunk, trunk_cache, backend.prefill_step_size)
    return score_branches(backend.model, trunk_cache, prepared), trunk_cache


def score_serial(backend, requests: list[DecisionRequest], *, trunk_cache=None, prepared=None):
    """Reference: same prompts, trunk prefilled once, each branch scored at batch size 1."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    prepared = prepared or prepare_forked(backend.tokenizer, requests)
    if trunk_cache is None:
        trunk_cache = make_prompt_cache(backend.model)
        _prefill(backend.model, prepared.trunk, trunk_cache, backend.prefill_step_size)
    results = []
    for branch, choices in zip(prepared.branches, prepared.choices):
        cache = copy.deepcopy(trunk_cache)
        _prefill(backend.model, branch[:-1], cache, backend.prefill_step_size)
        logits = backend.model(mx.array([branch[-1:]]), cache=cache)[0, -1, :]
        picked = logits[mx.array([c.token_id for c in choices])].astype(mx.float32)
        mx.eval(picked)
        results.append({c.id: float(v) for c, v in zip(choices, picked.tolist())})
    return results, trunk_cache
