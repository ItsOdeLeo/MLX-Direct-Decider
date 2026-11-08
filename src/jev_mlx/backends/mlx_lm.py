"""Single-question scoring using unmodified official MLX-LM models and caches."""

import copy
import hashlib
import json
import threading
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from ..prompt import prepare_prompt
from ..types import ABSTAIN_ID as ABSTAIN
from ..types import NO_MATCH_ID as NO_MATCH
from ..types import BackendOutput, DecisionRequest


class MLXLMBackend:
    """Serial local inference. No remote code, downloads, training, or generated actions.

    The official output head computes final-position logits. Restricting those logits
    to token codes is a classification rule, not a calibrated correctness probability.
    """

    def __init__(
        self,
        model: str,
        *,
        prefill_step_size: int = 256,
        max_prompt_tokens: int = 4096,
        cache_bytes: int = 512 * 1024**2,
        cache_entries: int = 16,
    ):
        if not isinstance(prefill_step_size, int) or prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        if max_prompt_tokens < 64 or cache_bytes < 1 or cache_entries < 2:
            raise ValueError("Invalid prompt/cache limits")
        path = Path(model).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("model must be an existing local MLX model directory")
        started = time.perf_counter()
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
        except ImportError as exc:
            raise RuntimeError(
                "Install mlx-direct-decider[mlx] on Apple Silicon to use this backend"
            ) from exc
        self.mx = mx
        self.model, self.tokenizer, config = load(
            str(path),
            tokenizer_config={"trust_remote_code": False},
            return_config=True,
        )
        mx.synchronize()
        self.load_ms = (time.perf_counter() - started) * 1000
        self.prefill_step_size = prefill_step_size
        self.max_prompt_tokens = max_prompt_tokens
        self._lock = threading.RLock()
        self._should_cancel = None
        self._make_cache = make_prompt_cache
        self._cache = LRUPromptCache(max_size=cache_entries, max_bytes=cache_bytes)
        quantization = config.get("quantization", config.get("quantization_config", {}))
        self.identity = {
            "name": path.name,
            "model_type": config.get("model_type"),
            "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
            if (path / "tokenizer.json").exists()
            else None,
            "quantization": {
                "default_bits": quantization.get("bits"),
                "group_size": quantization.get("group_size"),
                "mode": quantization.get("mode"),
                "layer_bits_counts": dict(
                    Counter(
                        str(v.get("bits")) for v in quantization.values() if isinstance(v, dict)
                    )
                ),
                "sha256": hashlib.sha256(
                    json.dumps(quantization, sort_keys=True).encode()
                ).hexdigest(),
            },
            "dependencies": {
                p: version(p) for p in ("mlx", "mlx-lm", "transformers", "tokenizers")
            },
            "cache_types": sorted({type(c).__name__ for c in make_prompt_cache(self.model)}),
        }

    @classmethod
    def from_loaded_model(
        cls,
        model,
        tokenizer,
        *,
        identity: dict,
        prefill_step_size: int = 256,
        max_prompt_tokens: int = 4096,
        cache_bytes: int = 512 * 1024**2,
        cache_entries: int = 16,
        should_cancel=None,
    ):
        """Share an already-loaded text model without loading or copying its weights.

        The model must implement the MLX-LM callable/cache contract and return an
        array of logits. A caller using a VLM may pass a text-only logits view.
        The caller owns model lifetime and serializes other users of these weights;
        this backend owns a separate bounded prompt cache. ``should_cancel`` is an
        optional callable checked at completed prefill boundaries and before scoring.
        It never interrupts an in-flight Metal operation or mutates saved snapshots.
        """
        if not callable(model) or not callable(getattr(tokenizer, "encode", None)):
            raise ValueError("Expected a callable loaded model and a text tokenizer")
        if not isinstance(identity, dict):
            raise ValueError("identity must describe the caller-owned model")
        if isinstance(prefill_step_size, bool) or not isinstance(prefill_step_size, int) \
                or prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in
               (max_prompt_tokens, cache_bytes, cache_entries)) \
                or max_prompt_tokens < 64 or cache_bytes < 1 or cache_entries < 2:
            raise ValueError("Invalid prompt/cache limits")
        if should_cancel is not None and not callable(should_cancel):
            raise ValueError("should_cancel must be callable")
        import mlx.core as mx
        from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

        backend = cls.__new__(cls)
        backend.mx = mx
        backend.model, backend.tokenizer = model, tokenizer
        backend.load_ms = 0.0
        backend.prefill_step_size = prefill_step_size
        backend.max_prompt_tokens = max_prompt_tokens
        backend._lock = threading.RLock()
        backend._should_cancel = should_cancel
        backend._make_cache = make_prompt_cache
        backend._cache = LRUPromptCache(max_size=cache_entries, max_bytes=cache_bytes)
        backend.identity = copy.deepcopy(identity)
        return backend

    def _check_cancelled(self):
        if self._should_cancel is not None and self._should_cancel():
            raise InterruptedError("Decision inference cancelled")

    def clear_cache(self):
        with self._lock:
            self._cache.trim_to(n_sequences=0)
            self.mx.clear_cache()

    def _prefill(self, tokens, cache):
        for start in range(0, len(tokens), self.prefill_step_size):
            self._check_cancelled()
            chunk = tokens[start : start + self.prefill_step_size]
            self.model(self.mx.array([chunk]), cache=cache)
            # The unused head is lazy. Evaluate all recurrent AND KV states.
            self.mx.eval([c.state for c in cache])
            self._check_cancelled()

    def _prepare(self, request, mode, cache_key, use_cache):
        prompt = prepare_prompt(self.tokenizer, request, mode=mode)
        if len(prompt.tokens) > self.max_prompt_tokens:
            raise ValueError(f"Prompt exceeds {self.max_prompt_tokens} tokens")
        namespace = (cache_key or "default", mode)
        system_namespace = (*namespace, "system")
        page_namespace = (*namespace, "page")
        cache, reused = None, 0
        if use_cache:
            page_cache, suffix = self._cache.fetch_nearest_cache(
                page_namespace, list(prompt.state_tokens)
            )
            if page_cache is not None and not suffix:
                cache, reused = page_cache, len(prompt.state_tokens)
            else:
                page_cache = None
                # Official nearest lookup may trim attention caches at an
                # arbitrary common prefix. Reuse only complete saved boundaries
                # so changed pages use the same prefill segments as fresh calls.
                system_cache, suffix = self._cache.fetch_nearest_cache(
                    system_namespace, list(prompt.system_tokens)
                )
                if system_cache is not None and not suffix:
                    cache, reused = system_cache, len(prompt.system_tokens)
        if cache is None:
            cache = self._make_cache(self.model)
        # Separate namespaces also prevent the official LRU from removing the
        # shorter system snapshot when inserting a trimmable page snapshot.
        # Both namespaces share the same entry and byte limits.
        if reused < len(prompt.system_tokens):
            self._prefill(prompt.tokens[reused : len(prompt.system_tokens)], cache)
            if use_cache:
                self._cache.insert_cache(
                    system_namespace,
                    list(prompt.system_tokens),
                    copy.deepcopy(cache),
                    cache_type="system",
                )
            cursor = len(prompt.system_tokens)
        else:
            cursor = reused
        self._prefill(prompt.tokens[cursor : len(prompt.state_tokens)], cache)
        if use_cache and reused < len(prompt.state_tokens):
            self._cache.insert_cache(
                page_namespace, list(prompt.state_tokens), copy.deepcopy(cache), cache_type="user"
            )
        info = {
            "hit": reused > 0,
            "scope": "state"
            if reused == len(prompt.state_tokens)
            else ("prefix" if reused else "cold"),
            "prompt_tokens": len(prompt.tokens),
            "reused_tokens": reused,
            "prefill_tokens": len(prompt.tokens) - reused,
            "system_tokens": len(prompt.system_tokens),
            "state_tokens": len(prompt.state_tokens),
            "bytes": self._cache.nbytes,
            "entries": len(self._cache),
        }
        return prompt, cache, list(prompt.tokens[len(prompt.state_tokens) :]), info

    def prewarm(self, request: DecisionRequest, *, cache_key=None):
        """Compute only the stable prefix; never precompute or cache a final answer."""
        with self._lock:
            self._check_cancelled()
            start = time.perf_counter()
            _, _, _, info = self._prepare(request, "code", cache_key, True)
            self.mx.synchronize()
            self._check_cancelled()
            return {"cache": info, "total_ms": (time.perf_counter() - start) * 1000}

    def score(self, request: DecisionRequest, *, cache_key=None, use_cache=True) -> BackendOutput:
        started = time.perf_counter()
        with self._lock:
            self._check_cancelled()
            inference_start = time.perf_counter()
            prompt, cache, suffix, info = self._prepare(request, "code", cache_key, use_cache)
            self._prefill(suffix[:-1], cache)
            self._check_cancelled()
            logits = self.model(self.mx.array([suffix[-1:]]), cache=cache)[0, -1, :]
            chosen = logits[self.mx.array([c.token_id for c in prompt.choices])].astype(
                self.mx.float32
            )
            self.mx.eval(chosen, [c.state for c in cache])
            self.mx.synchronize()
            self._check_cancelled()
            values = chosen.tolist()
            timing = self._timing(started, inference_start)
            return BackendOutput(
                raw_scores={c.id: float(v) for c, v in zip(prompt.choices, values)},
                model=copy.deepcopy(self.identity),
                timing=timing,
                cache=info,
            )

    def _timing(self, started, inference_start):
        end = time.perf_counter()
        return {
            "total_ms": (end - started) * 1000,
            "inference_ms": (end - inference_start) * 1000,
            "queue_ms": (inference_start - started) * 1000,
            "load_ms": self.load_ms,
            "peak_memory_bytes": self.mx.get_peak_memory(),
            "active_memory_bytes": self.mx.get_active_memory(),
        }

    def generate_baseline(
        self,
        request: DecisionRequest,
        *,
        mode="code",
        cache_key=None,
        use_cache=True,
        max_tokens=96,
    ):
        """Official unconstrained greedy generation, timed through all scheduled work.

        Code baseline uses exactly the scoring prompt, unrestricted vocabulary and
        max_tokens=1. JSON baseline uses the same semantic policy/state but requests
        an actual structured business ID. The supplementary json_code mode emits
        a JSON object containing the option code. Invalid/truncated output is a failure.
        """
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        started = time.perf_counter()
        with self._lock:
            inference_start = time.perf_counter()
            prompt, cache, suffix, info = self._prepare(request, mode, cache_key, use_cache)
            chunks = list(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    suffix,
                    prompt_cache=cache,
                    max_tokens=1 if mode == "code" else max_tokens,
                    sampler=make_sampler(temp=0.0),
                    prefill_step_size=self.prefill_step_size,
                )
            )
            self.mx.eval([c.state for c in cache])
            self.mx.synchronize()  # include MLX-LM's possible next-step prefetch
            raw_text = "".join(c.text for c in chunks)
            choice_id = None
            if mode == "code":
                choice_id = {c.code: c.id for c in prompt.choices}.get(raw_text.strip())
            elif mode == "json_code":
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict) and set(parsed) == {"choice"}:
                        value = parsed["choice"]
                        if isinstance(value, str):
                            choice_id = {c.code: c.id for c in prompt.choices}.get(value)
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict) and set(parsed) == {"candidate_id"}:
                        value = parsed["candidate_id"]
                        if isinstance(value, str) and value in {c.id for c in prompt.choices}:
                            choice_id = value
                except (ValueError, TypeError):
                    pass
            status = {NO_MATCH: "no_match", ABSTAIN: "abstain", None: "invalid"}.get(
                choice_id, "selected"
            )
            return {
                "candidate_id": choice_id if status == "selected" else None,
                "raw_selected_id": choice_id,
                "status": status,
                "schema_valid": choice_id is not None,
                "raw_text": raw_text,
                "timing": self._timing(started, inference_start),
                "cache": info,
                "model": copy.deepcopy(self.identity),
                "generation_tokens": chunks[-1].generation_tokens if chunks else 0,
                "finish_reason": chunks[-1].finish_reason if chunks else "empty",
            }
