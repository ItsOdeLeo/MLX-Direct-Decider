"""Public loaded-weight construction and cancellation; no MLX/Metal required."""
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from jev_mlx.backends.mlx_lm import MLXLMBackend


class LoadedModelTest(unittest.TestCase):
    def setUp(self):
        self.mx = ModuleType('mlx.core')
        self.mx.array = lambda value: value
        self.mx.eval = Mock()
        self.mx.synchronize = Mock()
        self.mx.clear_cache = Mock()
        self.loader = Mock(side_effect=AssertionError('must never reload weights'))
        mlx = ModuleType('mlx')
        mlx.core = self.mx
        mlx_lm = ModuleType('mlx_lm')
        mlx_lm.load = self.loader
        cache = ModuleType('mlx_lm.models.cache')
        cache.LRUPromptCache = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        cache.make_prompt_cache = Mock()
        self.cache = cache
        self.modules = patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': self.mx,
             'mlx_lm': mlx_lm, 'mlx_lm.models.cache': cache})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.model = Mock()
        self.tokenizer = SimpleNamespace(encode=lambda text: [1])

    def test_keeps_exact_objects_and_allocates_only_owned_bounded_cache(self):
        identity = {'name': 'external', 'extra': {'revision': 'one'}}
        backend = MLXLMBackend.from_loaded_model(self.model, self.tokenizer,
                    identity=identity, cache_bytes=1024**3)
        self.assertIs(backend.model, self.model)
        self.assertIs(backend.tokenizer, self.tokenizer)
        self.assertIs(backend.mx, self.mx)
        self.assertEqual(backend.load_ms, 0)
        self.assertEqual(backend._cache.max_bytes, 1024**3)
        self.assertEqual(backend._cache.max_size, 16)
        identity['extra']['revision'] = 'two'
        self.assertEqual(backend.identity['extra']['revision'], 'one')
        self.loader.assert_not_called()
        self.model.assert_not_called()
        self.cache.make_prompt_cache.assert_not_called()

    def test_owned_caches_are_separate_for_same_weights(self):
        first = MLXLMBackend.from_loaded_model(self.model, self.tokenizer, identity={})
        second = MLXLMBackend.from_loaded_model(self.model, self.tokenizer, identity={})
        self.assertIs(first.model, second.model)
        self.assertIsNot(first._cache, second._cache)

    def test_cancel_before_prefill_does_not_call_model(self):
        backend = MLXLMBackend.from_loaded_model(self.model, self.tokenizer,
                  identity={}, should_cancel=lambda: True)
        with self.assertRaises(InterruptedError):
            backend._prefill([1, 2, 3], [SimpleNamespace(state=[])])
        self.model.assert_not_called()

    def test_cancel_during_chunk_waits_until_state_materialized(self):
        cancelled = False
        def compute(*args, **kwargs):
            nonlocal cancelled
            cancelled = True
        self.model.side_effect = compute
        backend = MLXLMBackend.from_loaded_model(self.model, self.tokenizer,
                  identity={}, prefill_step_size=1, should_cancel=lambda: cancelled)
        with self.assertRaises(InterruptedError):
            backend._prefill([1, 2, 3], [SimpleNamespace(state=[])])
        self.assertEqual(self.model.call_count, 1)
        self.mx.eval.assert_called_once()

    def test_invalid_inputs_fail_without_loader(self):
        for options in ({'cache_bytes': 0}, {'cache_entries': 1},
                        {'max_prompt_tokens': True}, {'prefill_step_size': 0},
                        {'should_cancel': 'not a callback'}):
            with self.assertRaises(ValueError):
                MLXLMBackend.from_loaded_model(self.model, self.tokenizer,
                                              identity={}, **options)
        self.loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
