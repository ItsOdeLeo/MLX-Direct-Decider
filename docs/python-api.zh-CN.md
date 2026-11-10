# Python API

[返回项目首页](../README.zh-CN.md)

按照[快速开始](../README.zh-CN.md#quick-start)安装 MLX Direct Decider，然后将 `MLX_DIRECT_MODEL` 设置为已验证的本地检查点路径。参见[模型兼容性](models.zh-CN.md)。

## 做出决策

```python
import os
from mlx_direct_decider import Candidate, DecisionRequest, MLXDecisionEngine

engine = MLXDecisionEngine(os.environ["MLX_DIRECT_MODEL"])
request = DecisionRequest(
    state={"view": "library", "visible_titles": ["Amber Maps", "Cloud Songs"]},
    utterance="Play the first one",
    candidates=(
        Candidate("play.amber", "Play the visible course Amber Maps"),
        Candidate("play.cloud", "Play the visible course Cloud Songs"),
        Candidate("close.library", "Close the course library"),
    ),
    state_version=3,
)
result = engine.decide(request)
print(result.to_dict())
```

结果包含 `candidate_id`、`status`、`raw_selected_id`、`raw_scores`、`scores`、`margin`、`state_version`、`model`、`timing`、`cache`、`request_id` 和 `selected_value`。只有 `status == "selected"` 时才带有可执行的候选 ID。`no_match` 表示没有适用的选项；`abstain` 表示需要澄清或更多证据。会话还可能返回 `stale`。

`raw_scores` 是候选 token 的 logits。`scores` 是仅在所提供的候选以及 `__no_match__`、`__abstain__` 之间计算的 softmax；**它们不是经过校准的正确概率**。`margin` 是最高与次高原始 logit 的差值。默认拒绝阈值为 0.0，会拒绝完全相等的最高分。其他阈值应使用开发数据选择，并报告其限制。

布尔候选会保留带类型的值，包括 `False`：

```python
question = DecisionRequest.boolean(
    state={"player_status": "paused"},
    utterance="Is playback paused?",
    question="Is the player currently paused?",
)
answer = engine.decide(question)
if answer.status == "selected":
    print(answer.selected_value)  # A Python bool, not a string.
```

## 让决策与应用状态保持一致

`DecisionSession` 在推理前保存状态快照。推理期间发生的更新会将结果标记为过期。执行时会再次检查当前版本，验证结果完整且由本会话发出，并且只允许消费一次。

```python
from mlx_direct_decider import DecisionSession

session = DecisionSession(
    engine,
    state={"view": "notes"},
    candidates=(Candidate("close.notes", "Close the open notes window"),),
)
session.prewarm()  # Optional: computes prefix state, never a final answer.
decision = session.decide("Close it")

def apply_action(candidate):
    # Apply your application action here, then record its state atomically.
    version = session.update_state({"view": "desktop"}, candidates=())
    return {"executed": candidate.id, "state_version": version}

if decision.status == "selected":
    print(session.execute(decision, apply_action))
```

每次与决策有效性相关的状态或候选发生变化时，都要调用 `update_state`。回调在会话状态锁内运行，并且可以重入更新状态。即使回调失败，授权也会被消费，避免意外重复执行已经产生部分效果的操作。这些检查防止过期执行与重复执行，但不能证明模型理解了用户。

有关验证上限、缓存边界、结果语义和执行保障，参见[架构说明](architecture.zh-CN.md)。

## 共享调用方已加载的权重

`MLXLMBackend.from_loaded_model(model, tokenizer, identity=..., should_cancel=...)` 直接用已加载的模型构建后端，不调用加载器，也不复制权重。用 `MLXDecisionEngine(backend=backend)` 包装后使用。模型须返回 MLX logits 数组，并遵循 MLX-LM 缓存约定；VLM 调用方可以传入只输出文本 logits 的视图。模型的生命周期由应用负责，生成与打分必须由应用串行化。后端自带一个独立、有上限的 LRU 缓存；更换模型时请丢弃该后端。`should_cancel` 是可选的：它在每个预填充分块完成后检查，取消后不会返回决策；它无法中断正在 GPU 上执行的计算。

```python
from mlx_direct_decider.backends.mlx_lm import MLXLMBackend

backend = MLXLMBackend.from_loaded_model(
    loaded_model, loaded_tokenizer, identity={"name": "my-loaded-model"},
    cache_bytes=1024**3,
)
engine = MLXDecisionEngine(backend=backend)
backend.prewarm(request, cache_key="desktop")
result = engine.decide(request, cache_key="desktop")
```
