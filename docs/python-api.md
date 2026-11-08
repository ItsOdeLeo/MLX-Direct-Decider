# Python API

Install MLX Direct Decider using the [quick start](../README.md#quick-start), then set
`MLX_DIRECT_MODEL` to a verified local checkpoint. See [model compatibility](models.md).

## Make a decision

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

Results report `candidate_id`, `status`, `raw_selected_id`, `raw_scores`,
`scores`, `margin`, `state_version`, `model`, `timing`, `cache`,
`request_id`, and `selected_value`. Only `status == "selected"` carries an
executable candidate ID. `no_match` means no applicable choice; `abstain`
means clarification or more evidence is needed. Sessions can also return `stale`.

`raw_scores` are candidate-token logits. `scores` are a softmax restricted to
the supplied candidates plus `__no_match__` and `__abstain__`; **they are not
calibrated probabilities of correctness**. `margin` is the raw top-minus-runner-up
logit gap. The default rejection threshold is 0.0, which rejects exact ties.
Select any other threshold using development data and report its limitations.

Boolean choices preserve typed values, including `False`:

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

## Keep decisions aligned with application state

`DecisionSession` snapshots state before inference. Updates during inference
mark the result stale. Execution checks the current version again, verifies
that the result was issued intact by this session, and consumes it once.

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

Call `update_state` for every relevant state or candidate change. Callbacks run
under the session's state lock and may update it reentrantly. Callback failure
consumes authorization too, preventing accidental replay of a partial effect.
These checks prevent stale and replayed execution; they do not establish whether
the model understood the user.


See the [architecture](architecture.md) for validation limits, cache boundaries,
result semantics, and execution guarantees.

## Share caller-owned loaded weights

`MLXLMBackend.from_loaded_model(model, tokenizer, identity=..., should_cancel=...)`
constructs a backend without calling a loader or copying model weights. Wrap it
with `MLXDecisionEngine(backend=backend)`. The model must return MLX logits arrays
and expose the MLX-LM cache contract; VLM callers may supply a text logits view.
The application owns model lifetime and must serialize generation and scoring.
The backend owns a separate bounded LRU cache. Discard it when replacing the model.
`should_cancel` optionally stops work between completed prefill chunks and prevents
returning a decision after cancellation; it cannot interrupt in-flight GPU work.

```python
from jev_mlx.backends.mlx_lm import MLXLMBackend

backend = MLXLMBackend.from_loaded_model(
    loaded_model, loaded_tokenizer, identity={"name": "my-loaded-model"},
    cache_bytes=1024**3,
)
engine = MLXDecisionEngine(backend=backend)
backend.prewarm(request, cache_key="desktop")
result = engine.decide(request, cache_key="desktop")
```
