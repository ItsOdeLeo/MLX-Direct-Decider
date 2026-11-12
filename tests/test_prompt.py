import pytest

from jev_mlx import Candidate, DecisionRequest
from jev_mlx.prompt import compile_choices, prepare_prompt


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"
        )


def request(utterance="Close it", state=None):
    return DecisionRequest(
        state or {"window": "notes"}, utterance, (Candidate("close.notes", "Close notes"),)
    )


def test_verified_codes_keep_business_ids():
    choices = compile_choices(CharacterTokenizer(), request())
    assert choices[0].id == "close.notes"
    assert len({c.token_id for c in choices}) == 3
    assert choices[-2].id == "__no_match__"
    assert choices[-1].id == "__abstain__"


def test_full_candidate_bound_has_codes_for_both_reserved_choices():
    candidates = tuple(Candidate(f"item-{index}", f"Open item {index}") for index in range(64))
    value = DecisionRequest({}, "Open item 63", candidates)
    prompt = prepare_prompt(CharacterTokenizer(), value)
    assert len(prompt.choices) == 66
    assert len({choice.token_id for choice in prompt.choices}) == 66
    assert len({choice.code for choice in prompt.choices}) == 66
    assert [choice.id for choice in prompt.choices[:64]] == [c.id for c in candidates]
    assert [choice.description for choice in prompt.choices[:64]] == [c.description for c in candidates]
    assert [choice.id for choice in prompt.choices[-2:]] == ["__no_match__", "__abstain__"]


def test_snapshot_boundaries_exclude_new_utterance_and_page_changes():
    tokenizer = CharacterTokenizer()
    a = prepare_prompt(tokenizer, request())
    b = prepare_prompt(tokenizer, request("Leave it open"))
    c = prepare_prompt(tokenizer, request(state={"window": "library"}))
    assert a.state_tokens == b.state_tokens
    assert a.tokens != b.tokens
    assert a.system_tokens == c.system_tokens
    assert a.state_tokens != c.state_tokens
    assert a.tokens[: len(a.state_tokens)] == a.state_tokens


def test_tokenizer_collisions_rejected():
    class CollisionTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            return [1]

        def decode(self, tokens):
            return "0"

    with pytest.raises(ValueError, match="insufficient"):
        compile_choices(CollisionTokenizer(), request())


def test_code_merging_into_prompt_is_rejected():
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            if text.endswith("<assistant>0"):
                return [999]
            return super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="continuation"):
        prepare_prompt(MergingTokenizer(), request())


def test_standalone_and_continuation_token_ids_can_differ():
    class BoundaryTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            if text in ("0", "1", "2"):
                return [1000 + ord(text)]
            return super().encode(text, **kwargs)

        def decode(self, tokens):
            return super().decode([t - 1000 if t >= 1000 else t for t in tokens])

    tokenizer = BoundaryTokenizer()
    prompt = prepare_prompt(tokenizer, request())
    assert [c.code for c in prompt.choices] == ["0", "1", "2"]
    assert [c.token_id for c in prompt.choices] == [ord("0"), ord("1"), ord("2")]
    assert tokenizer.encode("0") == [1000 + ord("0")]


def test_candidate_order_changes_prompt_and_mapping():
    a = Candidate("a", "Open A")
    b = Candidate("b", "Open B")
    first = prepare_prompt(CharacterTokenizer(), DecisionRequest({}, "Open A", (a, b)))
    second = prepare_prompt(CharacterTokenizer(), DecisionRequest({}, "Open A", (b, a)))
    assert first.state_tokens != second.state_tokens
    assert first.choices[0].id != second.choices[0].id


def test_json_code_control_preserves_semantic_user_content_and_candidate_mapping():
    tokenizer = CharacterTokenizer()
    prompts = {
        mode: prepare_prompt(tokenizer, request(), mode=mode)
        for mode in ("code", "json", "json_code")
    }
    texts = {mode: tokenizer.decode(prompt.tokens) for mode, prompt in prompts.items()}
    assert len({text.split("<user>", 1)[1] for text in texts.values()}) == 1
    assert prompts["code"].choices == prompts["json"].choices == prompts["json_code"].choices
    assert '{"choice":"<option code>"}' in texts["json_code"]
    assert '{"candidate_id":"<id>"}' in texts["json"]
    assert "Reply with only the option code," in texts["code"]


def boolean(question, state=None):
    return DecisionRequest.boolean(
        state=state or {"player": "paused", "volume": 3},
        utterance="Is it paused?",
        question=question,
    )


def test_default_layout_is_question_first_and_unchanged():
    tokenizer = CharacterTokenizer()
    value = boolean("Is the player paused?")
    default = prepare_prompt(tokenizer, value)
    explicit = prepare_prompt(tokenizer, value, layout="question_first")
    assert default == explicit
    text = tokenizer.decode(default.tokens)
    assert text.index("QUESTION:") < text.index("CURRENT_STATE")
    # question_first has no shared state boundary beyond the system prompt.
    assert default.shared_tokens == default.system_tokens


def test_state_first_shares_system_and_state_across_questions():
    tokenizer = CharacterTokenizer()
    a = prepare_prompt(tokenizer, boolean("Is the player paused?"), layout="state_first")
    b = prepare_prompt(tokenizer, boolean("Is the volume above 5?"), layout="state_first")
    assert a.shared_tokens == b.shared_tokens
    assert len(a.system_tokens) < len(a.shared_tokens) < len(a.state_tokens)
    assert a.tokens[: len(a.shared_tokens)] == a.shared_tokens
    shared_text = tokenizer.decode(a.shared_tokens)
    assert "CURRENT_STATE" in shared_text and "QUESTION" not in shared_text
    assert a.state_tokens != b.state_tokens
    # Same text, reordered: the state block precedes the question block.
    text = tokenizer.decode(a.tokens)
    assert text.index("CURRENT_STATE") < text.index("QUESTION:") < text.index("CHOICES")
    other_state = prepare_prompt(
        tokenizer, boolean("Is the player paused?", {"player": "playing"}), layout="state_first"
    )
    assert other_state.shared_tokens != a.shared_tokens


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="layout"):
        prepare_prompt(CharacterTokenizer(), request(), layout="state_last")
