"""MLX Direct Decider: JEV-inspired local decisions for Apple Silicon."""

from .engine import DecisionBackend, MLXDecisionEngine
from .session import (
    DecisionExecutionError,
    DecisionSession,
    InvalidDecisionError,
    ReplayDecisionError,
    StaleDecisionError,
)
from .types import (
    ABSTAIN_ID,
    NO_MATCH_ID,
    BackendOutput,
    Candidate,
    DecisionRequest,
    DecisionResult,
    boolean_request,
)
from .vlm import MLXVLMDecisionEngine

__version__ = "0.2.0"

__all__ = [
    "ABSTAIN_ID",
    "NO_MATCH_ID",
    "BackendOutput",
    "Candidate",
    "DecisionBackend",
    "DecisionExecutionError",
    "DecisionRequest",
    "DecisionResult",
    "DecisionSession",
    "InvalidDecisionError",
    "MLXDecisionEngine",
    "MLXVLMDecisionEngine",
    "ReplayDecisionError",
    "StaleDecisionError",
    "boolean_request",
]
