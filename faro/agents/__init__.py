from faro.agents.base import (
    Agent,
    IntraExperimentAgent,
    InterPhaseAgent,
    PreExperimentAgent,
)
from faro.agents.bo_dose_response import DoseResponseBO
from faro.agents.bo_dose_response_mt import MultiTaskDoseResponseBO
from faro.agents.bo_oscillation import OscillationBO
from faro.agents.composed import ComposedAgent
from faro.agents.fov_finder import FOVFinderAgent

# BoTorch-based agents (lazy import — only available when botorch is installed)
try:
    from faro.agents.bo_botorch import BOptBoTorch
    from faro.agents.bo_botorch_oscillation import OscillationBOBoTorch
except ImportError:
    pass

__all__ = [
    "Agent",
    "BOptBoTorch",
    "ComposedAgent",
    "DoseResponseBO",
    "FOVFinderAgent",
    "IntraExperimentAgent",
    "InterPhaseAgent",
    "MultiTaskDoseResponseBO",
    "OscillationBO",
    "OscillationBOBoTorch",
    "PreExperimentAgent",
]
