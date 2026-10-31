"""Dataset adapters registered by name."""

from .captaincook4d import CaptainCook4DAdapter
from .holoassist import HoloAssistAdapter
from .impact import ImpactAdapter
from .industreal import IndustRealAdapter

ADAPTERS = {
    "captaincook4d": CaptainCook4DAdapter,
    "holoassist": HoloAssistAdapter,
    "impact": ImpactAdapter,
    "industreal": IndustRealAdapter,
}

__all__ = [
    "ADAPTERS",
    "CaptainCook4DAdapter",
    "HoloAssistAdapter",
    "ImpactAdapter",
    "IndustRealAdapter",
]
