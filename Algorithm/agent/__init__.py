from .mhgnn_agent import MHGNNAgent

REGISTRY_Agents = {
    "MHGNN": MHGNNAgent
}

__all__ = [
    "MHGNNAgent",
]