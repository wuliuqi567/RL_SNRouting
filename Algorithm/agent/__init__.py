from .mhgnn_agent import MHGNNAgent
from .gat_agent import GATAgent 
from .mpnn_agent import MPNNAgent
from .ddqn_agent import DDQNAgent
REGISTRY_Agents = {
    "MHGNN": MHGNNAgent,
    "GAT": GATAgent,
    "MPNN": MPNNAgent,
    "DDQN": DDQNAgent,
}

__all__ = [
    "MHGNNAgent", "GATAgent", "MPNNAgent", "DDQNAgent"
]