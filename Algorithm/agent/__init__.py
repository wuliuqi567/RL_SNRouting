from .mhgnn_agent import MHGNNAgent
from .gat_agent import GATAgent 
from .mpnn_agent import MPNNAgent
from .ddqn_agent import DDQNAgent
from .mhgnnedge_agent import MHGNNEDGEAgent
REGISTRY_Agents = {
    "MHGNN": MHGNNAgent,
    "GAT": GATAgent,
    "MPNN": MPNNAgent,
    "DDQN": DDQNAgent,
    "MHGNNKGE": MHGNNEDGEAgent,
}

__all__ = [
    "MHGNNAgent", "GATAgent", "MPNNAgent", "DDQNAgent", "MHGNNEDGEAgent"
]