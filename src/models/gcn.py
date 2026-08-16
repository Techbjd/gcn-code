"""Graph Convolutional Network classifiers for molecular property prediction."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

_POOLERS = {
    "sum": gnn.global_add_pool,
    "mean": gnn.global_mean_pool,
    "max": gnn.global_max_pool,
}


class GCNClassifier(nn.Module):
    """GCN / GIN graph classifier producing (logits, embedding) per graph."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        pooling: str = "sum",
        model_type: str = "gcn",
    ):
        super().__init__()
        if model_type not in {"gcn", "gin"}:
            raise ValueError(f"model_type must be 'gcn' or 'gin', got {model_type!r}")
        if pooling not in _POOLERS:
            raise ValueError(f"pooling must be one of {sorted(_POOLERS)}, got {pooling!r}")
        self.model_type = model_type
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        convs, bns = [], []
        in_dim = num_node_features
        for _ in range(num_layers):
            if model_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                convs.append(gnn.GINConv(nn=mlp, eps=0.0, train_eps=False))
            else:
                convs.append(gnn.GCNConv(in_dim, hidden_dim))
            bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = _POOLERS[self.pooling](x, batch)
        logits = self.head(embedding)
        return logits, embedding


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model state dict, creating the parent directory if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load a state dict into model and return it."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model