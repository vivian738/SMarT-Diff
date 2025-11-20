import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import GCNConv

from src.layers import DenseGCNConv


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 x 512 images into smaller 64 x 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 x 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 x 4 kernels and 2 x 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class GraphToConditioningAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, conditioning_channels, grid_size):
        """
        Converts graph-based features into a format suitable for ControlNet.
        """
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, conditioning_channels)
        self.grid_size = grid_size

    def forward(self, x, adj):
        # Apply GCN layers
        edge_index = adj.nonzero(as_tuple=False).t()  # Shape: (2, num_edges)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)  # Shape: (N, conditioning_channels)

        # Convert node features to grid
        feature_grid = torch.zeros(self.grid_size, self.grid_size, x.shape[-1], device=x.device)
        num_nodes = min(x.size(0), self.grid_size ** 2)
        feature_grid.view(-1, x.shape[-1])[:num_nodes] = x[:num_nodes]
        feature_grid = feature_grid.permute(2, 0, 1)  # Shape: (conditioning_channels, H, W)
        return feature_grid


class RUEEmbedding(nn.Module):
    def __init__(self, node_dim, global_dim):
        """
        RUE: Relational Universal Embedding
        Args:
            node_dim (int): Dimension of node features.
            global_dim (int): Dimension of the global graph embedding.
        """
        super().__init__()
        self.global_embedding = nn.Parameter(torch.randn(global_dim))  # Learnable global embedding
        self.node_to_global = nn.Linear(node_dim, global_dim)
        self.edge_to_global = nn.Linear(2 * node_dim, global_dim)
        self.transform = nn.Linear(global_dim, node_dim)

    def forward(self, x, adj):
        """
        Args:
            x (Tensor): Node features (B, N, F).
            edge_features (Tensor): Edge features (B, N, N, E).
            adj (Tensor): Adjacency matrix (B, N, N).
        Returns:
            global_emb: Global graph embedding (B, global_dim).
            updated_x: Updated node features (B, N, F).
        """
        B, N, F = x.size()
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, F)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, F)
        edge_features = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, 2F)
        # Mask edge features using adjacency matrix
        edge_features = edge_features * adj.unsqueeze(-1)  # Mask non-existent edges

        # Node-to-global aggregation
        node_global = torch.sum(self.node_to_global(x), dim=1)  # (B, global_dim)

        # Edge-to-global aggregation
        edge_global = torch.sum(self.edge_to_global(edge_features).sum(dim=1), dim=1)  # (B, global_dim)

        # Combine global features
        global_emb = node_global + edge_global + self.global_embedding  # (B, global_dim)

        # Return global and updated node features
        attention_weights = torch.sigmoid(torch.matmul(x, self.transform(global_emb).unsqueeze(-1)))
        updated_x = x * attention_weights  # Broadcast to nodes
        return global_emb, updated_x



class ControlNetEmbeddingWithRUE(nn.Module):


    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_hidden_channels: int,
        out_channels: int,
        global_dim: int,
        num_scales: int = 2
    ):
        super().__init__()

        self.rue = RUEEmbedding(
            node_dim=conditioning_embedding_channels,
            global_dim=global_dim
        )

        self.downscaling_convs = nn.ModuleList([
            DenseGCNConv(conditioning_embedding_channels if i == 0 else conditioning_hidden_channels,
                         conditioning_hidden_channels)
            for i in range(num_scales)
        ])

        self.num_scales = num_scales

        self.upscaling_convs = nn.ModuleList([
            DenseGCNConv(conditioning_hidden_channels, conditioning_hidden_channels)
            for _ in range(num_scales)
        ])

        self.conv_out = zero_module(nn.Conv2d(conditioning_hidden_channels, out_channels, kernel_size=3, padding=1))

        self.residual = nn.Identity()

    def forward(self, x, adj):
        global_emb, x = self.rue(x, adj)
        skip_connections = []

        # Downscaling (捕捉局部信息)
        for down_conv in self.downscaling_convs:
            x = down_conv(x, adj)
            x = F.silu(x)
            skip_connections.append(x)

        # Upscaling (融合局部和全局信息)
        for up_conv in self.upscaling_convs:
            skip = skip_connections.pop()
            x = x + skip  # 融合跳跃连接信息
            x = up_conv(x, adj)
            x = F.silu(x)

        embedding = self.conv_out(x.permute(0, 2, 1).unsqueeze(-1))

        embedding = embedding.permute(0, 2, 1, 3).squeeze(-1)

        return embedding, global_emb

