from torch import nn
import torch

from src.layers import DenseGCNConv, MLP


class ClassifierWithBias(nn.Module):
    def __init__(self, max_feat_num, depth, nhid, y_dim=3, global_emb_dim=3):
        super().__init__()
        self.nfeat = max_feat_num
        self.nhid = nhid
        self.layers = torch.nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.activation = torch.tanh

        # Condition (y + global_emb) MLP
        self.condition_mlp = nn.Sequential(
            nn.Linear(y_dim, nhid), # + global_emb_dim
            nn.SiLU(),
            nn.Linear(nhid, nhid),
        )

        # Fusion MLP for combining node, edge, and condition features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(nhid * 3 + self.nfeat + nhid, nhid),
            nn.SiLU(),
            nn.Linear(nhid, nhid),
        )

        # Final classifier
        self.classifier = nn.Linear(nhid, y_dim)

    def forward(self, x, adj, y): #, global_emb
        """
        Forward function for the classifier.

        Args:
            x (torch.Tensor): Node features of size (B, N, node_feat_dim)
            adj (torch.Tensor): Adjacency matrix of size (B, N, N)
            y (torch.Tensor): Conditioning vector of size (B, y_dim)
            global_emb (torch.Tensor): Global embedding vector of size (B, global_emb_dim)

        Returns:
            logits (torch.Tensor): Classification logits of size (B, y_dim)
        """
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        # Condition embedding (y + global_emb)
        avg_x = torch.cat(x_list, dim=-1).mean(dim=1)
        # condition_emb = self.condition_mlp(torch.cat([y, global_emb], dim=-1))  # (B, hidden_dim)
        condition_emb = self.condition_mlp(y)

        # Combine features
        combined_features = torch.cat([avg_x, condition_emb], dim=-1)  # (B, 3*hidden_dim + nfeat)
        fused_features = self.fusion_mlp(combined_features)  # (B, hidden_dim)

        # Classification
        logits = self.classifier(fused_features)  # (B, y_dim)
        return logits
