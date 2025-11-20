import math
import torch
from torch import nn as nn
import torch.nn.functional as F


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.elementwise_affine:
            return self.weight * torch.tanh(self.alpha * x) + self.bias
        else:
            return torch.tanh(self.alpha * x)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}"


class Attention_trans(torch.nn.Module):

    def __init__(self, in_dim,
                num_heads=8,
                qkv_bias=False,
                qk_norm=False,
                attn_drop=0.1,
                proj_drop=0.1,
                norm_layer=nn.LayerNorm,):
        super(Attention_trans, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME
        assert self.fast_attn, "scaled_dot_product_attention Not implemented"

        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, node_mask): #conditions=None,

        B, N, D = x.shape

        # B, head, N, head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, head, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = (node_mask[:, None, :, None] & node_mask[:, None, None, :]).expand(-1, self.num_heads, N, N)
        attn_mask = attn_mask.clone()
        attn_mask[attn_mask.sum(-1) == 0] = True

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# -------- Layer of ScoreNetwork_trans --------
class SELayer(torch.nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):

        super(SELayer, self).__init__()

        self.dropout = 0.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = Attention_trans(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, **block_kwargs)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=self.dropout,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, flags): #, conditions=None
        """

        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * modulate(self.norm1(self.attn(x, node_mask=flags)), shift_msa, scale_msa)
        x = x + gate_mlp * modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)

        return x


class OutLayer(nn.Module):
    # Structure Output Layer
    def __init__(self, max_n_nodes, hidden_size, atom_type, bond_type):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        final_size = atom_type + max_n_nodes * bond_type
        self.xedecoder = Mlp(in_features=hidden_size,
                             out_features=final_size, drop=0)

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, c, node_mask):
        x_all = self.xedecoder(x)
        B, N, D = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x_all = modulate(self.norm_final(x_all), shift, scale)

        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out

        bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)
        bond_out = e_in[..., :1] + bond_out

        ##### standardize adj_out
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))

        return atom_out, bond_out


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)
        self.fc2 = linear_layer(hidden_features, out_features)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
