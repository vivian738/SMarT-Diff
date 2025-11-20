import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from src.layers import DenseGCNConv, MLP, LayerNorm, ClusterContinuousEmbedder, TimestepEmbedder, CategoricalEmbedder
from src.transformer import SELayer, OutLayer
from utils.graph_utils import mask_x, mask_adjs, pow_tensor, node_feature_to_matrix
from src.attention import AttentionLayer
from src.controlnet import ControlNetEmbeddingWithRUE


class BaselineNetworkLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

        super(BaselineNetworkLayer, self).__init__
        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2*conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
                            use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)
        
    def forward(self, x, adj, flags):
    
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:,_,:,:])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)) , flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid  = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

            elif _==self.num_layers-1:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)   

    def forward(self, x, adj, flags): #, sx=None, sa=None

        # condition = self.conditionnet(sx, sa)
        # condition = condition.unsqueeze(1).expand(-1, x.shape[1], -1)  # B x N x D
        # x = torch.cat([x, condition], dim=-1)

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):

            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score


class ScoreNetworkA(BaselineNetwork):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(ScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                                            c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')
        
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        # self.conditionnet = ControlNetEmbeddingWithRUE(self.nfeat, self.nhid, self.nhid, 3)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
            elif _==self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init  #fdim
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)  

    def forward(self, x, adj, flags): #, sx, sa

        # conditions, _ = self.conditionnet(sx, sa)  #(bs ,n, nhid)

        adjc = pow_tensor(adj, self.c_init)  #(bs ,c_init, n, n)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            if _ == 0:
                x, adjc = self.layers[_](x, adjc, flags) #, conditions
            else:
                x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)  #bs, 38, 38, fdim
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)
        
        self.mask = self.mask.to(score.device)  #1, 38, 38
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score
    


class ScoreNetworkX(torch.nn.Module):

    def __init__(self, max_feat_num, depth, nhid):

        super(ScoreNetworkX, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        # self.conditionnet = ControlNetEmbeddingWithRUE(self.nfeat, self.nhid, self.nfeat, 3)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid)) # self.neat * 2
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags): #, sx, sa

        # condition, global_emb = self.conditionnet(sx, sa)  # B x N x D
        # x = torch.cat([x, condition], dim=-1) # B x N x 2D

        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x #, global_emb


class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()

        self.depth = depth
        self.c_init = c_init

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x


class ScoreTransformer(torch.nn.Module):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers,
                 c_final, num_heads=4, cond_dropout=0.01):
        super().__init__()
        self.n_feat = max_feat_num
        self.c_final = c_final
        self.x_embedder = nn.Linear(self.n_feat + max_node_num * c_final, nhid, bias=False)

        self.t_embedder = TimestepEmbedder(nhid)

        self.y_embedding_list = torch.nn.ModuleList()
        self.y_embedding_list.append(CategoricalEmbedder(self.n_feat + max_node_num * c_final, nhid, cond_dropout)) #sx
        self.y_embedding_list.append(ClusterContinuousEmbedder(2, nhid, cond_dropout))

        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SELayer(nhid, num_heads))

        self.out_layer = OutLayer(
                        max_n_nodes=max_node_num,
                        hidden_size=nhid,
                        atom_type=self.n_feat,
                        bond_type=1 #c_final
                    )
        self.mask = torch.ones([max_node_num, max_node_num]) - torch.eye(max_node_num)
        self.mask.unsqueeze_(0)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.layers:
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.out_layer.adaLN_modulation[0], 0)

    def forward(self, x, adj, flags, t, sx, sa, y):
        """

            :param x:  B x N x F_i
            :param adj: B x C x N x N
        """
        adjc = pow_tensor(adj, self.c_final)
        sac = pow_tensor(sa, self.c_final)
        x_in, e_in, sx_in, sa_in = x, adjc.permute(0,2,3,1), sx, sac.permute(0,2,3,1)
        bs, n, _ = x.size()
        x = torch.cat([x, e_in.reshape(bs, n, -1)], dim=-1)
        sx = torch.cat([sx, sa_in.reshape(bs, n, -1)], dim=-1)
        x = self.x_embedder(x)
        c1 = self.y_embedding_list[0](sx, self.training)
        c2 = self.t_embedder(t)
        c3 = self.y_embedding_list[1](y, self.training)
        c = c1 + c2[:, None, :] + c3[:, None, :]
        for _ in range(self.num_layers):
            x = self.layers[_](x, c, flags)

        # X: B * N * dx, E: B * N * N * de
        # out_shape = (adj.shape[0], adj.shape[1], -1)
        x_out, e_out = self.out_layer(x, x_in, e_in, c, flags)
        # x_out = self.final(x_out).view(*out_shape)
        e_out = e_out.squeeze(-1)
        x = mask_x(x_out, flags)
        e = mask_adjs(e_out, flags)
        return x, e
