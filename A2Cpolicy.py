# A3C for Molecular Graph Generation using a pretrained Diffusion Model
import yaml
from easydict import EasyDict as edict
import numpy as np
from dgl.data.utils import load_graphs
import argparse
import torch.optim as optim

from data.ligand2ppgraph import match_score
from data.smile_to_graph import construct_atomic_number_array, construct_discrete_edge_matrix
from torch.nn.utils.rnn import pad_sequence

from tqdm import trange
from sampler import Sampler_mol
from src.layers import glorot, zeros, DenseGCNConv
from src.loss import get_score_fn
from src.ggcn_layers import GGCNEncoderBlock
from src.solver import ReverseDiffusionPredictor, EulerMaruyamaPredictor, ReverseDiffusionPredictorA3C
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
    load_ema_from_ckpt, load_sampling_fn, load_eval_settings, load_classifer_from_ckpt, load_sde
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol, mask_x, mask_adjs
from utils.plot import save_graph_list, plot_graphs_list
from src.metrics import eval_graph_list
import math
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from mini_moses.metrics.metrics import get_all_metrics

from mini_moses.metrics.utils import SA, QED
import time
from torch.nn import Parameter
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pickle

from utils.loader import load_device
from torch.distributions import Normal

from vision_lstm.vision_lstm2 import VisionLSTM2

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
MAX_NUM_PP_GRAPHS=8

import torch
import torch.nn as nn

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        Args:
            x: (batch_size, num_nodes, in_features)
            adj: (batch_size, num_nodes, num_nodes)  -- adjacency matrix (binary or weighted)
        Returns:
            out: (batch_size, num_nodes, out_features)
        """
        # Add self-loop: Identity matrix
        batch_size, num_nodes, _ = x.size()
        device = x.device
        I = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self_loop = adj + I

        # Degree normalization
        degree = adj_with_self_loop.sum(dim=-1, keepdim=True)  # (batch_size, num_nodes, 1)
        degree = degree.clamp(min=1e-6)  # prevent division by zero
        norm_adj = adj_with_self_loop / degree

        # GCN propagation
        agg = torch.bmm(norm_adj, x)  # (batch_size, num_nodes, in_features)
        out = self.linear(agg)
        return out


class A3CDenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(A3CDenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, x, adj):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        if adj.dim() == 4:
            CB, B, N, _ = adj.size()

            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, :, idx, idx] = 1 if not self.improved else 2
        else:
            B, N, _ = adj.size()

            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias


        return out

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, conv_dim, hidden_dim=32):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, action_dim)  # 输出离散动作的 logits
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        self.activation = torch.tanh
        self.fdim = state_dim + 2 * conv_dim
        self.readout = nn.Linear(self.fdim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.convs = torch.nn.ModuleList()
        for _ in range(2):
            if _ == 0:
                self.convs.append(
                    A3CDenseGCNConv(state_dim, conv_dim))  # self.neat * 2
            else:
                self.convs.append(A3CDenseGCNConv(conv_dim, conv_dim))


    def forward(self, x, adj):
        x_list = [x]
        _x = x
        for _ in range(len(self.convs)):
            _x = self.convs[_](_x, adj)
            _x = self.activation(_x)
            x_list.append(_x)
        xs = torch.cat(x_list, dim=-1)

        state = self.readout(xs)
        if state.dim() == 4:
            state = state.mean(dim=2)
        else:
            state = state.mean(dim=1)
        state = self.shared(state)
        # logits = self.fc2(F.relu(self.fc1(state)))
        mu = self.fc_mu(state)
        log_std = self.fc_log_std(state).clamp(min=-5, max=2)
        value = self.value_head(state).squeeze(-1)
        return mu, log_std, value

# --- Worker Environment Interface ---
def mols_to_graph(mol, max_atoms=64):
    """
    从 mol 成分子图的节点特征矩阵 (X) 和邻接矩阵 (A)。
    :param mol: 分子mol格式
    :param max_atoms: 固定最大原子数，用于统一矩阵大小
    :return: 节点特征矩阵 X 和邻接矩阵 A
    """
    # 获取原子数
    num_atoms = mol.GetNumAtoms()
    if num_atoms > max_atoms:
        raise ValueError(f"Number of atoms ({num_atoms}) exceeds max_atoms ({max_atoms})")

    X = construct_atomic_number_array(mol, out_size=max_atoms)
    A = construct_discrete_edge_matrix(mol, out_size=max_atoms)

    zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    x_ = np.zeros((64, 10), dtype=np.float32)
    for i in range(64):
        ind = zinc250k_atomic_num_list.index(X[i])
        x_[i, ind] = 1.
    X = torch.tensor(x_).to(torch.float32)
    # single, double, triple and no-bond; the last channel is for virtual edges
    A = np.concatenate([A[:3], 1 - np.sum(A[:3], axis=0, keepdims=True)],
                       axis=0).astype(np.float32)

    X = X[:, :-1]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
    A = torch.tensor(A.argmax(axis=0))  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
    # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
    A = torch.where(A == 3, 0, A + 1).to(torch.float32)

    return X, A


def graph_to_image(X, A, max_nodes=64):
    """
    将分子图转换为图像格式。
    :param X: 节点特征矩阵 (n_nodes, n_features)
    :param A: 邻接矩阵 (n_nodes, n_nodes)
    :param max_nodes: 固定节点数，图像宽高 (max_nodes, max_nodes)
    :return: 图像张量 (C, H, W)
    """
    n_nodes, n_features = X.shape
    # 初始化图像张量
    img = np.zeros((n_features + 1, max_nodes, max_nodes))  # 通道数 = 节点特征数 + 邻接矩阵通道

    # 填充节点特征通道
    for i in range(n_features):
        img[i, :n_nodes, :n_nodes] = np.expand_dims(X[:, i], axis=1)  # 节点特征
        # feature_values = X[:, i]
        # if n_nodes > 0:
        #     img[i, :n_nodes, :n_nodes] = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))

    # 填充邻接矩阵通道
    img[-1, :n_nodes, :n_nodes] = A

    return torch.tensor(img, dtype=torch.float32)


class Worker: #(mp.Process)
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue,
                 args, configt, ckpt_dict, results,
                 model, sde_x, sde_adj, shape_x, shape_adj,
                 device, scaf_, train_graph_list, log_dir, log_name,
                 pp_graphs, bbb_predictor
                 ):
        super(Worker, self).__init__()
        self.gnet = gnet
        self.opt = opt
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.configt = configt
        self.ckpt_dict = ckpt_dict
        self.results = results
        self.pp_graphs = pp_graphs
        # self.name = f'worker_{idx}'
        self.args = args
        # self.local_net = ActorCritic(self.configt.data.max_feat_num, 1, self.configt.data.max_node_num)

        load_seed(self.args.seed)
        self.device = device
        self.model=model
        self.sde_x = sde_x
        self.sde_adj = sde_adj
        self.scaf_ = scaf_
        self.shape_x = shape_x
        self.shape_adj = shape_adj
        self.log_dir = log_dir
        self.log_name = log_name
        self.train_graph_list = train_graph_list
        self.BBB_predictor = bbb_predictor


    def run(self):
        num_sampling_rounds = math.ceil(10000 / self.args.sample.batch_size)
        score_fn = get_score_fn(self.sde_x, self.sde_adj, self.model, train=False, continuous=True)

        if self.args.sampler.predictor == 'Reverse':
            predictor_fn = ReverseDiffusionPredictorA3C
        else:
            predictor_fn = EulerMaruyamaPredictor

        predictor_obj = predictor_fn(self.sde_x, self.sde_adj, score_fn, self.args.sample.probability_flow)

        total_step = 1
        while self.global_ep < num_sampling_rounds:
            buffer_s, buffer_a, buffer_r, buffer_lp = [], [], [], []
            # -------- Initial sample --------
            time.time()
            g_mod = self.global_ep % 3
            start = g_mod * self.args.sample.batch_size
            end = min((g_mod + 1) * self.args.sample.batch_size, len(self.scaf_))
            scaf_r = self.scaf_[start:end]
            self.init_flags, sx, sa = init_flags(self.train_graph_list, self.configt, scaf_r,
                                                 self.args.sample.batch_size)
            self.init_flags = self.init_flags.to(self.device[0])
            sx = sx.to(self.device[0])
            sa = sa.to(self.device[0])
            # y = torch.tensor([0.75, 2.5]).repeat(self.args.sample.batch_size, 1) # ZINC250k
            y = torch.tensor([0.7, 0.5]).repeat(self.args.sample.batch_size, 1) # ChEMBL
            y = y.to(self.device[0])


            # self.local_net = self.local_net.to(self.device[0])
            self.gnet = self.gnet.to(self.device[0])
            self.BBB_predictor = self.BBB_predictor.to(self.device[0])

            diff_steps = self.sde_adj.N
            timesteps = torch.linspace(self.sde_adj.T, self.args.sample.eps, diff_steps, device=self.device[0])

            done = False
            step_count = 0
            while not done and step_count < 10:
                xs, adjs = [], []
                ep_r = []

                with torch.no_grad():
                    x = self.sde_x.prior_sampling(self.shape_x).to(self.device[0])
                    adj = self.sde_adj.prior_sampling_sym(self.shape_adj).to(self.device[0])
                    flags = self.init_flags
                    x = mask_x(x, flags)
                    adj = mask_adjs(adj, flags)

                    mu, log_std, value = self.gnet(x, adj)
                    mu = torch.clamp(mu, min=-10.0, max=10.0)
                    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
                    std = log_std.exp()
                    dist = Normal(mu, std)
                    action = dist.rsample()
                    log_prob = dist.log_prob(action).detach()
                    for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
                        t = timesteps[i]
                        vec_t = torch.ones(self.shape_adj[0], device=t.device) * t

                        x, adj, x_mean, adj_mean = predictor_obj.update_fn(x, adj, flags, vec_t, sx, sa, y, action)

                samples_int = quantize_mol(adj_mean)

                samples_int = samples_int - 1
                samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

                adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
                x = torch.where(x_mean > 0.5, 1, 0)
                x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5


                xs.append(x), adjs.append(adj)
                xs, adjs = torch.concat(xs, dim=0), torch.concat(adjs, dim=0)
                gen_mols, num_mols_wo_correction = gen_mol(xs, adjs, self.configt.data.data)

                reward, mapping_scores = self.compute_reward(gen_mols, self.pp_graphs)

                ep_r.append(reward.detach().cpu())
                done = self.check_termination_condition(ep_r, mapping_scores)

                buffer_s.append((x_mean.cpu(), adj_mean.cpu()))
                buffer_a.append(action.cpu())
                buffer_r.append(reward.cpu())
                buffer_lp.append(log_prob.cpu())
                self.update_global(buffer_s, buffer_a, buffer_r, done)
                # self.update_global_ppo(buffer_s, buffer_a, buffer_r, buffer_lp, done)
                buffer_s, buffer_a, buffer_r, buffer_lp = [], [], [], []
                # self.local_net.load_state_dict(self.gnet.state_dict())

                total_step += 1
                if done:
                    self.global_ep += 1
                    self.global_ep_r = 0.99 * self.global_ep_r + 0.01 * (torch.stack(ep_r).sum(dim=0))
                    self.res_queue.append(self.global_ep_r)
                step_count += 1


            num_mols = len(gen_mols)

            gen_smiles = mols_to_smiles(gen_mols)
            gen_smiles = [smi for smi in gen_smiles if len(smi)]

            # -------- Save generated molecules --------
            with open(os.path.join(self.log_dir, f'{self.log_name}_A3C.txt'), 'a') as f:
                for smiles in gen_smiles:
                    f.write(f'{smiles}\n')
            self.results.append((num_mols, num_mols_wo_correction, gen_mols, gen_smiles))
            if done is False:
                self.global_ep += 1


    def update_global(self, states, actions, rewards, done, gamma=0.95):
        R = torch.zeros_like(rewards[-1]).to('cpu') if done else rewards[-1].detach().to('cpu')
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        states_x = torch.stack([s[0] for s in states]).to(self.device[0])  # e.g., shape (T, B, ...)
        states_adj = torch.stack([s[1] for s in states]).to(self.device[0])
        actions = torch.stack(actions).to(self.device[0])
        returns = torch.stack(returns).to(self.device[0])

        mu, log_std, value = self.gnet(states_x, states_adj)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions)
        log_probs = log_probs.sum(dim=-1)

        td = returns.detach() - value
        critic_loss = td.pow(2)
        actor_loss = -log_probs * td.detach()

        loss = (actor_loss + critic_loss).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        del actions, rewards, states_adj, states_x, mu, log_std, value, returns, log_probs
        torch.cuda.empty_cache()

    def update_global_ppo(self, states, actions, rewards, old_log_probs, done, gamma=0.95, eps_clip=0.2):
        R = torch.zeros_like(rewards[-1]).to('cpu') if done else rewards[-1].detach().to('cpu')
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).to(self.device[0])

        states_x = torch.stack([s[0] for s in states]).to(self.device[0])
        states_adj = torch.stack([s[1] for s in states]).to(self.device[0])
        actions = torch.stack(actions).to(self.device[0])
        old_log_probs = torch.stack(old_log_probs).to(self.device[0])

        # 前向计算新策略
        mu, log_std, value = self.gnet(states_x, states_adj)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = log_std.exp()
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions)

        advantage = returns.detach() - value

        ratio = (new_log_probs - old_log_probs).sum(-1).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        del actions, rewards, states_adj, states_x, mu, log_std, value, returns, old_log_probs, new_log_probs
        torch.cuda.empty_cache()

    def compute_reward(self, gen_mols, pp_graphs, target_score=0.7):
        # similarity with reference
        # sim_node = F.cosine_similarity(x.flatten(), x_ref.flatten(), dim=0)
        # sim_adj = F.cosine_similarity(adj.flatten(), adj_ref.flatten(), dim=0)
        # sim = 0.5 * (sim_node + sim_adj)

        # pharmacophore graph matching score
        mapping_scores = []
        for m in gen_mols:
            pharma_match_score_list = [match_score(m, pp_graph) for pp_graph in pp_graphs]
            mapping_score = max(pharma_match_score_list)
            mapping_scores.append(mapping_score)

        mapping_scores = torch.tensor(mapping_scores, dtype=torch.float32)

        # BBB predictor
        target_name = self.log_name.split('-')[-1].lower()
        if target_name == 'lrrk2':
            data_list = []
            for i, mol in enumerate(gen_mols):
                try:
                    X, A = mols_to_graph(mol)
                except:
                    continue
                img_tensor = graph_to_image(X, A)
                data_list.append(img_tensor)
            inputs = torch.stack(data_list).to(self.device[0])
            y_hat = self.BBB_predictor(inputs)
            output_BBB = torch.softmax(y_hat, dim=1)[:, 1].to("cpu")
            rewards = 0.5 * (1.0 - torch.abs(mapping_scores - target_score) + output_BBB)
            rewards = rewards.clamp(min=0.0)
        else:
            rewards = 0.5 * (1.0 - torch.abs(mapping_scores - target_score))
            rewards = rewards.clamp(min=0.0)

        return rewards, mapping_scores

    def check_termination_condition(self, reward_history, mapping_scores, target_map_threshold=0.6):
        if len(reward_history) == 0:
            return False

        success_rate = (mapping_scores >= target_map_threshold).float().mean().item()

        if success_rate > 0.6:  # 60% 的生成分子接近目标
            return True

        return False

# --- Main A3C Training Loop ---
def train(args, pp_graph_list):
    device = load_device()
    ckpt_dict = load_ckpt(args, device)
    configt = ckpt_dict['model_config']
    gnet = ActorCritic(configt.data.max_feat_num, 2, configt.data.max_node_num)
    gnet.share_memory()
    opt = optim.RMSprop(gnet.parameters(), lr=0.01)
    log_folder_name, log_dir, _ = set_log(configt, is_train=False)
    log_name = f"{args.ckpt}-sample-{args.sample.template}"
    logger = Logger(str(os.path.join(log_dir, f'{log_name}.log')), mode='a')

    if not check_log(log_folder_name, log_name):
        start_log(logger, configt)
        train_log(logger, configt)
    sample_log(logger, args)

    # -------- Load models --------
    model = load_model_from_ckpt(ckpt_dict['params'], ckpt_dict['state_dict'], device)

    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)
    max_node_num = configt.data.max_node_num

    if configt.data.data in ['QM9', 'ZINC250k', 'MOSES', 'ChEMBL']:
        shape_x = (args.sample.batch_size, max_node_num, configt.data.max_feat_num)
        shape_adj = (args.sample.batch_size, max_node_num, max_node_num)
    else:
        shape_x = (configt.data.batch_size, max_node_num, configt.data.max_feat_num)
        shape_adj = (configt.data.batch_size, max_node_num, max_node_num)

    # -------- Generate samples --------
    logger.log(f'GEN SEED: {args.sample.seed}')
    load_seed(args.sample.seed)

    train_smiles, test_smiles = load_smiles(configt.data.data)
    train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

    train_graph_list, _ = load_data(configt, get_graph_list=True)  # for init_flags
    with open(f'data/{configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
        test_graph_list = pickle.load(f)  # for NSPDK MMD

    # single target
    with open(f'data/{args.sample.template}.csv') as f:
        lines = f.readlines()
    scaf_ = []
    for line in lines:
        m = Chem.MolFromSmiles(line.strip())
        scaf_.append(m)

    #  dual targets
    # with open(f'data/{args.sample.template.lower()}_pdb/{args.sample.template.lower()}.csv') as f:
    #     lines = f.readlines()
    # scaf_ = []
    # for line in lines:
    #     m = Chem.MolFromSmiles(line.strip())
    #     scaf_.append(m)

    global_ep, global_ep_r = 0, 0.0  # mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    res_queue, results = [], []
    # workers = [Worker(gnet, opt, global_ep, global_ep_r,
    #                   res_queue, i, args, configt, ckpt_dict,
    #                   model, sde_x, sde_adj, shape_x, shape_adj,
    #                   device, scaf_, train_graph_list, log_dir, log_name) for i in range(4)]
    # [w.start() for w in workers]
    # [w.join() for w in workers]
    # pp_graph_list = dgl.batch(pp_graph_list)
    BBB_predictor = VisionLSTM2(
        dim=200,  # latent dimension (192 for ViL-T)
        depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
        patch_size=8,  # patch_size (results in 64 patches for 32x32 images)
        input_shape=(10, 64, 64),  # RGB images with resolution 32x32
        output_shape=(2,),  # classifier with 10 classes
        drop_path_rate=0.02,  # stochastic depth parameter (disabled for ViL-T)
    )
    checkpoint = torch.load("vision_lstm/BBB_predictor_best.pth")
    BBB_predictor.load_state_dict(checkpoint['model_state_dict'])
    BBB_predictor.eval()

    worker = Worker(gnet, opt, global_ep, global_ep_r,
                      res_queue, args, configt, ckpt_dict, results,
                      model, sde_x, sde_adj, shape_x, shape_adj,
                      device, scaf_, train_graph_list, log_dir,
                    log_name, pp_graph_list, BBB_predictor)
    worker.run()
    # results = []
    # while True:
    #     r = res_queue.get()
    #     if r is not None:
    #         results.append(r)
    #     else:
    #         break
    num_mols, num_mols_wo_correction=0, 0
    gen_mols, gen_smiles=[],[]
    for res in results:
        num_mols_, num_mols_wo_correction_, gen_mols_, gen_smiles_ = res
        num_mols += num_mols_
        num_mols_wo_correction += num_mols_wo_correction_
        gen_mols.extend(gen_mols_)
        gen_smiles.extend(gen_smiles_)

    # -------- Evaluation --------
    scaf_smiles = [Chem.MolToSmiles(m) for m in scaf_]
    scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device[0], n_jobs=8, test=test_smiles,
                             test_scaffolds=scaf_smiles, train=train_smiles)
    num_success = 0
    for m in gen_mols:
        if m is None:
            continue
        s = SA(m)
        q = QED(m)
        if s < 4 and q > 0.67:
            num_success += 1
    success_rate = num_success / num_mols
    scores_nspdk = eval_graph_list(test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

    logger.log(f'Number of molecules: {num_mols}')
    logger.log(f'validity w/o correction: {num_mols_wo_correction / num_mols}')
    logger.log(f'success rate: {success_rate}')
    for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty', 'IntDiv', 'Scaf/TestSF']:
        logger.log(f'{metric}: {scores[metric]}')
    logger.log(f'NSPDK MMD: {scores_nspdk}')
    logger.log('=' * 100)


def get_config():
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='ICT/controlgsde/')
    parser.add_argument('--config', type=str, default='sample_chembl',
                                    help="Path of config file")
    parser.add_argument('--comment', type=str, default="",
                                    help="A single line comment for the experiment")

    parser.add_argument('--seed', type=int, default=42)


    args = parser.parse_args()

    config_dir = f'./config/{args.config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = args.seed

    return config


if __name__ == '__main__':
    config = get_config()
    # mp.set_start_method('spawn', force=True)
    pp_graph_list, _ = load_graphs(f"data/{config.sample.template.lower()}_pdb/{config.sample.template.lower()}_phar_graphs.bin")
    for pp_graph in pp_graph_list:
        pp_graph.ndata['h'] = \
            torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
    train(config, pp_graph_list)