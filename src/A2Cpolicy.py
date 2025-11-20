# A3C for Molecular Graph Generation using a pretrained Diffusion Model
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Categorical

from main import get_config
import torch.nn.functional as F
from tqdm import trange
from sampler import Sampler_mol
from src.layers import DenseGCNConv
from src.loss import get_score_fn
from src.solver import ReverseDiffusionPredictor, EulerMaruyamaPredictor
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
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pickle

from utils.loader import load_device
from torch.distributions import Normal

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.attn_weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.attn_bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x, adj):
        # 通过图注意力机制进行调整
        h = torch.matmul(x, self.attn_weight)  # 计算变换后的节点特征
        h = F.leaky_relu(h + self.attn_bias)  # 非线性激活函数

        # 结合邻接矩阵进行加权
        h = torch.matmul(adj, h)  # 加权求和
        return h


# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)  # 均值
        self.sigma = nn.Linear(hidden_dim, action_dim)  # 标准差

    def forward(self, x):
        x = self.shared(x)
        mu = self.mu(x)
        sigma = torch.softplus(self.sigma(x))  # 保证标准差为正
        return mu, sigma

# --- Worker Environment Interface ---
class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, idx, args):
        super(Worker, self).__init__()
        self.gnet = gnet
        self.opt = opt
        self.local_net = ActorCritic(args.state_dim, args.action_dim)
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.name = f'worker_{idx}'
        self.args = args
        self.device = load_device()
        self.ckpt_dict = load_ckpt(self.args, self.device)
        self.configt = self.ckpt_dict['model_config']
        self.convs = torch.nn.ModuleList()
        self.activation = torch.tanh
        self.readout = nn.Linear(self.configt.data.max_node_num, self.configt.data.max_feat_num)
        for _ in range(4):
            self.convs.append(DenseGCNConv(self.configt.data.max_feat_num, self.configt.data.max_node_num))

        load_seed(self.args.seed)

    def run(self):

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.args.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.args)

        # -------- Load models --------
        self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)

        sde_x = load_sde(self.configt.sde.x)
        sde_adj = load_sde(self.configt.sde.adj)
        max_node_num = self.configt.data.max_node_num

        if self.configt.data.data in ['QM9', 'ZINC250k', 'MOSES', 'chembl']:
            shape_x = (self.args.sample.batch_size, max_node_num, self.configt.data.max_feat_num)
            shape_adj = (self.args.sample.batch_size, max_node_num, max_node_num)
        else:
            shape_x = (self.configt.data.batch_size, max_node_num, self.configt.data.max_feat_num)
            shape_adj = (self.configt.data.batch_size, max_node_num, max_node_num)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.args.sample.seed}')
        load_seed(self.args.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)  # for init_flags
        with open(f'data/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD
        with open(f'data/{self.args.sample.template}.csv') as f:
            lines = f.readlines()
        scaf_ = []
        for line in lines:
            m = Chem.MolFromSmiles(line.strip())
            scaffold = MurckoScaffold.GetScaffoldForMol(m)
            scaf_.append(scaffold)
        num_sampling_rounds = math.ceil(10000 / self.args.sample.batch_size)
        xs, adjs = [], []
        score_fn = get_score_fn(sde_x, sde_adj, self.model, train=False, continuous=True)

        if self.args.sampler.predictor == 'Reverse':
            predictor_fn = ReverseDiffusionPredictor
        else:
            predictor_fn = EulerMaruyamaPredictor

        predictor_obj = predictor_fn(sde_x, sde_adj, score_fn, self.args.sample.probability_flow)

        total_step = 1
        while self.global_ep.value < self.args.max_episodes:
            buffer_s, buffer_a, buffer_r = [], [], []
            # -------- Initial sample --------
            ep_r = 0
            time.time()
            start = self.global_ep.value * self.args.sample.batch_size
            end = min((self.global_ep.value + 1) * self.args.sample.batch_size, len(scaf_))
            scaf_r = scaf_[start:end]
            self.init_flags, sx, sa, template = init_flags(self.train_graph_list, self.configt, scaf_r,
                                                 self.args.sample.batch_size)
            self.init_flags = self.init_flags.to(self.device[0])
            sx = sx.to(self.device[0])
            sa = sa.to(self.device[0])
            template = template.to(self.device[0])

            x = sde_x.prior_sampling(shape_x).to(self.device)
            adj = sde_adj.prior_sampling_sym(shape_adj).to(self.device)
            flags = self.init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, self.args.sample.eps, diff_steps, device=self.device)
            x_list = []
            for _ in range(len(self.convs)):
                _x = self.convs[_](x, adj[:, _, :, :])
                _x = self.activation(_x)
                x_list.append(_x)
            xs = torch.cat(x_list, dim=-1)

            state = self.readout(xs)


            for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                mu, sigma = self.local_net(state)
                dist = Normal(mu, sigma)
                action = dist.sample()

                x, adj, x_mean, adj_mean = predictor_obj.update_fn(x, adj, flags, vec_t, sx, sa, action)
                x_list = []
                for _ in range(len(self.convs)):
                    _x = self.convs[_](x, adj[:, _, :, :])
                    _x = self.activation(_x)
                    x_list.append(_x)
                xs = torch.cat(x_list, dim=-1)

                next_state = self.readout(xs)

                reward = self.compute_reward(x, adj, template)

                ep_r += reward
                done = self.check_termination_condition(x, adj, ep_r)

                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % self.args.update_freq == 0 or done:
                    self.update_global(buffer_s, buffer_a, buffer_r, next_state, done)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.local_net.load_state_dict(self.gnet.state_dict())

                state = next_state
                total_step += 1
                if done:
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    with self.global_ep_r.get_lock():
                        self.global_ep_r.value = 0.99 * self.global_ep_r.value + 0.01 * ep_r
                    self.res_queue.put(self.global_ep_r.value)
                    break
            samples_int = quantize_mol(adj_mean)

            samples_int = samples_int - 1
            samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

            adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
            x = torch.where(x_mean > 0.5, 1, 0)
            x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5

            xs.append(x), adjs.append(adj)
        xs, adjs = torch.concat(xs, dim=0), torch.concat(adjs, dim=0)

        gen_mols, num_mols_wo_correction = gen_mol(xs, adjs, self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
            for smiles in gen_smiles:
                f.write(f'{smiles}\n')

        # -------- Evaluation --------
        scaf_smiles = [Chem.MolToSmiles(m) for m in scaf_]
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles,
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
        scores_nspdk = eval_graph_list(self.test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']

        logger.log(f'Number of molecules: {num_mols}')
        logger.log(f'validity w/o correction: {num_mols_wo_correction / num_mols}')
        logger.log(f'success rate: {success_rate}')
        for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty', 'IntDiv', 'Novelty', 'Scaf/TestSF']:
            logger.log(f'{metric}: {scores[metric]}')
        logger.log(f'NSPDK MMD: {scores_nspdk}')
        logger.log('=' * 100)

    def update_global(self, states, actions, rewards, next_state, done):
        R = 0.0 if done else self.local_net(torch.tensor(next_state, dtype=torch.float32))[1].item()
        buffer_v_target = []
        for r in reversed(rewards):
            R = r + self.args.gamma * R
            buffer_v_target.insert(0, R)

        loss = self.compute_loss(states, actions, buffer_v_target)
        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.gnet.parameters()):
            gp._grad = lp.grad
        self.opt.step()

    def compute_loss(self, states, actions, targets):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        targets = torch.tensor(targets, dtype=torch.float32)
        logits, values = self.local_net(states)
        td = targets.unsqueeze(1) - values
        critic_loss = td.pow(2)
        probs = Categorical(logits=logits)
        actor_loss = -probs.log_prob(actions) * td.detach().squeeze()
        total_loss = (actor_loss + critic_loss).mean()
        return total_loss

    # def apply_action_to_state(self, x, adj, action):
    #     """
    #     Modify x or adj based on the action.
    #     You can define action as:
    #     - choosing a position to grow atom
    #     - changing bond type
    #     - modifying atom type
    #     """
    #     action_weight = torch.sigmoid(action)
    #     gcn_layer = GraphAttentionLayer(x.size(-1), x.size(-1))
    #     x_new = gcn_layer(x, adj)
    #     adj_new = adj + action_weight[:, None, None] * adj
    #     x_new = F.relu(x_new)
    #     return x_new, adj_new

    def compute_reward(self, x, adj, template):
        x_ref, adj_ref = template
        sim_node = F.cosine_similarity(x.flatten(), x_ref.flatten(), dim=0)
        sim_adj = F.cosine_similarity(adj.flatten(), adj_ref.flatten(), dim=0)
        sim = 0.5 * (sim_node + sim_adj)
        return sim

    def check_termination_condition(self, x, adj, reward_history, max_steps=1000, threshold=1e-5, target_qed_threshold=0.8,
                                    min_reward_change=1e-4):
        # 检查图的稳定性
        x_diff = torch.abs(x - x.mean(dim=0, keepdim=True)).max()
        adj_diff = torch.abs(adj - adj.mean(dim=0, keepdim=True)).max()

        if x_diff < threshold and adj_diff < threshold:
            return True  # 图结构稳定，可以终止

        # 检查奖励变化
        if len(reward_history) > 1 and abs(reward_history[-1] - reward_history[-2]) < min_reward_change:
            return True  # 奖励变化小，任务完成

        # 检查是否超过最大步骤数
        if len(reward_history) >= max_steps:
            return True  # 超过最大步骤数，终止

        return False

# --- Main A3C Training Loop ---
def train(args):
    gnet = ActorCritic(args.state_dim, args.action_dim)
    gnet.share_memory()
    opt = optim.Adam(gnet.parameters(), lr=args.lr)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, args) for i in range(args.n_workers)]
    [w.start() for w in workers]
    results = []
    while True:
        r = res_queue.get()
        if r is not None:
            results.append(r)
        else:
            break
    [w.join() for w in workers]
    return results




if __name__ == '__main__':
    config = get_config()
    train(config)