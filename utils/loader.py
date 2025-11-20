import torch
import random
import numpy as np

from src.classifier import ClassifierWithBias
from src.models import ScoreNetworkA, ScoreNetworkX, ScoreNetworkX_GMH, ScoreTransformer
from sde import VPSDE, VESDE, subVPSDE
from torch.nn.parallel import DistributedDataParallel as DDP

from src.loss import get_sde_loss_fn
from src.solver import get_pc_sampler, S4_solver
from src.metrics import gaussian, gaussian_emd
from utils.ema import ExponentialMovingAverage


def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    elif model_type == 'ScoreTransformer':
        model = ScoreTransformer(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)

    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
            # model = DDP(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')

    # model.requires_grad_(False)

    # # Step 3: 仅解冻 conditionnet
    # model.conditionnet.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr,
                                    weight_decay=config_train.weight_decay)
    # optimizer = torch.optim.Adam(model.conditionnet.parameters(), lr=1e-4, weight_decay=config_train.weight_decay)

    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    from data.data_loader_mol import dataloader
    return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    sx_b = batch[2].to(device_id)
    sa_b = batch[3].to(device_id)
    y_b = batch[4].to(device_id)
    return x_b, adj_b, sx_b, sa_b, y_b


def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    
    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True, 
                                likelihood_weighting=False, eps=config.train.eps)  #classifier guided: guidance_weight=1.0
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, device):
    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num  = config_train.data.max_node_num

    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    if config_module.predictor == 'S4':
        get_sampler = S4_solver
    else:
        get_sampler = get_pc_sampler

    if config_train.data.data in ['QM9', 'ZINC250k', 'MOSES', 'ChEMBL']:
        shape_x = (config_sample.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (config_sample.batch_size, max_node_num, max_node_num)
    else:
        shape_x = (config_train.data.batch_size, max_node_num, config_train.data.max_feat_num)
        shape_adj = (config_train.data.batch_size, max_node_num, max_node_num)
        
    sampling_fn = get_sampler(sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj, 
                                predictor=config_module.predictor, corrector=config_module.corrector,
                                snr=config_module.snr, scale_eps=config_module.scale_eps, 
                                n_steps=config_module.n_steps, 
                                probability_flow=config_sample.probability_flow, 
                                continuous=True, denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id)
    return sampling_fn


def load_model_params(config):
    config_m = config.model
    max_feat_num = config.data.max_feat_num

    params = {'model_type': config_m.type, 'max_feat_num': max_feat_num,
                    'max_node_num': config.data.max_node_num, 'nhid': config_m.nhid,
                    'num_layers': config_m.num_layers, 'c_final': config_m.c_final,
                    'num_heads': config_m.num_heads}
    return params


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    if ts is not None:
        config.ckpt = ts
    path = f'./checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt_dict = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model

def load_classifer_from_ckpt(params, state_dict, device):
    classifier = ClassifierWithBias(params['max_feat_num'], params['num_layers']-3, params['nhid'])
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    classifier.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            classifier = torch.nn.DataParallel(classifier, device_ids=device)
        classifier = classifier.to(f'cuda:{device[0]}')
    return classifier

def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree':gaussian_emd, 
                'cluster':gaussian_emd, 
                'orbit':gaussian,
                'spectral':gaussian_emd}
    return methods, kernels
