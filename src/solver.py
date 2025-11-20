import torch
import numpy as np
import abc
from tqdm import trange

from src.loss import get_score_fn
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE
from torch.nn import functional as F


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde_x, sde_adj, score_fn, probability_flow=False):
    super().__init__()
    self.sde_x = sde_x
    self.sde_adj = sde_adj
    # Compute the reverse SDE/ODE
    self.rsde_x = sde_x.reverse(score_fn, probability_flow)
    self.rsde_adj = sde_adj.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, adj, flags, t, sx, sa, y):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde_x, sde_adj, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde_x = sde_x
    self.sde_adj = sde_adj
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, adj, flags, t, sx, sa, y):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde_x, sde_adj, score_fn, probability_flow=False):
    super().__init__(sde_x, sde_adj, score_fn, probability_flow)

  def update_fn(self, x, adj, flags, t, sx, sa):
    dt = -1. / self.rsde_x.N

    score_x, score_adj = self.score_fn(x, adj, flags, t, sx, sa)

    drift_x, diffusion_x = self.rsde_x.sde(x, t, score_x)
    x_mean = x + drift_x * dt
    x = x_mean + diffusion_x[:, None, None] * np.sqrt(-dt) * gen_noise(x, flags, sym=False)

    drift_adj, diffusion_adj = self.rsde_adj.sde(adj, t, score_adj)
    adj_mean = adj + drift_adj * dt
    adj = adj_mean + diffusion_adj[:, None, None] * np.sqrt(-dt) * gen_noise(adj, flags, sym=True)

    return x, adj, x_mean, adj_mean


class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde_x, sde_adj, score_fn, probability_flow=False):
    super().__init__(sde_x, sde_adj, score_fn, probability_flow)

  def update_fn(self, x, adj, flags, t, sx, sa, y): #sa

    score_x, score_adj = self.score_fn(x, adj, flags, t, sx, sa, y)

    f_x, G_x = self.rsde_x.discretize(x, t, score_x)
    z = gen_noise(x, flags, sym=False)
    x_mean = x - f_x
    x = x_mean + G_x[:, None, None] * z

    f_adj, G_adj = self.rsde_adj.discretize(adj, t, score_adj)
    z = gen_noise(adj, flags, sym=True)
    adj_mean = adj - f_adj
    adj = adj_mean + G_adj[:, None, None] * z

    return x, adj, x_mean, adj_mean

class ReverseDiffusionPredictorA3C(Predictor):
  def __init__(self, sde_x, sde_adj, score_fn, probability_flow=False):
    super().__init__(sde_x, sde_adj, score_fn, probability_flow)

  def update_fn(self, x, adj, flags, t, sx, sa, y, action):

    score_x, score_adj = self.score_fn(x, adj, flags, t, sx, sa, y)

    # let action adjust score
    action_scale, action_shift = action.chunk(2, dim=-1)  # shape: (B, 1) each
    action_scale = torch.tanh(action_scale).view(-1, 1, 1) * 0.1
    action_shift = torch.tanh(action_shift).view(-1, 1, 1) * 0.05

    adjusted_score_x = score_x * (1.0 + action_scale) + action_shift
    adjusted_score_adj = score_adj * (1.0 + action_scale) + action_shift

    f_x, G_x = self.rsde_x.discretize(x, t, adjusted_score_x)
    z = gen_noise(x, flags, sym=False)
    x_mean = x - f_x
    x = x_mean + G_x[:, None, None] * z

    f_adj, G_adj = self.rsde_adj.discretize(adj, t, adjusted_score_adj)
    z = gen_noise(adj, flags, sym=True)
    adj_mean = adj - f_adj
    adj = adj_mean + G_adj[:, None, None] * z

    return x, adj, x_mean, adj_mean


class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde_x, sde_adj, score_fn, snr, scale_eps, n_steps):
    pass

  def update_fn(self, x, adj, flags, t, sx, sa):
    return x, adj, x, adj


class LangevinCorrector(Corrector):
  def __init__(self, sde_x, sde_adj, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde_x, sde_adj, score_fn, snr, scale_eps, n_steps)

  def update_fn(self, x, adj, flags, t, sx, sa):
    sde_x = self.sde_x
    sde_adj = self.sde_adj
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    score_x, score_adj = score_fn(x, adj, flags, t, sx, sa)

    if isinstance(sde_x, VPSDE) or isinstance(sde_x, subVPSDE):
      timestep = (t * (sde_x.N - 1) / sde_x.T).long()
      alpha = sde_x.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_x
      noise = gen_noise(x, flags, sym=False)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps

    if isinstance(sde_adj, VPSDE) or isinstance(sde_adj, subVPSDE):

      timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
      alpha = sde_adj.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_adj
      noise = gen_noise(adj, flags, sym=True)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      adj_mean = adj + step_size[:, None, None] * grad
      adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps

    return x, adj, x_mean, adj_mean


class Adamomentum(Corrector):
  def __init__(self, sde_x, sde_adj, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde_x, sde_adj, score_fn, snr, scale_eps, n_steps)

  def update_fn(self, x, adj, flags, t, sx, sa, y):
    sde_x, sde_adj = self.sde_x, self.sde_adj
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps
    eps = 0.0001
    global momentum_buffer_x
    global m_term_x, grad_pre_x, step_adam_x
    global momentum_buffer_adj
    global m_term_adj, grad_pre_adj, step_adam_adj

    grad_x, grad_adj = score_fn(x, adj, flags, t, sx, sa, y)

    if isinstance(sde_x, VPSDE) or isinstance(sde_x, subVPSDE):
      timestep = (t * (sde_x.N - 1) / sde_x.T).long()
      alpha = sde_x.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      noise = gen_noise(x, flags, sym=False)
      momentum_buffer_x_norm = torch.norm(momentum_buffer_x.reshape(grad_x.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

      if momentum_buffer_x_norm == 0:
        beta_1 = 0
        m_term_x = grad_x
        m_term_x_norm = torch.norm(m_term_x.reshape(m_term_x.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / m_term_x_norm) ** 2 * 2 * alpha
        step_adam_x = step_size
      else:
        grad_diff = grad_x - grad_pre_x
        grad_diff_norm = torch.norm(grad_diff.reshape(grad_diff.shape[0], -1), dim=-1).mean()
        beta_1 = max(0, min(1. - eps, (
                  (1. - step_adam_x[:, None, None, None].mean() * (grad_diff_norm / momentum_buffer_x_norm))
                  / (1. + step_adam_x[:, None, None, None].mean() * (grad_diff_norm / momentum_buffer_x_norm))) ** 2))
        m_term_x = beta_1 * m_term_x + (1 - beta_1) * grad_x
        m_term_x_norm = torch.norm(m_term_x.reshape(m_term_x.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / m_term_x_norm) ** 2 * 2 * alpha
        step_adam_x = step_size * (1 + beta_1) ** 2

      x_mean = x + step_adam_x[:, None, None] * m_term_x
      momentum_buffer_x = x_mean - x
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      grad_pre_x = grad_x

    for i in range(n_steps):
      noise = gen_noise(adj, flags)
      momentum_buffer_adj_norm = torch.norm(momentum_buffer_adj.reshape(grad_adj.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

      if momentum_buffer_adj_norm == 0:
        beta_1 = 0
        m_term_adj = grad_adj
        m_term_adj_norm = torch.norm(m_term_adj.reshape(m_term_adj.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / m_term_adj_norm) ** 2 * 2 * alpha
        step_adam_adj = step_size
      else:
        grad_diff = grad_adj - grad_pre_adj
        grad_diff_norm = torch.norm(grad_diff.reshape(grad_diff.shape[0], -1), dim=-1).mean()

        beta_1 = max(0, min(1. - eps, (
                  (1. - step_adam_adj[:, None, None, None].mean() * (grad_diff_norm / momentum_buffer_adj_norm))
                  / (1. + step_adam_adj[:, None, None, None].mean() * (
                    grad_diff_norm / momentum_buffer_adj_norm))) ** 2))
        m_term_adj = beta_1 * m_term_adj + (1 - beta_1) * grad_adj
        m_term_adj_norm = torch.norm(m_term_adj.reshape(m_term_adj.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / m_term_adj_norm) ** 2 * 2 * alpha
        step_adam_adj = step_size * (1 + beta_1) ** 2

      adj_mean = adj + step_adam_adj[:, None, None] * m_term_adj
      momentum_buffer_adj_norm = adj_mean - adj
      adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      grad_pre_adj = grad_adj
    return x, adj, x_mean, adj_mean


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model, init_flags, sx, sa, y):

    score_fn = get_score_fn(sde_x, sde_adj, model, train=False, continuous=continuous)

    if predictor == 'Reverse':
      predictor_fn = ReverseDiffusionPredictor
    else:
      predictor_fn = EulerMaruyamaPredictor

    if corrector == 'Langevin':
      corrector_fn = LangevinCorrector
    elif corrector == 'Adamomentum':
      corrector_fn = Adamomentum
    else:
      corrector_fn = NoneCorrector

    predictor_obj = predictor_fn(sde_x, sde_adj, score_fn, probability_flow)
    corrector_obj = corrector_fn(sde_x, sde_adj, score_fn, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      # x[~flags] = sx[~flags]
      adj = mask_adjs(adj, flags)
      # adj[~flags] = sa[~flags]
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      global momentum_buffer_x
      global momentum_buffer_adj
      momentum_buffer_x = torch.zeros_like(x)
      momentum_buffer_adj = torch.zeros_like(adj)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, adj, x_mean, adj_mean = corrector_obj.update_fn(x, adj, flags, vec_t, sx, sa, y)
        x, adj, x_mean, adj_mean = predictor_obj.update_fn(_x, adj, flags, vec_t, sx, sa, y)
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
  return pc_sampler


# -------- S4 solver --------
def S4_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='None', corrector='None', 
                        snr=0.1, scale_eps=1.0, n_steps=1, 
                        probability_flow=False, continuous=False,
                        denoise=True, eps=1e-3, device='cuda'):

  def s4_solver(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      dt = -1. / diff_steps

      # -------- Rverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t
        vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2) 

        # -------- Score computation --------
        score_x = score_fn_x(x, adj, flags, vec_t)
        score_adj = score_fn_adj(x, adj, flags, vec_t)

        Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
        Sdrift_adj  = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

        # -------- Correction step --------
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)
      
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * score_x
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * score_adj
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        # -------- Prediction step --------
        x_mean = x
        adj_mean = adj
        mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
        
        x = x + Sdrift_x * dt
        adj = adj + Sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt) 
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

        x_mean = mu_x
        adj_mean = mu_adj
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), 0
  return s4_solver
