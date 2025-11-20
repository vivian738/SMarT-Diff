import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from src.classifier import ClassifierWithBias
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
    load_ema, load_loss_fn, load_batch, load_ckpt, load_model_from_ckpt
from utils.logger import Logger, set_log, start_log, train_log


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)

        self.params = load_model_params(self.config)
    
    def train(self, ts, finetune=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train,
                                                                                self.device)
        if finetune:
            self.ckpt_dict = load_ckpt(self.config, self.device)
            self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)

        self.ema = load_ema(self.model, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)
        # self.classifier = ClassifierWithBias(self.params_x['max_feat_num'], self.params_x['num_layers']-3, self.params_x['nhid']).to(f'cuda:{self.device[0]}')

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model.train()

            for _, train_b in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                x, adj, sx, sa, y = load_batch(train_b, self.device)
                loss_subject = (x, adj, sx, sa, y)

                loss, loss_x, loss_adj = self.loss_fn(self.model, *loss_subject)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)

                self.optimizer.step()

                # -------- EMA update --------
                self.ema.update(self.model.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            self.model.eval()
            for _, test_b in enumerate(self.test_loader):   
                
                x, adj, sx, sa, y = load_batch(test_b, self.device)
                loss_subject = (x, adj, sx, sa, y)

                with torch.no_grad():
                    self.ema.store(self.model.parameters())
                    self.ema.copy_to(self.model.parameters())

                    loss, loss_x, loss_adj = self.loss_fn(self.model, *loss_subject)
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema.restore(self.model.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                        f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ', verbose=False)

            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''

                torch.save({ 
                    'model_config': self.config,
                    'params' : self.params,
                    'state_dict': self.model.state_dict(),
                    'ema': self.ema.state_dict()
                    }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
                torch.save(self.optimizer.state_dict(),
                           f'./checkpoints/{self.config.data.data}/{self.ckpt}_optimizer.pth')
            
            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | '
                            f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}')
        print(' ')
        return self.ckpt