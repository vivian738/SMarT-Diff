import torch
import argparse
import time
from trainer import Trainer
from sampler import Sampler, Sampler_mol
import yaml
from easydict import EasyDict as edict

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




def main(work_type):
    ts = time.strftime('%b%d-%H_%M_%S', time.gmtime())

    config = get_config()

    # -------- Train --------
    if work_type == 'train':
        trainer = Trainer(config) 
        ckpt = trainer.train(ts, finetune=False)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler_mol(config)
            sampler.sample()

    # -------- Generation --------
    elif work_type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k','ChEMBL']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
        
    else:
        raise ValueError(f'Wrong type : {work_type}')

if __name__ == '__main__':

    main(work_type='sample')
