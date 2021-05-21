import argparse
import logging
import os
import sys

import torch

def reset_wandb_env():
  exclude = {
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_API_KEY",
  }
  for k, v in os.environ.items():
    if k.startswith("WANDB_") and k not in exclude:
      del os.environ[k]


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='trial')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--save_model', type=bool, default=True)
    args, _ = parser.parse_known_args()
    return args


def create_logger(save_dir):
    log_path = os.path.join(save_dir, 'out.log')
    logger = logging.getLogger('{}'.format(save_dir))
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Handlers might be already added to logger (when using jupyter)
    if len(logger.handlers) == 0:
        logger.addHandler(fh) 
        logger.addHandler(ch)

    return logger


def save_best(args, epoch, model, metrics_log):
    state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'args': args,
            'metrics_log': metrics_log
            }

    filename = os.path.join(args.save_dir, 'checkpoint.pt.tar')
    torch.save(state, filename)

    
def reset_wandb_env():
  exclude = {
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_API_KEY",
  }
  
  for k, v in os.environ.items():
    if k.startswith("WANDB_") and k not in exclude:
      del os.environ[k]
