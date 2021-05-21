import os, sys
import time
import json
import argparse
import copy

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from dataset import pdf_create_dataloaders, circles_create_dataloaders, mog_create_dataloaders, DebugSampler, batch_to_dac_compatible
from attention import StackedISAB, PMA, MAB, ISAB
from utils import create_logger, save_best, reset_wandb_env
from metrics import AverageMeter
from train import train_fn

# W&B
import wandb

# ***** Model ******

def anchored_cluster_loss(logits, anchor_idxs, labels):
  B = labels.shape[0]
  labels = labels.argmax(-1)
  anchor_labels = labels[torch.arange(B), anchor_idxs]
  targets = (labels == anchor_labels.unsqueeze(-1)).float()
  return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)


def sample_anchors(B, N, mask=None, device='cpu'):
  if mask is None:
    return torch.randint(0, N, [B])
  else:
    mask = mask.view(B, N)
    anchor_idxs = torch.zeros(B, dtype=torch.int64)
    for b in range(B):
      if mask[b].sum() < N:
        idx_pool = mask[b].bitwise_not().nonzero().view(-1)
        anchor_idxs[b] = idx_pool[torch.randint(len(idx_pool), [1])]
    return anchor_idxs


class DACModel(nn.Module):
  def __init__(self, args):
    super().__init__()
    
    self.args = args
    
    # self.encoder = nn.Sequential(nn.Linear(args.input_size, 64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
    self.encoder = nn.Identity()
    self.isab = StackedISAB(args.input_size, args.dim, args.num_inds, args.num_blocks, p=args.drop_p,
                            ln=args.ln, num_heads=args.num_heads)
    # Use layer norm here as well?
    self.mab = MAB(args.dim, args.dim, args.dim, ln=args.ln, p=args.drop_p, num_heads=args.num_heads)
    
    if args.use_isab2:
      self.isab2 = StackedISAB(args.dim, args.dim, args.num_inds, args.num_blocks, p=args.drop_p,
                            ln=args.ln, num_heads=args.num_heads)
    
    # Use more layers here?
    self.fc = nn.Linear(args.dim, 1)
    
  def forward(self, X, anchor_idxs, mask=None):
    
    X = self.encoder(X)
    H_enc = self.isab(X, mask=mask)
    anchors = H_enc[torch.arange(X.shape[0]), anchor_idxs].unsqueeze(1)
    H_enc = self.mab(H_enc, anchors)
    
    if self.args.use_isab2:
      o = self.fc(self.isab2(H_enc, mask=mask))
    else:
      o = self.fc(H_enc)

    return o

  def loss_fn_anchored(self, X, labels):
    # Cannot vectorize loss computation over batch due to variable size of 
    # data (variable number of textboxes in each page).

    anchor_idxs = sample_anchors(X.shape[0], X.shape[1]).to(X.device)

    logits = self.forward(X, anchor_idxs)
    loss = anchored_cluster_loss(logits, anchor_idxs, labels)

    # Return mean of loss
    return loss

  def cluster_anchored(self, X, max_iter=50, verbose=True, check=False):
    # TODO - leave 0 label for unclustered instances

    B, N = X.shape[0], X.shape[1]
    self.eval()

    with torch.no_grad():
      anchor_idxs = sample_anchors(B, N)
      logits = self.forward(X, anchor_idxs)
#       labels = torch.zeros_like(logits).squeeze(-1).int()
      mask = (logits > 0.0)
      done = mask.sum((1,2)) == N

      labels = mask.squeeze(-1).long()

      for i in range(2, max_iter+1):
        anchor_idxs = sample_anchors(B, N, mask=mask)
        logits = self.forward(X, anchor_idxs, mask=mask)
        ind = logits > 0.0
        labels[(ind*mask.bitwise_not()).squeeze(-1)] = i
        mask[ind] = True

        num_processed = mask.sum((1,2))
        done = num_processed == N
        if verbose:
            print(num_processed)
        if done.sum() == B:
           break

    fail = done.sum() < B

    if check:
        return None, labels, torch.zeros(1), fail
    else:
        return None, labels, torch.zeros(1)
      
      
# ***** Train functions ******  
  
def create_model(args):
  
  model = DACModel(args)
  optimizer = torch.optim.Adam(model.parameters(), args.lr)
  
  return model, model.loss_fn_anchored, optimizer
  

def artif_run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl, val_dl, val_dl_cluster, eval_fn, device):
  train_loss = AverageMeter()
  model.train()

  for i, batch in enumerate(train_dl):

    # Convert labels to be compatible with DAC model
    batch['label'] = F.one_hot(batch['label'])

    # Forward      
    loss = model.loss_fn_anchored(batch['X'].to(device), batch['label'].to(device))

    # Backward
    loss.backward()

    train_loss.update(loss.item())

    # Optimize
    optimizer.step()
    optimizer.zero_grad() 

  # Eval
  metrics_log = {'val_loss': None, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
  if args.eval_during_training:
    if ((t+1)%args.eval_freq==0) or (t==0):
      metrics_log = eval_fn(model, criterion, val_dl_cluster, args, device, True)
    else:
      metrics_log = eval_fn(model, criterion, val_dl, args, device, False)
    
  # Compute early stop loss
  early_stop_metrics_log = eval_fn(model, criterion, early_stop_dl, args, device, False)
  
  metrics_log['early_stop_loss'] = early_stop_metrics_log['val_loss']
  metrics_log['train_loss'] = train_loss.avg
  
  return metrics_log


def pdf_run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl, val_dl, val_dl_cluster, eval_fn, device):
  train_loss = AverageMeter()
  model.train()

  for i, batch in enumerate(train_dl):

    # Convert batch to be compatible with DAC model
    batch = batch_to_dac_compatible(batch, args.augment_pdf_data)

    # Forward
    loss = model.loss_fn_anchored(batch['X'].to(device), batch['label'].to(device))

    # Backward
    loss.backward()

    train_loss.update(loss.item())

    # optimize
    if (i+1)%args.batch_size == 0:
      optimizer.step()
      optimizer.zero_grad()
      
  # Last batch might not be whole
  optimizer.step()
  optimizer.zero_grad()
  
  # Eval
  metrics_log = {'val_loss': None, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
  if args.eval_during_training:
    if ((t+1)%args.eval_freq==0) or (t==0):
      metrics_log = eval_fn(model, criterion, val_dl_cluster, args, device, True)
    else:
      metrics_log = eval_fn(model, criterion, val_dl, args, device, False)

  # Compute early stop loss
  early_stop_metrics_log = eval_fn(model, criterion, early_stop_dl, args, device, False)
  
  metrics_log['early_stop_loss'] = early_stop_metrics_log['val_loss']
  metrics_log['train_loss'] = train_loss.avg
  
  return metrics_log


def artif_eval_fn(model, criterion, val_dl, args, device, eval_clusters):
  
  with torch.no_grad():
      
    ari = AverageMeter()
    nmi = AverageMeter()
    val_loss = AverageMeter()
    num_failures = AverageMeter()

    model.eval()
      
    for batch in val_dl:

      # Convert labels to be compatible with DAC model
      batch['label'] = F.one_hot(batch['label'])

      loss = model.loss_fn_anchored(batch['X'].to(device), batch['label'].to(device))
      val_loss.update(loss.item())
      
      # Cluster
      if eval_clusters:
        params, labels, ll, fail = model.cluster_anchored(batch['X'].to(device), max_iter=args.max_iter, verbose=False, check=True)

        labels = labels.cpu().numpy()
        true_labels = batch['label'].argmax(-1).numpy()

        for sid in range(len(labels)):
          ari.update(adjusted_rand_score(true_labels[sid], labels[sid]))
          nmi.update(normalized_mutual_info_score(true_labels[sid], labels[sid], average_method='arithmetic'))

        num_failures.update(int(fail))
        
    if eval_clusters:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': ari.avg, 'val_nmi': nmi.avg, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': num_failures.avg}
    else:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
  
  return metrics_log


def pdf_eval_fn(model, criterion, val_dl, args, device, eval_clusters):
  
  with torch.no_grad():

    ari = AverageMeter()
    nmi = AverageMeter()
    val_loss = AverageMeter()
    num_failures = AverageMeter()

    model.eval()

    for batch in val_dl:
      # Convert batch to be compatible with DAC model
      batch = batch_to_dac_compatible(batch)

      loss = model.loss_fn_anchored(batch['X'].to(device), batch['label'].to(device))
      val_loss.update(loss.item())

      # Cluster
      if eval_clusters:
        params, labels, ll, fail = model.cluster_anchored(batch['X'].to(device), max_iter=args.max_iter, verbose=False, check=True)

        labels = labels[0].cpu().numpy()
        true_labels = batch['label'][0].argmax(-1).numpy()

        ari.update(adjusted_rand_score(true_labels, labels))
        nmi.update(normalized_mutual_info_score(true_labels, labels, average_method='arithmetic'))
        num_failures.update(int(fail))
          
    if eval_clusters:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': ari.avg, 'val_nmi': nmi.avg, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': num_failures.avg}
    else:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}

  return metrics_log


# ******** Main *******

def main():
  
  parser = argparse.ArgumentParser()
  
  # For grouping in wandb
  parser.add_argument('--model_type', type=str, default='dac')
  parser.add_argument('--experiment', type=str, default='basic')
  parser.add_argument('--job', type=str, default='train')
  parser.add_argument('--save_dir', type=str, default='../runs/debug')
  
  # Run parameters
  parser.add_argument('--n_epochs', type=int, default=10000)
  parser.add_argument('--print_freq', type=int, default=1)
  parser.add_argument('--eval_freq', type=int, default=10)
  parser.add_argument('--early_stop', type=int, default=10)
  parser.add_argument('--n_folds', type=int, default=0)
  parser.add_argument('--n_folds_done', type=int, default=0)
  parser.add_argument('--save_model', dest='save_model', action='store_true')
  parser.set_defaults(save_model=False)
  parser.add_argument('--test', dest='test', action='store_true')
  parser.set_defaults(test=False)
  parser.add_argument('--no_eval', dest='eval_during_training', action='store_false')
  parser.set_defaults(eval_during_training=True)
  
  # Model parameters
  parser.add_argument('--lr', type=float, default=1e-04)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--dim', type=int, default=256)
  parser.add_argument('--num_blocks', type=int, default=8)
  parser.add_argument('--num_inds', type=int, default=32)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--no_isab2', dest='use_isab2', action='store_false')
  parser.set_defaults(use_isab2=True)
  parser.add_argument('--drop_p', type=float, default=0)
  parser.add_argument('--layer_norm', dest='ln', action='store_true')
  parser.set_defaults(ln=False)
  parser.add_argument('--max_iter', type=int, default=30)
  
  # Dataset parameters
  parser.add_argument('--dataset', type=str, default='pdf')
  parser.add_argument('--debug_dataset', type=int, default=0)
  
  args, _ = parser.parse_known_args()
  
  if args.dataset == 'circles':
    parser.add_argument('--n_sets_train', type=int, default=100000)
    parser.add_argument('--n_sets_val', type=int, default=1000)
    parser.add_argument('--n_elements', type=int, default=100)
    parser.add_argument('--n_clusters_low', type=int, default=2)
    parser.add_argument('--n_clusters_high', type=int, default=6)
  
  elif args.dataset == 'mog':
    parser.add_argument('--n_sets_train', type=int, default=100000)
    parser.add_argument('--n_sets_val', type=int, default=1000)
    parser.add_argument('--n_elements', type=int, default=100)
    parser.add_argument('--n_clusters_low', type=int, default=2)
    parser.add_argument('--n_clusters_high', type=int, default=6)
  
  else:
    pass
  
  parser.add_argument('--circles_dont_shuffle', dest='circles_shuffle', action='store_false')
  parser.set_defaults(circles_shuffle=True)
  parser.add_argument('--augment_pdf_data', dest='augment_pdf_data', action='store_true')
  parser.set_defaults(augment_pdf_data=False)
  
  args, _ = parser.parse_known_args()
  
  if args.drop_p == 0:
    args.drop_p = None
  
  if args.n_folds:
    
    for i in range(args.n_folds_done, args.n_folds):
      args_copy = copy.deepcopy(args)
      args_copy.fold_number = i
      
      if args.dataset == 'mog':
        args_copy.input_size=2
        train_fn(args_copy, artif_run_epoch, artif_eval_fn, create_model, mog_create_dataloaders)
      elif args.dataset == 'circles':
        args_copy.input_size = 2
        train_fn(args_copy, artif_run_epoch, artif_eval_fn, create_model, circles_create_dataloaders)
      elif args.dataset == 'pdf':
        args_copy.input_size = 6
        train_fn(args_copy, pdf_run_epoch, pdf_eval_fn, create_model, pdf_create_dataloaders)
      else:
        print('Incorrect dataset')
        return -1    
      
  else:
    
    if args.dataset == 'mog':
      args.input_size = 2
      train_fn(args, artif_run_epoch, artif_eval_fn, create_model, mog_create_dataloaders)
    elif args.dataset == 'circles':
      args.input_size = 2
      train_fn(args, artif_run_epoch, artif_eval_fn, create_model, circles_create_dataloaders)
    elif args.dataset == 'pdf':
      args.input_size = 6
      train_fn(args, pdf_run_epoch, pdf_eval_fn, create_model, pdf_create_dataloaders)
    else:
      print('Incorrect dataset')
      return -1    
  
if __name__=="__main__":
  main()
