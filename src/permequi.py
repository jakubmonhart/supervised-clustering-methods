import os
import argparse
import json
import random
import copy

import torch
from torch import nn
from torch.utils import data
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from dataset import DebugSampler, batch_to_same_cluster, circles_create_dataloaders, pdf_create_dataloaders, mog_create_dataloaders
from utils import create_logger, save_best
from metrics import AverageMeter, accuracy
from cluster import cluster_spectral, cluster_iterative
from train import train_fn

# W&B
import wandb


# ******* Model *******

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
  

class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    
#    if mask is not None:
#      x_ = x.view(1, -1, x.shape[-1])
#      mask_ = mask.view(1, -1)
#      x_[mask_] = 0
      
    xm, _ = x.max(1, keepdim=True) # Max with dim 1 works if inputs is of shape (batch_size, set_size, input_size)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x 
  

class PermEqui2_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    
   # if mask is not None:
   #   x_ = x.view(1, -1, x.shape[-1])
   #   mask_ = mask.view(1, -1)
   #   x_[mask_] = 0
   #   
   #   x_sum = x.sum(1, keepdim=True)
   #   x_len = (mask == False).sum(1, keepdim=True)
   #   
   #   xm = x_sum/x_len
   #   
   # else:
   #   xm = x.mean(1, keepdim=True) # Mean with dim 1 works if inputs is of shape (batch_size, set_size, input_size)
      
    xm = x.mean(1, keepdim=True) # Mean with dim 1 works if inputs is of shape (batch_size, set_size, input_size)

    xm = self.Lambda(xm)  
    x = self.Gamma(x)
    x = x - xm
    return x 


class PermEquiModel(nn.Module):
  def __init__(self, args):
    super().__init__()
    
    self.anchor_embed_method = args.anchor_embed_method
    
    # Choose mean or max permequi layer
    if args.pme_version == 'mean':
      PermEqui2 = PermEqui2_mean
    elif args.pme_version == 'max':
      PermEqui2 = PermEqui2_max
    else:
      print('Incorrect args.pme_version setting')
      return -1
    
    if args.pme_cluster == 'iterative':
      self.infer_clusters = self.cluster_iterative
    elif args.pme_cluster == 'spectral':
      self.infer_clusters = self.cluster_similarity
    else:
      print('Incorrect args.pme_cluster setting')
      return -1
    
    encoder_layers = [PermEqui2(args.in_dim, args.hid_dim), nn.ELU(inplace=True)]
    
    for i in range(args.encoder_n_layers-1):
      encoder_layers += [PermEqui2(args.hid_dim, args.hid_dim), nn.ELU(inplace=True)]
    
    aencoder_layers = [nn.Linear(args.in_dim, args.hid_dim), nn.ReLU(inplace=True)]
    
    for i in range(args.aencoder_n_layers-1):
      aencoder_layers += [nn.Linear(args.hid_dim, args.hid_dim), nn.ReLU(inplace=True)]
    decoder_layers = []
    
    if self.anchor_embed_method == 'cat':
      hid_dim = 2*args.hid_dim
    else:
      hid_dim = args.hid_dim
    
    for i in range(args.decoder_n_layers):
      decoder_layers += [PermEqui2(hid_dim, hid_dim), nn.ELU(inplace=True)]

    decoder_layers += [nn.Linear(hid_dim, args.out_dim)]
    
    self.encoder = nn.Sequential(*encoder_layers)
    self.anchor_encoder = nn.Sequential(*aencoder_layers)
    self.decoder = nn.Sequential(*decoder_layers)
    
  def forward(self, X, anchor_idxs):
    
    anchors = X[torch.arange(X.shape[0]), anchor_idxs].unsqueeze(1)
    enc = self.encoder(X)
    anch_enc = self.anchor_encoder(anchors)
  
    return self.decoder_forward(enc, anch_enc)
  
  def decoder_forward(self, enc, anch_enc):
    
    if self.anchor_embed_method == 'sum':
      h = enc + anch_enc
    elif self.anchor_embed_method == 'cat':
      h = torch.cat([enc, anch_enc.repeat(1, enc.shape[1], 1)], dim=2)
    elif self.anchor_embed_method == 'max':
      h = torch.cat([enc.unsqueeze(2), anch_enc.repeat(1, enc.shape[1], 1).unsqueeze(2)], dim=2)
      h, _ = torch.max(h, dim=2)
    
    return self.decoder(h) 
    
  
  def cluster_iterative(self, X, max_iter=20, verbose=False, check=False):

    # Precompute
    enc = self.encoder(X)
    all_anch_enc = self.anchor_encoder(X)
    
    B, N = X.shape[0], X.shape[1]
    self.eval()

    with torch.no_grad():
      anchor_idxs = sample_anchors(B, N)
      anch_enc = all_anch_enc[torch.arange(B), anchor_idxs].unsqueeze(1)
      logits = self.decoder_forward(enc, anch_enc)
      mask = (logits > 0.0)
      done = mask.sum((1,2)) == N

      labels = mask.squeeze(-1).long()

      for i in range(2, max_iter+1):
        anchor_idxs = sample_anchors(B, N, mask=mask)
        anch_enc = all_anch_enc[torch.arange(B), anchor_idxs].unsqueeze(1)
        logits = self.decoder_forward(enc, anch_enc, mask=mask)
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
        return labels, fail
    else:
        return labels   
  
  def cluster_similarity(self, X):
    B, N = X.shape[0], X.shape[1]
    enc = self.encoder(X)
    enc = enc.unsqueeze(1).repeat(1, N, 1, 1)
    enc = enc.reshape(-1, N, enc.shape[-1])

    anch_enc = self.anchor_encoder(X)
    anch_enc = anch_enc.reshape(-1, anch_enc.shape[-1]).unsqueeze(1)
    
#     print('reshapes done')

    logits = self.decoder_forward(enc, anch_enc)
    logits = logits.reshape(B, N, N)
    
#     print('decoder computed')

    plabel = torch.empty(B, N).int()

    for sid in range(len(logits)):
#       print('spectral')
      simi_matrix = logits[sid]
      simi_matrix = 0.5*(simi_matrix + simi_matrix.transpose(-2,-1))
      simi_matrix_prob = torch.sigmoid(simi_matrix).detach()
      plabel[sid] = cluster_spectral(simi_matrix_prob)
      
    return plabel, 0  
  
  
# ****** Train functions *******

def artif_eval_fn(model, criterion, val_dl, args, device, eval_clusters):
  '''
    If eval_clusters is True, val_dl must return batch of size 1. For validation loss
    only, batch size can be arbitrary.
  '''
  
  with torch.no_grad():
    
    val_loss = AverageMeter()
    
    if eval_clusters:
      ari = AverageMeter()
      nmi = AverageMeter()
      num_failures = AverageMeter()

    model.eval()

    for batch in val_dl:
      
      B, N = batch['X'].shape[0], batch['X'].shape[1]
      anchor_idxs = sample_anchors(B, N)
      anchor_labels = batch['label'][torch.arange(B), anchor_idxs].unsqueeze(1)
      target = (batch['label'] == anchor_labels).float().to(device)
    
      # Forward
      logits = model(batch['X'].to(device), anchor_idxs)
      loss = criterion(logits.squeeze(-1), target)

      val_loss.update(loss.item())

      # cluster
      if eval_clusters:
        
        plabel, fail = model.infer_clusters(batch['X'].to(device))
        plabel = plabel.cpu()
        
        for sid in range(B):
          ari.update(adjusted_rand_score(batch['label'][sid], plabel[sid]))
          nmi.update(normalized_mutual_info_score(batch['label'][sid], plabel[sid], average_method='arithmetic'))
        
        num_failures.update(int(fail))
  
    if eval_clusters:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': ari.avg, 'val_nmi': nmi.avg, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': num_failures.avg}
    else:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
  
  return metrics_log


def artif_run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl, val_dl, val_dl_cluster, eval_fn, device):
  
  # Train
  train_loss = AverageMeter()
  model.train()

  for batch in train_dl:
    B, N = batch['X'].shape[0], batch['X'].shape[1]
    anchor_idxs = sample_anchors(B, N)
    anchor_labels = batch['label'][torch.arange(B), anchor_idxs].unsqueeze(1)
    target = (batch['label'] == anchor_labels).float().to(device)
  
    # Forward
    logits = model(batch['X'].to(device), anchor_idxs)
    loss = criterion(logits.squeeze(-1), target)

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


def pdf_eval_fn(model, criterion, val_dl, args, device, eval_clusters):

  with torch.no_grad():
    
    val_loss = AverageMeter()
    ari = AverageMeter()
    nmi = AverageMeter()
    num_failures = AverageMeter()

    model.eval()

    for batch in val_dl:
      
      batch['X'] = batch['X'].unsqueeze(0)
      batch['label'] = batch['label'].unsqueeze(0)
    
      B, N = batch['X'].shape[0], batch['X'].shape[1]
      anchor_idxs = sample_anchors(B, N)
      anchor_labels = batch['label'][torch.arange(B), anchor_idxs].unsqueeze(1)
      target = (batch['label'] == anchor_labels).float().to(device)
      
      # Forward
      logits = model(batch['X'].to(device), anchor_idxs)
      loss = criterion(logits.squeeze(-1), target)

      val_loss.update(loss.item())

      # cluster
      if eval_clusters:
        plabel, fail = model.infer_clusters(batch['X'].to(device))
        plabel = plabel.cpu()
        
        for sid in range(B):
          ari.update(adjusted_rand_score(batch['label'][sid], plabel[sid]))
          nmi.update(normalized_mutual_info_score(batch['label'][sid], plabel[sid], average_method='arithmetic'))
        
        num_failures.update(int(fail))
  
    if eval_clusters:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': ari.avg, 'val_nmi': nmi.avg, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
    else:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
  
  return metrics_log


def pdf_run_epoch(t, args, model, criterion, optimizer, train_dl, val_dl, early_stop_dl, val_dl_cluster, eval_fn, device):
  
  # Train
  train_loss = AverageMeter()
  model.train()

  for i, batch in enumerate(train_dl):
    batch['X'] = batch['X'].unsqueeze(0)
    batch['label'] = batch['label'].unsqueeze(0)
    
    B, N = batch['X'].shape[0], batch['X'].shape[1]
    anchor_idxs = sample_anchors(B, N)
    anchor_labels = batch['label'][torch.arange(B), anchor_idxs].unsqueeze(1)
    target = (batch['label'] == anchor_labels).float().to(device)

    # Forward
    logits = model(batch['X'].to(device), anchor_idxs)
    loss = criterion(logits.squeeze(-1), target)

    # Backward
    loss.backward()
    train_loss.update(loss.item())

    # Optimize
    if (i+1)%args.batch_size == 0:
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


def create_model(args):
  model = PermEquiModel(args)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), args.lr)
        
  return model, criterion, optimizer

# ********** Main ***********

def main():
  
  parser = argparse.ArgumentParser()
  
  # For grouping in wandb
  parser.add_argument('--model_type', type=str, default='permequi')
  parser.add_argument('--experiment', type=str, default='basic')
  parser.add_argument('--job', type=str, default='train')
  parser.add_argument('--save_dir', type=str, default='../runs/debug')
  
  # Run parameters
  parser.add_argument('--n_epochs', type=int, default=10000)
  parser.add_argument('--early_stop', type=int, default=20) # There tend to be some oscillation in validation loss after 100th epoch or so.., need to keep early stop a little bit larger
  parser.add_argument('--print_freq', type=int, default=1)
  parser.add_argument('--eval_freq', type=int, default=10)
  parser.add_argument('--n_folds', type=int, default=0)
  parser.add_argument('--n_folds_done', type=int, default=0)
  parser.add_argument('--save_model', dest='save_model', action='store_true')
  parser.set_defaults(save_model=False)
  parser.add_argument('--test', dest='test', action='store_true')
  parser.set_defaults(test=False)
  parser.add_argument('--no_eval', dest='eval_during_training', action='store_false')
  parser.set_defaults(eval_during_training=True)
  
  # Model parameters
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--pme_version', type=str, default='max')
  parser.add_argument('--encoder_n_layers', type=int, default=3)
  parser.add_argument('--aencoder_n_layers', type=int, default=3)
  parser.add_argument('--decoder_n_layers', type=int, default=3)
  parser.add_argument('--hid_dim', type=int, default=256)
  parser.add_argument('--anchor_embed_method', type=str, default='cat')
  parser.add_argument('--pme_cluster', type=str, default='spectral')
 
  # Dataset parameters
  parser.add_argument('--dataset', type=str, default='circles')
  parser.add_argument('--debug_dataset', type=int, default=0)
  parser.add_argument('--n_sets_train', type=int, default=100000)
  parser.add_argument('--n_sets_val', type=int, default=1000)
  parser.add_argument('--n_elements', type=int, default=100)
  parser.add_argument('--n_clusters_low', type=int, default=2)
  parser.add_argument('--n_clusters_high', type=int, default=6)
  parser.add_argument('--circles_dont_shuffle', dest='circles_shuffle', action='store_false')
  parser.set_defaults(circles_shuffle=True)
  
  args, _ = parser.parse_known_args()
      
  if args.n_folds:
    for i in range(args.n_folds_done, args.n_folds):
      args_copy = copy.deepcopy(args)
      args_copy.fold_number = i
      
      if args.dataset == 'mog':
        args_copy.in_dim = 2
        args_copy.out_dim = 1
        train_fn(args_copy, artif_run_epoch, artif_eval_fn, create_model, mog_create_dataloaders)
      elif args.dataset == 'circles':
        args_copy.in_dim = 2
        args_copy.out_dim = 1
        train_fn(args_copy, artif_run_epoch, artif_eval_fn, create_model, circles_create_dataloaders)
      elif args.dataset == 'pdf':
        args_copy.in_dim = 6
        args_copy.out_dim = 1
        train_fn(args_copy, pdf_run_epoch, pdf_eval_fn, create_model, pdf_create_dataloaders)
      else:
        print('Incorrect dataset')
        return -1    
      
  else:
    
    if args.dataset == 'mog':
      args.in_dim = 2
      args.out_dim = 1
      train_fn(args, artif_run_epoch, artif_eval_fn, create_model, mog_create_dataloaders)
    elif args.dataset == 'circles':
      args.in_dim = 2
      args.out_dim = 1
      train_fn(args, artif_run_epoch, artif_eval_fn, create_model, circles_create_dataloaders)
    elif args.dataset == 'pdf':
      args.in_dim = 6
      args.out_dim = 1
      train_fn(args, pdf_run_epoch, pdf_eval_fn, create_model, pdf_create_dataloaders)
    else:
      print('Incorrect dataset')
      return -1    
    
    
if __name__=="__main__":
  main()
