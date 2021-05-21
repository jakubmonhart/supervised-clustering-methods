import os
import sys
import json
import argparse
import copy

from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np

from train import train_fn
from dataset import DebugSampler, pdf_create_dataloaders, circles_create_dataloaders, mog_create_dataloaders
from utils import get_arguments, save_best, create_logger
from metrics import AverageMeter, accuracy
from cluster import cluster_iterative, cluster_spectral
from attention import AddCompat, MultiCompat, StackedSAB, StackedISAB

# W&B
import wandb

# ***** Model *****

class ABCModel(nn.Module):
  def __init__(self, input_size, hidden_size=None, n_enc_layers=2, num_heads=1, compat='multi', isab=False):
    super(ABCModel, self).__init__()

    if hidden_size is None:
        hidden_size = input_size
        
    self.fc = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
    )

#    self.enc = SAB(hidden_size, hidden_size, num_heads=num_heads, compat=compat)
#    self.enc2 = SAB(hidden_size, hidden_size, num_heads=num_heads, compat=compat)
    if isab:
      self.enc = StackedISAB(hidden_size, hidden_size, num_inds=32, num_blocks=n_enc_layers, num_heads=num_heads, compat=compat)
    else:
      self.enc = StackedSAB(hidden_size, hidden_size, num_blocks=n_enc_layers, num_heads=num_heads, compat=compat)
    

    self.fc_q = nn.Linear(hidden_size, hidden_size)
    self.fc_k = nn.Linear(hidden_size, hidden_size)
    
    if compat=='multi':
      self.compat = MultiCompat(hidden_size)
    elif compat=='add':
      self.compat = AddCompat(hidden_size)
    else:
      print(f'Compatibility function {compat} not implemented.')
      return -1

  def forward(self, X):
    X1 = self.fc(X)
    X2 = self.enc(X1)                        # (N, L, H)
    Q  = self.fc_q(X2)
    K  = self.fc_k(X2)
    E  = self.compat(Q, K)                   # (N, L, L)
    logits = 0.5 * (E + E.transpose(-2, -1)) # force symmetry
    return logits

# ***** Train functions ******
  
def create_model(args):
  model  = ABCModel(args.input_size, args.hidden_size, args.n_enc_layers, args.num_heads, args.compat, args.isab)
  optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.BCEWithLogitsLoss()
  
  return model, criterion, optimizer

  
def artif_run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl, val_dl, val_dl_cluster, eval_fn, device):
  train_loss = AverageMeter()
  model.train()

  for i, batch in enumerate(train_dl):
    X = batch['X']
    y = batch['label']

    X = X.to(device)
    A = (y.unsqueeze(-2) == y.unsqueeze(-1)).float().to(device)

    logits = model(X)
    loss = criterion(logits, A)
    loss.backward()

    train_loss.update(loss.item())

    # optimize
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
    
  early_stop_metrics_log = eval_fn(model, criterion, early_stop_dl, args, device, False)
  metrics_log['early_stop_loss'] = early_stop_metrics_log['val_loss']
  metrics_log['train_loss'] = train_loss.avg
  
  return metrics_log


def artif_eval_fn(model, criterion, val_dl, args, device, eval_clusters):
  '''
    If eval_clusters is False, only val_loss is computed.
    
    If eval_clusters is True, on top of val_loss, clusters are infered and clustering metrics computed.
  '''
  
  # Cluster method
  if args.cluster_method == 'spectral':
    cluster = cluster_spectral
  elif args.cluster_method == 'iterative':
    cluster = cluster_iterative
  else:
    logger.warning('incorrect cluster method. Should be "spectral" or "iterative".')
    return -1
  
  model.eval()
  
  ari = AverageMeter()
  nmi = AverageMeter()
  val_loss = AverageMeter()
  acc = AverageMeter()
  tpr = AverageMeter()
  tnr = AverageMeter()

  with torch.no_grad():  
    for batch in val_dl:
      X = batch['X']
      y = batch['label']

      X = X.to(device)
      A = (y.unsqueeze(-2) == y.unsqueeze(-1)).float().to(device)

      # Model output and loss
      logits = model(X)
      loss = criterion(logits, A)
      val_loss.update(loss.item())
      
      if eval_clusters:

        # Accuracy of similarity prediction
        # Confusion matrix computation inside accuracy() takes a lot of time. 
        bacc, btpr, btnr = accuracy(((logits.flatten()>0)*1).cpu().numpy(), A.flatten().cpu().numpy())
        acc.update(bacc.item())
        tpr.update(btpr.item())
        tnr.update(btnr.item())

        for sid in range(len(logits)):
          
          # Cluster
          simi_matrix_prob = torch.sigmoid(logits[sid]).cpu()
          plabels = cluster(simi_matrix_prob)

          ari.update(adjusted_rand_score(y[sid], plabels))
          nmi.update(normalized_mutual_info_score(y[sid], plabels, average_method='arithmetic'))
  
  
    if eval_clusters:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': ari.avg, 'val_nmi': nmi.avg, 'val_acc': acc.avg, 
                     'val_tpr': tpr.avg, 'val_tnr': tnr.avg, 'val_num_failures': None}
    else:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}
          
  return metrics_log


def pdf_run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl,  val_dl, val_dl_cluster, eval_fn, device):
  
  train_loss = AverageMeter()
  model.train()

  for i, batch in enumerate(train_dl):
    X = batch['X']
    y = batch['label']

    X = X.to(device).unsqueeze(0)
    A = (y.unsqueeze(-2) == y.unsqueeze(-1)).float().to(device)

    logits = model(X).squeeze(0)
    loss = criterion(logits, A)
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

  early_stop_metrics_log = eval_fn(model, criterion, early_stop_dl, args, device, False)
  metrics_log['early_stop_loss'] = early_stop_metrics_log['val_loss']
  metrics_log['train_loss'] = train_loss.avg
  
  return metrics_log
 
  
def pdf_eval_fn(model, criterion, val_dl, args, device, eval_clusters):
  
  # Cluster method
  if args.cluster_method == 'spectral':
    cluster = cluster_spectral
  elif args.cluster_method == 'iterative':
    cluster = cluster_iterative
  else:
    logger.warning('incorrect cluster method. Should be "spectral" or "iterative".')
    return -1
  
  model.eval()

  val_loss = AverageMeter()
  ari = AverageMeter()
  nmi = AverageMeter()
  acc = AverageMeter()
  tpr = AverageMeter()
  tnr = AverageMeter()

  with torch.no_grad():  
    for batch in val_dl:

      X = batch['X']
      y = batch['label']

      X = X.to(device).unsqueeze(0)
      A = (y.unsqueeze(-2) == y.unsqueeze(-1)).float().to(device)

      # Model output and loss
      logits = model(X).squeeze(0)
      loss = criterion(logits, A)
      val_loss.update(loss.item())
      
      # cluster
      if eval_clusters:
        
        # Accuracy of similarity prediction
        bacc, btpr, btnr = accuracy(((logits.flatten()>0)*1).cpu().numpy(), A.flatten().cpu().numpy())
        acc.update(bacc.item())
        tpr.update(btpr.item())
        tnr.update(btnr.item())

        # Cluster
        simi_matrix_prob = torch.sigmoid(logits).cpu()
        plabels = cluster(simi_matrix_prob)

        ari.update(adjusted_rand_score(y, plabels))
        nmi.update(normalized_mutual_info_score(y, plabels, average_method='arithmetic'))

    if eval_clusters:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': ari.avg, 'val_nmi': nmi.avg, 'val_acc': acc.avg, 
                     'val_tpr': tpr.avg, 'val_tnr': tnr.avg, 'val_num_failures': None}
    else:
      metrics_log = {'val_loss': val_loss.avg, 'val_ari': None, 'val_nmi': None, 'val_acc': None, 
                     'val_tpr': None, 'val_tnr': None, 'val_num_failures': None}

  return metrics_log


# ******* Main *******

def main():
  
  parser = argparse.ArgumentParser()
  
  # For grouping in wandb
  parser.add_argument('--model_type', type=str, default='abc')
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
  
  # Parameters
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--batch_size', type=int, default=32)  
  parser.add_argument('--hidden_size', type=int, default=128)
  parser.add_argument('--n_enc_layers', type=int, default=2)
  parser.add_argument('--num_heads', type=int, default=1)
  parser.add_argument('--compat', type=str, default='multi')
  parser.add_argument('--isab', dest='isab', action='store_true')
  parser.set_defaults(isab=False)
  parser.add_argument('--cluster_method', type=str, default='spectral')
  
  # Dataset parameters
  parser.add_argument('--dataset', type=str, default='pdf')
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
  
