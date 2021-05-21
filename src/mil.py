import sys, os
import argparse
import json
import copy

from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import torch
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import create_logger, save_best
from dataset import PdfDataset, CirclesDataset, DebugSampler, batch_to_mil_pair_indicators, batch_to_mil_pairs, pdf_create_dataloaders, circles_create_dataloaders, mog_create_dataloaders
from metrics import AverageMeter, accuracy
from attention import PMA
from cluster import cluster_spectral, cluster_iterative
from train import train_fn

# W&B
import wandb


# ****** Model ******
class BagModel(nn.Module):
  '''
  Model for solving MIL problems

  Args:
      prepNN: neural network created by user processing input before aggregation function (subclass of torch.nn.Module)
      afterNN: neural network created by user processing output of aggregation function and outputing final output of BagModel (subclass of torch.nn.Module)
      aggregation_func: mil.max and mil.mean supported, any aggregation function with argument 'dim' and same behaviour as torch.mean can be used

  Returns:
      Output of forward function.
  '''

  def __init__(self, prepNN, afterNN, aggregation_func):
    super().__init__()

    self.prepNN = prepNN
    self.aggregation_func = aggregation_func
    self.afterNN = afterNN

  def forward(self, input):    
    ids = input[1]
    input = input[0]

    # Modify shape of bagids if only 1d tensor
    if (ids.dim() == 1):
      ids = ids.unsqueeze(0)

    inner_ids = ids[-1]
    device = input.device
    NN_out = self.prepNN(input)
    unique_ids = torch.unique(inner_ids)
    output = torch.empty((len(unique_ids), len(NN_out[0])), device = device)
    
    for i, bid in enumerate(unique_ids):
      mask = (inner_ids == bid)
      output[i] = self.aggregation_func(NN_out[mask], dim=0)

    output = self.afterNN(output)

    if (ids.shape[0] == 1):
      return output
    else:
      new_ids = torch.cat([ids[:,torch.nonzero((inner_ids == i), as_tuple=False)[0]] for i in unique_ids], dim=1)
      return (output, new_ids[:-1])


class BagModelPrepNN(nn.Module):
  def __init__(self, args):
    super().__init__()

    if args.prep_n_layers == 1:
      layers = [nn.Linear(args.in_dim, args.hid_dim), nn.ReLU(inplace=True)]
    else:
      layers = [nn.Linear(args.in_dim, args.hid_dim),
                nn.ReLU(inplace=True)]

      for l in range(args.prep_n_layers-2):
        layers += [nn.Linear(args.hid_dim, args.hid_dim)]
        layers += [nn.ReLU(inplace=True)]

      layers += [nn.Linear(args.hid_dim, args.hid_dim), nn.ReLU(inplace=True)]

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class BagModelAfterNN(nn.Module):
  def __init__(self, args):
    super().__init__()

    if args.after_n_layers == 1:
      layers = [nn.Linear(args.hid_dim, args.out_dim)]
    else:
      layers = [nn.Linear(args.hid_dim, args.hid_dim),
                nn.ReLU(inplace=True)]

      for l in range(args.after_n_layers-2):
        layers += [nn.Linear(args.hid_dim, args.hid_dim)]
        layers += [nn.ReLU(inplace=True)]

      layers += [nn.Linear(args.hid_dim, args.out_dim)]

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)
  

class AttentionPool(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    
    self.pma = PMA(dim_X=dim_in, dim=dim_out, num_inds=1)
  
  def forward(self, x, dim = 0):
    return self.pma(x.unsqueeze(0))
      
    
# ***** Clustering helper functions ******

def probs2similarity(probs, n_elements):
  A = torch.zeros(n_elements, n_elements)
  start = 0
  end = n_elements
  for i in range(n_elements):
    A[:,i] = torch.cat([torch.zeros(i), probs[start:end]])
    start = end
    end += n_elements-(i+1)
  
  return A + torch.t(A) - A*torch.eye(n_elements)


# ****** Train functions ******

def eval_fn(model, criterion, val_dl, args, device, eval_clusters):
  '''
    If eval_clusters is False, only val_loss is computed, val_dl (dataloader with validation data) can load batch with more than one set. For loss computation,
    not all pairs need to be sampled, number of sampled pairs is set with arg.n_pairs (same way as with training phase).
    
    If eval_clusters is True, on top of val_loss, clusters are infered and clustering metrics computed. For cluster inference to work,
    val_dl needs to return only one set in each batch at a time.
  '''
  
  # Cluster method
  if args.cluster_method == 'spectral':
    cluster = cluster_spectral
  elif args.cluster_method == 'iterative':
    cluster = cluster_iterative
  else:
    logger.warning('incorrect cluster method. Should be "spectral" or "iterative".')
    return -1
  
  if eval_clusters:
    n_pairs = 0
    pair_bag_len = 0
  else:
    n_pairs = args.n_pairs
    pair_bag_len = args.pair_bag_len
  
  pairs_replacement = True
  
  with torch.no_grad():

    val_loss = AverageMeter()
    
    ari = AverageMeter()
    nmi = AverageMeter()
    acc = AverageMeter()
    tpr = AverageMeter()
    tnr = AverageMeter()

    model.eval()

    for i, batch in enumerate(val_dl):
#       print(f'eval batch: {i}/{len(val_dl)}')
      
      # Save for clustering
      y = batch['label']
      n_elements = len(batch['X'])

      # Convert batch to bags
      if args.pair_indicators:
        batch = batch_to_mil_pair_indicators(batch, n_pairs, pairs_replacement)
      else:
        batch = batch_to_mil_pairs(batch, n_pairs, args.stratify, pair_bag_len)
        
      logits = model((batch['X'].to(device), batch['B'].to(device)))
      val_loss.update(criterion(logits, batch['T'].to(device)).item())

      # Compute cluster a similarity metrics only once per eval_freq
      if eval_clusters:
        
        bacc, btpr, btnr = accuracy(logits.argmax(-1).cpu().numpy(), batch['T'].numpy())
        acc.update(bacc.item())
        tpr.update(btpr.item())
        tnr.update(btnr.item())

        # Cluster
        probs = F.softmax(logits, dim=1)[:,1].cpu()
        simi_matrix_prob = probs2similarity(probs, n_elements)

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


def run_epoch(t, args, model, criterion, optimizer, train_dl, early_stop_dl, val_dl, val_dl_cluster, eval_fn, device):
  
  train_loss = AverageMeter()
  model.train()

  for i, batch in enumerate(train_dl):
    
#     print(f'train batch: {i}/{len(train_dl)}')
    # Convert batch to bags
    if args.pair_indicators:
      batch = batch_to_mil_pair_indicators(batch, args.n_pairs, args.train_pairs_replacement)
    else:
      batch = batch_to_mil_pairs(batch, args.n_pairs, args.stratify, args.pair_bag_len)

    # Forward
    logits = model((batch['X'].to(device), batch['B'].to(device)))
    loss = criterion(logits, batch['T'].to(device))

    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_loss.update(loss.item())

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
  # Create pre- and after- aggregation layers
  prepNN = BagModelPrepNN(args)
  afterNN = BagModelAfterNN(args)

  # Choose pooling method
  if args.pool_method == 'attention': 
    pool = AttentionPool(args.hid_dim, args.hid_dim)
  elif args.pool_method == 'mean':
    pool = torch.mean
  else:
    return -1
  
  # Create model, criterion and optimizer
  model = BagModel(prepNN, afterNN, pool)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), args.lr)
  
  return model, criterion, optimizer


# ********** Main **********

def main():
  
  parser = argparse.ArgumentParser()
  
  # For grouping in wandb
  parser.add_argument('--model_type', type=str, default='mil')
  parser.add_argument('--experiment', type=str, default='basic')
  parser.add_argument('--job', type=str, default='train')
  parser.add_argument('--save_dir', type=str, default='../runs/debug')
  
  # Run parameters
  parser.add_argument('--n_epochs', type=int, default=10000)
  parser.add_argument('--early_stop', type=int, default=10)
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
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--prep_n_layers', type=int, default=3)
  parser.add_argument('--after_n_layers', type=int, default=3)
  parser.add_argument('--hid_dim', type=int, default=256)
  parser.add_argument('--cluster_method', type=str, default='spectral')
  parser.add_argument('--pool_method', type=str, default='mean')
  
  # Dataset parameters
  parser.add_argument('--dataset', type=str, default='pdf')
  parser.add_argument('--debug_dataset', type=int, default=0)
  parser.add_argument('--n_sets_train', type=int, default=10000)
  parser.add_argument('--n_sets_val', type=int, default=1000)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--n_elements', type=int, default=100)
  parser.add_argument('--n_clusters_low', type=int, default=2)
  parser.add_argument('--n_clusters_high', type=int, default=6)
  parser.add_argument('--circles_dont_shuffle', dest='circles_shuffle', action='store_false')
  parser.set_defaults(circles_shuffle=True)
  
  # Model specific dataset parameters
  parser.add_argument('--pair_indicators', dest='pair_indicators', action='store_true')
  parser.set_defaults(pair_indicators=False)
  parser.add_argument('--stratify', dest='stratify', action='store_true')
  parser.set_defaults(stratify=False)
  parser.add_argument('--n_pairs', type=int, default=50)
  parser.add_argument('--pair_bag_len', type=int, default=0)
  
  args, _ = parser.parse_known_args()
  
  args.out_dim = 2
  
  if args.n_folds:
    for i in range(args.n_folds_done, args.n_folds):
      args_copy = copy.deepcopy(args)
      args_copy.fold_number = i
      
      if args.dataset == 'pdf':
        if args.pair_indicators:
          args_copy.in_dim = 7
        else:
          args_copy.in_dim = 18
        # Run training
        train_fn(args_copy, run_epoch, eval_fn, create_model, pdf_create_dataloaders)
      elif args.dataset == 'circles':
        if args.pair_indicators:
          args_copy.in_dim = 3
        else:
          args_copy.in_dim = 6
        # Run training
        train_fn(args_copy, run_epoch, eval_fn, create_model, circles_create_dataloaders)
      elif args.dataset == 'mog':
        if args.pair_indicators:
          args_copy.in_dim = 3
        else:
          args_copy.in_dim = 6
        # Run training
        train_fn(args_copy, run_epoch, eval_fn, create_model, mog_create_dataloaders)
        
    
  else:
    if args.dataset == 'pdf':
      if args.pair_indicators:
        args.in_dim = 7
      else:
        args.in_dim = 18
      # Run training
      train_fn(args, run_epoch, eval_fn, create_model, pdf_create_dataloaders)
    elif args.dataset == 'circles':
      if args.pair_indicators:
        args.in_dim = 3
      else:
        args.in_dim = 6
      # Run training
      train_fn(args, run_epoch, eval_fn, create_model, circles_create_dataloaders)
    elif args.dataset == 'mog':
      if args.pair_indicators:
        args.in_dim = 3
      else:
        args.in_dim = 6
      # Run training
      train_fn(args, run_epoch, eval_fn, create_model, mog_create_dataloaders)


if __name__=="__main__":
    main()

