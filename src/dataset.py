import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torch.utils import data
from torch.distributions import Dirichlet, Categorical

import pandas as pd
import numpy as np

import random
import itertools

import math

# **** PDF dataset ****
class PdfDataset(Dataset):
  def __init__(self, filepath):
    cols = ['pid', 'nid', 'x0', 'y0', 'x1', 'y1', 'fsize', 'clen']
    df = pd.read_csv(filepath)[cols]

    # Grouped to pages
    self.dfs = list(df.groupby('pid'))

  def __getitem__(self, key):
    pid, page = self.dfs[key]

    X = page.drop(['pid', 'nid'], axis=1).to_numpy()
    label = page['nid'].to_numpy()
    pid = [pid for i in range(len(X))]

    return {'X': torch.Tensor(X), 'label': torch.LongTensor(label),
            'sid': torch.LongTensor(pid)}

  def __len__(self):
    return len(self.dfs)

  def collate(self, batch):
    X = torch.cat([b['X'] for b in batch], dim=0)
    label = torch.cat([b['label'] for b in batch], dim=0)
    sid = torch.cat([b['sid'] for b in batch], dim=0)

    return {'X': torch.Tensor(X), 'label': torch.LongTensor(label),
            'sid': torch.LongTensor(sid)}


def pdf_create_dataloaders(args):
  
  if args.model_type in ['dac', 'permequi', 'abc']:
    batch_size, cluster_batch_size = 1, 1
  elif args.model_type=='mil':
    batch_size, cluster_batch_size = args.batch_size, 1
  else:
    return -1

  dataset = PdfDataset('../data/processed/train.csv')
  test_d = PdfDataset('../data/processed/test.csv')

  if args.debug_dataset:
    
    # Debug data
    train_dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate, sampler=DebugSampler(args.debug_dataset))
    val_cluster_dl = data.DataLoader(dataset, batch_size=cluster_batch_size, collate_fn=dataset.collate, sampler=DebugSampler(args.debug_dataset))

    val_dl, early_stop_dl, val_dl = train_dl, train_dl, train_dl
    
    train_len = args.debug_dataset
    early_stop_len, val_len = train_len, train_len
    
  elif args.n_folds:
    
    if not args.test:
      # Don't use test dataset. Hold out small part of training part of dataset for early stop, evaluate on validation part.
      
      indices = np.arange(len(dataset))
      val_indices = np.loadtxt(f'../data/processed/fold{args.fold_number}.csv', dtype='int')
      train_indices = np.delete(indices, val_indices)
      
      # Hold out small part of train set for early stopping
      l = round(len(train_indices)*0.95)
      shuffled_indices = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(42+args.fold_number))
      early_stop_indices = train_indices[shuffled_indices[l:]]
      train_indices = train_indices[shuffled_indices[:l]]
      
      train_sampler = data.SubsetRandomSampler(train_indices)
      early_stop_sampler = data.SubsetRandomSampler(early_stop_indices)
      val_sampler = data.SubsetRandomSampler(val_indices)
      
      train_dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate, sampler=train_sampler)
      early_stop_dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate, sampler=early_stop_sampler)
      val_dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate, sampler=val_sampler)
      val_cluster_dl = data.DataLoader(dataset, batch_size=cluster_batch_size, collate_fn=dataset.collate, sampler=val_sampler)
      
      train_len, early_stop_len, val_len = len(train_indices), len(early_stop_indices), len(val_indices)
      
    else:
      # Use test dataset for evaluation, trainig part can therefore be larger, hold out small part for early stopping.
      train_indices = np.arange(len(dataset))

      # Hold out small part of train set for early stopping
      l = round(len(dataset)*0.95)
      shuffled_train_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42+args.fold_number))
      train_indices = shuffled_train_indices[:l]
      early_stop_indices = shuffled_train_indices[l:]
      
      train_sampler = data.SubsetRandomSampler(train_indices)
      early_stop_sampler = data.SubsetRandomSampler(early_stop_indices)
      
      train_dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate, sampler=train_sampler)
      early_stop_dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate, sampler=early_stop_sampler)
      val_dl = data.DataLoader(test_d, batch_size=batch_size, collate_fn=dataset.collate, shuffle=False)
      val_cluster_dl = data.DataLoader(test_d, batch_size=cluster_batch_size, collate_fn=dataset.collate, shuffle=False)
      
      train_len, early_stop_len, val_len = len(train_indices), len(early_stop_indices), len(test_d)
    
  else:
    if not args.test:
      train_len = round(len(dataset)*0.8)
      val_len = len(dataset) - train_len
      
      early_stop_len = round(train_len*0.05)
      train_len -= early_stop_len

      train_d, early_stop_d, val_d = data.random_split(dataset, [train_len, early_stop_len ,val_len], generator=torch.Generator().manual_seed(42))

      train_dl = data.DataLoader(train_d, batch_size=batch_size, collate_fn=dataset.collate, shuffle=True)
      early_stop_dl = data.DataLoader(early_stop_d, batch_size=batch_size, collate_fn=test_d.collate, shuffle=False)
      val_dl = data.DataLoader(val_d, batch_size=batch_size, collate_fn=dataset.collate, shuffle=False)
      val_cluster_dl = data.DataLoader(val_d, batch_size=cluster_batch_size, collate_fn=dataset.collate, shuffle=False)
      
      train_len, early_stop_len, val_len = len(train_d), len(early_stop_d), len(val_d)
    
    else:
      early_stop_len = round(len(dataset)*0.05)
      train_len = len(dataset)-early_stop_len
      
      rain_d, early_stop_d = data.random_split(dataset, [train_len, early_stop_len], generator=torch.Generator().manual_seed(42))
      
      train_dl = data.DataLoader(train_d, batch_size=batch_size, collate_fn=dataset.collate, shuffle=True)
      early_stop_dl = data.DataLoader(early_stop_d, batch_size=batch_size, collate_fn=test_d.collate, shuffle=False)
      val_dl = data.DataLoader(test_d, batch_size=batch_size, collate_fn=dataset.collate, shuffle=False)
      val_dl = data.DataLoader(test_d, batch_size=cluster_batch_size, collate_fn=dataset.collate, shuffle=False)
      
      train_len, early_stop_len, val_len = len(train_d), len(early_stop_d), len(test_d)
      
  return train_dl, early_stop_dl, val_dl, val_cluster_dl, (train_len, early_stop_len, val_len)
      
# ***** Circles *****
class CirclesDataset(Dataset):
  def __init__(self, n_sets, n_elements, n_clusters_low, n_clusters_high, shuffle):
    '''
      n_sets: number of sets in dataset
      n_elemnts: number of elements in each set
      n_clusters_low: lowest number of clusters in set
      n_clusters_high: highest number of clusters in set
      
      number of clusters in set is choosen randomly
    '''
    
    self.n_sets = n_sets
    self.n_elements = n_elements
    self.shuffle = shuffle
    
    self.points = torch.empty((n_sets, n_elements, 2))
    self.labels = torch.empty((n_sets, n_elements), dtype=int)
    self.counts = torch.empty((n_sets,), dtype=int)
    self.sid = torch.empty((n_sets, n_elements))
    
    for i, k in enumerate(torch.randint(n_clusters_low, n_clusters_high+1, size=(n_sets,))):
      points, labels = generate_circles_set(n_elements, int(k))
      
      if self.shuffle:
        shuffler = np.random.permutation(len(points))
        points = points[shuffler]
        labels = labels[shuffler]
          
      self.points[i] = points
      self.labels[i] = labels
      self.counts[i] = k

  def __len__(self):
    return self.n_sets
  
  def __getitem__(self,key):
    X = self.points[key]
    label = self.labels[key]
    sid = [key for i in range(len(X))]
    
    return {'X': X, 'label': torch.LongTensor(label), 'sid': torch.LongTensor(sid)}
  
  def collate(self, batch):
    X = torch.cat([b['X'] for b in batch], dim=0)
    label = torch.cat([b['label'] for b in batch], dim=0)
    sid = torch.cat([b['sid'] for b in batch], dim=0)
    
    return {'X': torch.Tensor(X), 'label': torch.LongTensor(label),
            'sid': torch.LongTensor(sid)}
    

def generate_circles_set(L, k):
  centers = torch.from_numpy(0.5 * _sample_in_unit_circle(k)).to(dtype=torch.float) # (k, 2)
  radii   = torch.clamp(torch.normal(0.3, 0.1, size=(k,)), min=0.2, max=0.4) # (k,)
  freqs   = np.random.multinomial(L-k, [1/k,]*k) + 1
  labels  = torch.from_numpy(np.repeat(np.arange(k), freqs)) # (L,)
  points = torch.empty((L, 2))
  cumsum = 0
  for i, freq in enumerate(freqs):
    angles = np.linspace(0, 2*math.pi, num=freq, endpoint=False) + np.random.uniform(0, 2*math.pi/freq, size=freq)
    directions = torch.tensor([[math.cos(angle), math.sin(angle)] for angle in angles])
    points[cumsum:cumsum+freq] = centers[i] + radii[i].unsqueeze(-1) * directions
    cumsum += freq
  return points, labels


def _sample_in_unit_circle(size=1):
  radii  = np.sqrt(np.random.uniform(0, 1, size=size))
  thetas = np.random.uniform(0, 2*math.pi, size=size)
  return np.array([[r*math.cos(theta), r*math.sin(theta)] for r, theta in zip(radii, thetas)])


def circles_create_dataloaders(args):
  '''
    No need to differentiate test runs and runs for hyperparameter optimization.
  '''

  if args.model_type in ['dac', 'abc', 'permequi']:
    batch_size, cluster_batch_size = args.batch_size, args.batch_size
  elif args.model_type in ['mil']:
    batch_size, cluster_batch_size = args.batch_size, 1
  else:
    return -1 

  
  train_d = CirclesDataset(args.n_sets_train, args.n_elements, args.n_clusters_low,
                           args.n_clusters_high, args.circles_shuffle)
  early_stop_d = CirclesDataset(round(args.n_sets_train*0.01), args.n_elements, args.n_clusters_low,
                                args.n_clusters_high, args.circles_shuffle)
  val_d = CirclesDataset(args.n_sets_val, args.n_elements, args.n_clusters_low,
                         args.n_clusters_high, args.circles_shuffle)

  if args.model_type=='mil':
    collate = train_d.collate
  else:
    collate = None
  
  # train and val data loaders
  train_dl = data.DataLoader(train_d, batch_size=batch_size, shuffle=True, collate_fn=collate)
  early_stop_dl = data.DataLoader(early_stop_d, batch_size=batch_size, shuffle=False, collate_fn=collate)
  val_dl = data.DataLoader(val_d, batch_size=batch_size, shuffle=False, collate_fn=collate)
  val_cluster_dl = data.DataLoader(val_d, batch_size=cluster_batch_size, shuffle=False, collate_fn=collate)
  
  return train_dl, early_stop_dl, val_dl, val_cluster_dl, (len(train_d), len(early_stop_d), len(val_d))


# ****** MoG *******
class MoGDataset(Dataset):
  def __init__(self, n_sets, n_elements, n_clusters_low, n_clusters_high):
    '''
      n_sets: number of sets in dataset
      n_elemnts: number of elements in each set
      n_clusters_low: lowest number of clusters in set
      n_clusters_high: highest number of clusters in set
      
      number of clusters in set is choosen randomly
    '''
    
    self.n_sets = n_sets
    self.n_elements = n_elements
    
    data = sample_mog(n_sets, n_elements, n_clusters_low, n_clusters_high)
    
    self.points = data['X']
    self.labels = data['labels']
    
  def __len__(self):
    return self.n_sets
  
  def __getitem__(self,key):
    X = self.points[key]
    label = self.labels[key]
    sid = [key for i in range(len(X))]
    
    return {'X': X, 'label': torch.LongTensor(label), 'sid': torch.LongTensor(sid)}
  
  def collate(self, batch):
    X = torch.cat([b['X'] for b in batch], dim=0)
    label = torch.cat([b['label'] for b in batch], dim=0)
    sid = torch.cat([b['sid'] for b in batch], dim=0)
    
    return {'X': torch.Tensor(X), 'label': torch.LongTensor(label),
            'sid': torch.LongTensor(sid)}


class MultivariateNormalDiag():
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.dim_params = 2*dim

  def sample_params(self, shape, device='cpu'):
    shape = torch.Size(shape) + torch.Size([self.dim])
    mu = 3.0 * torch.randn(shape).to(device)
    sigma = (math.log(0.25) + 0.1*torch.randn(shape)).exp().to(device)
    return torch.cat([mu, sigma], -1)

  def sample(self, params):
    mu = params[...,:self.dim]
    sigma = params[...,self.dim:]
    eps = torch.randn(mu.shape).to(mu.device)
    return mu + eps * sigma

      
def sample_mog(B, N, K_low, K_high, alpha=1.0):
  mvn = MultivariateNormalDiag(2)
  labels = sample_labels(B, N, K_low, K_high, alpha=alpha)
  params = mvn.sample_params([B, K_high])
  gathered_params = torch.gather(params, 1, labels.unsqueeze(-1).repeat(1, 1, params.shape[-1]))
  X = mvn.sample(gathered_params)
  dataset = {'X':X, 'labels':labels}

  return dataset
  

def sample_labels(B, N, K_low, K_high, alpha=1.0):
  pi = Dirichlet(alpha*torch.ones(K_high)).sample([B])
  K = torch.randint(K_low, K_high+1, size=(B,))
  to_use = torch.zeros(B, K_high).int()
  for i, k in enumerate(K):
    to_use[i, :k] = 1    
  pi = pi * to_use
  pi = pi/pi.sum(1, keepdim=True)
  labels = Categorical(probs=pi).sample([N])
  labels = labels.transpose(0,1).contiguous()
  return labels


def mog_create_dataloaders(args):
  '''
    No need to differentiate test runs and runs for hyperparameter optimization.
  '''

  if args.model_type in ['dac', 'abc']:
    batch_size, cluster_batch_size = args.batch_size, args.batch_size
  elif args.model_type in ['mil', 'permequi']:
    batch_size, cluster_batch_size = args.batch_size, 1
  else:
    return -1 

  
  train_d = MoGDataset(args.n_sets_train, args.n_elements, args.n_clusters_low,
                           args.n_clusters_high)
  early_stop_d = MoGDataset(round(args.n_sets_train*0.01), args.n_elements, args.n_clusters_low,
                                args.n_clusters_high)
  val_d = MoGDataset(args.n_sets_val, args.n_elements, args.n_clusters_low,
                         args.n_clusters_high)

  if args.model_type=='mil':
    collate = train_d.collate
  else:
    collate = None
  
  # train and val data loaders
  train_dl = data.DataLoader(train_d, batch_size=batch_size, shuffle=True, collate_fn=collate)
  early_stop_dl = data.DataLoader(early_stop_d, batch_size=batch_size, shuffle=False, collate_fn=collate)
  val_dl = data.DataLoader(val_d, batch_size=batch_size, shuffle=False, collate_fn=collate)
  val_cluster_dl = data.DataLoader(val_d, batch_size=cluster_batch_size, shuffle=False, collate_fn=collate)
  
  return train_dl, early_stop_dl, val_dl, val_cluster_dl, (len(train_d), len(early_stop_d), len(val_d))


# ****** MIL *******
# def batch_to_mil_pairs(batch, n_pairs, pairs_replacement, pair_bag_len):

#   def add_pair(bag, pair, pair_bag_len):
#     # Does it suffice to concat only some elements of the set to the pair?
    
#     pair = bag[list(pair)].flatten()
    
#     if pair_bag_len:
#       pair = pair.repeat(pair_bag_len, 1)
#       # Choose pair_set_len elements of set randomly
#       idxs = random.sample(range(len(bag)), k=pair_bag_len)
#       new_bag = bag[idxs]
      
#     else:  
#       pair = pair.repeat(len(bag), 1)
#       new_bag = bag
      
#     return torch.cat([new_bag,pair],dim=1)

#   bid = 0

#   unique_sid = batch['sid'].unique()

#   ret = {'X': [], 'T': [], 'B': []}

#   for sid in unique_sid:
#     mask = (batch['sid'] == sid)
#     bag = batch['X'][mask] 
#     label = batch['label'][mask]

#     if n_pairs:
#       if pairs_replacement:
#         # Choose n_pairs random pairs (pair of same elements can occur to force the model to put it into same cluster) 
#         pairs = [random.choices(range(len(bag)), k=2) for i in range(n_pairs)]
#       else:
#         # Choose n_pairs random pairs (pair of same element does not occur)
#         pairs = [random.sample(range(len(bag)), k=2) for i in range(n_pairs)]
#     else:
#       if pairs_replacement:
#         # Create all possible pairs (pair of same elements do occur - to force the model to put them into same cluster)
#         pairs = list(itertools.combinations_with_replacement(range(len(bag)), 2))
#         n_pairs = len(pairs)
#       else:
#         # Create all possible pairs (pair of same elements do not occur)
#         pairs = list(itertools.combinations(range(len(bag)), 2))
#         n_pairs = len(pairs)
        
#     # Create labels
#     ret['T'].append(torch.LongTensor([(label[pair[0]] == label[pair[1]])*1 for pair in pairs]))

#     ret['X'].append(torch.cat([add_pair(bag, list(pair), pair_bag_len) for pair in pairs], dim=0))

#     if pair_bag_len:
#       l = pair_bag_len
#     else:
#       l = len(bag)
#     ret['B'].append(torch.cat([torch.LongTensor(np.ones(l)*(bid+i)) for i in range(len(pairs))], dim=0))

#     bid += n_pairs

#   ret['T'] = torch.cat(ret['T'], dim=0)
#   ret['X'] = torch.cat(ret['X'], dim=0)
#   ret['B'] = torch.cat(ret['B'], dim=0)
#   return ret


def batch_to_mil_pairs(batch, n_pairs, stratify, pair_bag_len):

  def add_pair(bag, pair, pair_bag_len):
    # Does it suffice to concat only some elements of the set to the pair?
    
    pair = bag[pair].flatten()
    
    
    if pair_bag_len:
      pair = pair.repeat(pair_bag_len, 1)
      # Choose pair_set_len elements of set randomly
      idxs = random.sample(range(len(bag)), k=pair_bag_len)
      new_bag = bag[idxs]
      
    else:  
      pair = pair.repeat(len(bag), 1)
      new_bag = bag
      
    return torch.cat([new_bag,pair],dim=1)

  bid = 0

  unique_sid = batch['sid'].unique()

  ret = {'X': [], 'T': [], 'B': []}

  for sid in unique_sid:
    mask = (batch['sid'] == sid)
    bag = batch['X'][mask] 
    label = batch['label'][mask]
  
    # Create all possible pairs with replacements
    pool = range(len(bag))
    if n_pairs:
      
      if stratify:
        
        pairs = []
        n_p = n_pairs//2
        n_n = n_pairs - n_p
                
        for i in range(200): # If there is only one cluster, or majority of elemetns are in one cluster, stratify fails. 200 seems to be a good value
      
          pair = random.choices(pool, k=2)
          if n_p and (label[pair[0]] == label[pair[1]]):
            pairs.append(pair)
            n_p-=1
          elif n_n and (label[pair[0]] != label[pair[1]]):
            pairs.append(pair)
            n_n-=1
          else:
            pass   
          
          # If positive and negative pairs sampled, break. 
          if (n_p==0 and n_n==0):
            break
            
      else:
        pairs = [random.choices(pool, k=2) for i in range(n_pairs)]
        
    else:
      pairs = list(itertools.combinations_with_replacement(pool, 2))

    pairs = torch.LongTensor(pairs)
              
    ret['T'].append(torch.LongTensor([(label[pair[0]] == label[pair[1]])*1 for pair in pairs]))
    ret['X'].append(torch.cat([add_pair(bag, pair, pair_bag_len) for pair in pairs], dim=0))

    if pair_bag_len:
      l = pair_bag_len
    else:
      l = len(bag)
      
    ret['B'].append(torch.cat([torch.LongTensor(np.ones(l)*(bid+i)) for i in range(len(pairs))], dim=0))

    bid += len(pairs)

  ret['T'] = torch.cat(ret['T'], dim=0)
  ret['X'] = torch.cat(ret['X'], dim=0)
  ret['B'] = torch.cat(ret['B'], dim=0)
  
  return ret


def batch_to_mil_pair_indicators(batch, n_pairs, pairs_replacement):

  def add_indicator(bag, pair):
    pair_indicator = torch.zeros((len(bag), 1))
    pair_indicator[pair[0]] = 1
    pair_indicator[pair[1]] = 1
    return torch.cat([bag,pair_indicator], dim=1)

  bid = 0

  unique_sid = batch['sid'].unique()

  ret = {'X': [], 'T': [], 'B': []}

  for sid in unique_sid:
    mask = (batch['sid'] == sid)
    bag = batch['X'][mask] 
    label = batch['label'][mask]

    if n_pairs:
      if pairs_replacement:
        # Choose n_pairs random pairs (pair of same elements can occur to force the model to put it into same cluster) 
        pairs = [random.choices(range(len(bag)), k=2) for i in range(n_pairs)]
      else:
        # Choose n_pairs random pairs (pair of same element does not occur)
        pairs = [random.sample(range(len(bag)), k=2) for i in range(n_pairs)]
    else:
      if pairs_replacement:
        # Create all possible pairs (pair of same elements do occur - to force the model to put them into same cluster)
        pairs = list(itertools.combinations_with_replacement(range(len(bag)), 2))
        n_pairs = len(pairs)
      else:
        # Create all possible pairs (pair of same elements do not occur)
        pairs = list(itertools.combinations(range(len(bag)), 2))
        n_pairs = len(pairs)

    # Create labels
    ret['T'].append(torch.LongTensor([(label[pair[0]] == label[pair[1]])*1 for pair in pairs]))

    ret['X'].append(torch.cat([add_indicator(bag, list(pair)) for pair in pairs], dim=0))

    ret['B'].append(torch.cat([torch.LongTensor(np.ones(len(bag))*(bid+i)) for i in range(len(pairs))], dim=0))

    bid += n_pairs

  ret['T'] = torch.cat(ret['T'], dim=0)
  ret['X'] = torch.cat(ret['X'], dim=0)
  ret['B'] = torch.cat(ret['B'], dim=0)
  return ret


# ******** MLP *********
def batch_to_same_cluster(batch, variable=True, n_elements = 0):
  unique_sid = batch['sid'].unique()
  ret = {'X': [], 'T': []}
  
  if variable:
    # Dataset with variable number of inputs and outputs (need to sample)
    
    for sid in unique_sid:
      mask = (batch['sid'] == sid)
      X = batch['X'][mask] 
      label = batch['label'][mask]
      input_size = len(X[0])

      if n_elements > len(bag):
        # Sample with replacement but ensure all instances occur
  #             sample = list(range(len(bag))) + random.choices(range(len(bag)), k=(n_instances-len(bag)))
        sample = random.sample(range(len(bag)),
                               len(bag)) + random.choices(range(len(bag)), k=(n_elements-len(bag)))

      else:
        # Sample without replacement
        sample = random.sample(range(len(bag)), n_elements)

        X = X[sample]
        label = label[sample]

        X = torch.cat([X, torch.zeros((n_elements, 1))], dim=1)

        X = X.flatten().repeat(n_instances, 1)

        for i in range(n_elements):
          X[i, input_size+i*(input_size+1)] = 1

      # Create target
      T = torch.cat([((label == i)*1).unsqueeze(0) for i in label],dim=0)

      ret['X'].append(X)
      ret['T'].append(T)
      
    ret['X'] = torch.cat(ret['X'], dim=0)
    ret['T'] = torch.cat(ret['T'], dim=0)
      
  else:
    # Dataset with same number of elements in each set
    
    X = batch['X']
    batch_size , n_elements, input_size = X.shape
    
    X = torch.cat([X, torch.zeros((batch_size,n_elements,1))], dim=2)
    X = X.flatten(start_dim=1)
    X = X.unsqueeze(1).repeat(1, n_elements, 1)
    
    for i in range(n_elements):
      X[:, i, input_size+i*(input_size+1)] = 1
    
    T = []

    for label in batch['label']:
      T.append(torch.cat([((label == i).unsqueeze(0).long()) for i in label], dim=0).unsqueeze(0))

    T = torch.cat(T, dim=0)
    
    ret['X'] = X
    ret['T'] = T

  return ret

# ******* ABC *******


# ******* MCL *******
def batch_to_mcl_pairs(batch):
  # SPN model used to predict similarity of two elements of instance is not symmetric,
  # we therefore create each pair of elemnts in both possible orders (eg. 'AB' and 'BA')
  # to force symmetry.
  # We also use pair made out of same element since model should classify this pair as similar.

  unique_sid = batch['sid'].unique()
  ret = {'X': [], 'T': []}

  # Create pairs only of instances from same page
  for sid in unique_sid:
    mask = (batch['sid'] == sid)
    bag = batch['X'][mask] 
    label = batch['label'][mask]

    feat1, feat2 = PairEnum(bag)
    X = torch.cat([feat1,feat2], dim=1)
    T = Class2Simi(label)

    ret['X'].append(X)
    ret['T'].append(T)

  ret['X'] = torch.cat(ret['X'], dim=0)
  ret['T'] = torch.cat(ret['T'], dim=0)
  return ret


# Prepares target for training of MCL network. Uses SPN for pairwise labels if SPN specified, otherwise
# creates pairwise labels from true clustering.
def prepare_mcl_target(batch, target, SPN=None, SPN_type='fc'):    
  if SPN is None:
    target = Class2Simi(target, mode='hinge')
  else:
    if SPN_type == 'fc':
      feat1, feat2 = PairEnum(batch['X'])
      X = torch.cat([feat1,feat2], dim=1)
      target = SPN.forward(X).argmax(-1)
      target = target.float()
      target[target==0] = -1  # Simi:1, Dissimi:-1
    elif SPN_type == 'mil':
      batch = batch_to_mil_pairs(batch, n_pairs = 0)
      target  = SPN(batch['X'], batch['B']).argmax(-1).float()
      target[target==0] = -1

  return target.detach()


def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


def Class2Simi(x,mode='cls',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

  
# ********* DAC ********
def batch_to_dac_compatible(batch, augment=False):
  
  X = batch['X']
  label = batch['label']
  
  # use only some clusters from set for data augmentation
  if augment: 
    unique_label = torch.unique(label)
    k = random.choice(range(len(unique_label))) + 1
    unique_label = random.sample(unique_label.tolist(), k=k)
    mask = sum([label == i for i in unique_label]) > 0
    X = X[mask]
    label = label[mask]

  # One-hot encode labels
  label = F.one_hot(torch.LongTensor(label))
  
  ret = {}
  ret['X'] = X.unsqueeze(0)
  ret['label'] = label.unsqueeze(0)
      
  return ret


# ******* Data functions for debugging ********
class DebugSampler(Sampler):
  # Use only on n_samples to overfit
  def __init__(self, n_samples):
      self.n_samples = n_samples

  def __iter__(self):
      return iter(range(self.n_samples))

  def __len__(self):
      return self.n_samples
