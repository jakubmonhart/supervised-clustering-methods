import torch
from sklearn.cluster import SpectralClustering

import numpy as np
from numpy.linalg import eigvalsh

import warnings
warnings.filterwarnings('ignore') 

def eigengaps(A):
    """
    
    From https://github.com/DramaCow/ABC/blob/master/lib/spectral.py 
    
    Predict number of clusters based on the eigengap.
    Parameters
    ----------
    A : array-like, shape: (*, n, n)
        Affinity matrices. Each element of a matrix contains a measure of similarity between two data points.
    """
    
    device = A.device
        
    # assume square matrices
    n = A.shape[-1]

    # degree vector
    deg = A.sum(-1).unsqueeze(-2)                        # (*, 1, n)
    
    # inverse sqrt degree matrix:
    # since degree matrix is a diagonal matrix, the 
    # inverse is just the reciprocal of the diagonals
    D = ((1 / deg) * torch.eye(n, device=device)).sqrt() # (*, n, n)

    # normalised Laplacian matrix
    L = torch.matmul(D, torch.matmul(A, D))              # (*, n, n)

    # eigengaps defined as the difference between consecutive sorted
    # (descending) eigenvalues of the normalised Laplacian matrix
    gaps = np.ascontiguousarray(np.flip(np.diff(eigvalsh(L.cpu())), axis=-1))
    
    return torch.from_numpy(gaps) # result should be on cpu
  

def cluster_spectral(Ap):
  '''
  Take Ap "similarity matrix" as input (probability). Entry > 0.5 means similar, otherwise disimilar.
  '''
  
  
  kp = (eigengaps(Ap).argmax(-1) + 1).cpu() # Predict number of clusters
  Ap = Ap.cpu()

  return torch.tensor(SpectralClustering(int(kp), affinity='precomputed').fit(Ap).labels_)
  
  
def cluster_iterative(Ap):
  '''
  Take Ap "similarity matrix" (probabilities) as input. Entry > 0.5 means similar, otherwise disimilar.
  '''
  
  ne = Ap.size(1)
  threshold = 0.5

  selected = (Ap > threshold).int()
  # Force anchors to be in their clusters
  selected[torch.eye(ne).bool()] = 1

  C = -torch.ones(ne)
  cid = 0
  while ne > 0:
    scores = torch.sum(Ap * selected, dim=1)/torch.sum(selected, dim=1)
    # Set nan values to zero
    scores[torch.isnan(scores)] = torch.tensor([-float('inf')])

    anchor = torch.argmax(scores)
    cls_idxs = selected[anchor].clone()
    C[cls_idxs.bool()] = cid

    # remove selected
    selected[cls_idxs.bool()] = 0
    selected[:, cls_idxs.bool()] = 0

    nselected = torch.sum(cls_idxs)

    if nselected == 0:
      C[C==-1] = cid
      break

    ne -= nselected
    cid += 1
    
  return C
