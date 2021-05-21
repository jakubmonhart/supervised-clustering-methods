import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AddCompat(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.v = nn.Parameter(torch.Tensor(size)) # importance vector
    bound = 1 / self.v.shape[0]**.5
    self.v.data.uniform_(-bound, bound)
  
  def forward(self, Q, K):
    Q = Q.unsqueeze(-2)        # (*, m, 1, H)
    K  = K.unsqueeze(-3)           # (*, 1, n, H)
    QK = torch.tanh(Q + K)              # (*, m, n, H)
    A_logits = torch.matmul(QK, self.v) # (*, m, n)
    
    return A_logits
  
class MultiCompat(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
  
  def forward(self, Q, K):
    A_logits = (Q @ K.transpose(1,2)) / math.sqrt(self.dim)
    
    return A_logits

class MAB(nn.Module):
  def __init__(self, dim_X, dim_Y, dim, num_heads=4, ln=False, p=None, compat='multi'):
    super().__init__()
    self.num_heads = num_heads
    self.fc_q = nn.Linear(dim_X, dim)
    self.fc_k = nn.Linear(dim_Y, dim)
    self.fc_v = nn.Linear(dim_Y, dim)
    self.fc_o = nn.Linear(dim, dim)

    self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
    self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
    self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
    self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()
    
    if compat == 'multi':
      self.compat = MultiCompat(dim)
    elif compat == 'add':
      self.compat = AddCompat(dim//num_heads)
    else:
      print(f'Compatibility function {compat} not implemented.')
      return -1

  def forward(self, X, Y, mask=None):
    Q, K, V = self.fc_q(X), self.fc_k(Y), self.fc_v(Y)
    Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
    K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
    V_ = torch.cat(V.chunk(self.num_heads, -1), 0)
  
    # Compute compatibility
    A_logits = self.compat(Q_, K_)
    
    if mask is not None:
      mask = mask.squeeze(-1).unsqueeze(1)
      mask = mask.repeat(self.num_heads, Q.shape[1], 1)
      A_logits.masked_fill_(mask, -100.0)
    A = torch.softmax(A_logits, -1)

    attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
    O = self.ln1(Q + self.dropout1(attn))
    O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
    return O

class SAB(nn.Module):
  def __init__(self, dim_X, dim, **kwargs):
    super().__init__()
    self.mab = MAB(dim_X, dim_X, dim, **kwargs)

  def forward(self, X, mask=None):
    return self.mab(X, X, mask=mask)

class StackedSAB(nn.Module):
  def __init__(self, dim_X, dim, num_blocks, **kwargs):
    super().__init__()
#    self.blocks = nn.ModuleList(
#          [SAB(dim_X, dim, **kwargs)] + \
#          [SAB(dim, dim, **kwargs)]*(num_blocks-1))

    self.blocks = nn.ModuleList([SAB(dim_X, dim, **kwargs)])
    for i in range(num_blocks-1):
      self.blocks.append(SAB(dim, dim, **kwargs))
      
  def forward(self, X, mask=None):
    for sab in self.blocks:
      X = sab(X, mask=mask)
    return X

class PMA(nn.Module):
  '''
  If used as pooling:
    - num_inds: should be set to 1 for pooling
    - dim_x: dimension of input
    - dim: dimension of output
  '''
  
  def __init__(self, dim_X, dim, num_inds, **kwargs):
    super().__init__()
    self.I = nn.Parameter(torch.Tensor(1, num_inds, dim))
    nn.init.xavier_uniform_(self.I)
    self.mab = MAB(dim, dim_X, dim, **kwargs)

  def forward(self, X, mask=None):
    return self.mab(self.I.repeat(X.shape[0], 1, 1), X, mask=mask)

class ISAB(nn.Module):
  def __init__(self, dim_X, dim, num_inds, **kwargs):
    super().__init__()
    self.pma = PMA(dim_X, dim, num_inds, **kwargs)
    self.mab = MAB(dim_X, dim, dim, **kwargs)

  def forward(self, X, mask=None):
    return self.mab(X, self.pma(X, mask=mask))

class StackedISAB(nn.Module):
  def __init__(self, dim_X, dim, num_inds, num_blocks, **kwargs):
    super().__init__()
    
    # Bug in DAC's authors code
#         self.blocks = nn.ModuleList(
#                 [ISAB(dim_X, dim, num_inds, **kwargs)] + \
#                 [ISAB(dim, dim, num_inds, **kwargs)]*(num_blocks-1))

    self.blocks = nn.ModuleList([ISAB(dim_X, dim, num_inds, **kwargs)])
    for i in range(num_blocks-1):
      self.blocks.append(ISAB(dim, dim, num_inds, **kwargs))

  def forward(self, X, mask=None):
    for isab in self.blocks:
        X = isab(X, mask=mask)
    return X
