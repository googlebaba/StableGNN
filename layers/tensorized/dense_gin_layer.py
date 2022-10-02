import torch
from torch import nn as nn
from torch.nn import functional as F

"""
    <Dense/Tensorzied version of the GraphSage layer>
    
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
    
    ! code started from the dgl diffpool examples dir
"""

class DenseGIN(nn.Module):
    def __init__(self, infeat, outfeat, apply_func, dropout, residual=False, use_bn=True,
                 mean=False, add_self=False, init_eps=0, last_layer=False):
        super().__init__()
        self.apply_func = apply_func
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.dropout = dropout
        self.residual = residual
        self.last_layer = last_layer
        
        if infeat != outfeat:
            self.residual = False
        
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight,
            gain=nn.init.calculate_gain('relu'))

        self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))

    def forward(self, x, adj):
        h_in = x               # for residual connection
        
        if self.use_bn and not hasattr(self, 'bn'):
            self.bn = nn.BatchNorm1d(adj.size(1)).to(adj.device)

        #if self.mean:
        #    adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = (1+self.eps) * h_in + h_k_N
        if self.apply_func is not None:
            h = self.apply_func(h_k)
        if self.last_layer:
            h = F.dropout(h, self.dropout, training=self.training)
        else:
            h = F.relu(h) # non-linear activation
            h = F.dropout(h, self.dropout, training=self.training)
            
 
        if self.residual:
            h_k = h_in + h_k    # residual connection
        
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return 'BN' + super(DenseGraphSage, self).__repr__()
        else:
            return super(DenseGraphSage, self).__repr__()
