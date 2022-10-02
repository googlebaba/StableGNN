import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
from scipy.linalg import block_diag

import dgl
from torch.autograd import Variable
from .dgl_layers import GraphSage, GraphSageLayer, DiffPoolBatchedGraphLayer
from .tensorized_layers import *
from .model_utils import batch2tensor
import time
from .weighted_loss import CrossEntropyLoss
class HSIC_weight(nn.Module):
    """
    DiffPool Fuse
    """

    def __init__(self, final_readout, embedding_size, n, cat=False):
        super(HSIC_weight, self).__init__()
        self.final_readout = final_readout
        self.embedding_size = embedding_size
        self.n = n
        #print("para", sample_weight)
        #self.params = nn.ParameterList([sample_weight])
        
        self.weight = nn.Parameter(torch.ones((n, 1)))
        #self.params = nn.ParameterList([nn.Parameter(self.weight)])
        #self.sample_weights = sample_weight * sample_weight
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data,
                                                     gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
    def loss_dependence(self, emb1, emb2, sample_weights, dim):
        R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
        W1 = torch.mm(sample_weights, sample_weights.t())
        K1 = W1*torch.mm(emb1, emb1.t())
        K2 = W1*torch.mm(emb2, emb2.t())
        RK1 = torch.mm(R, K1)
        RK2 = torch.mm(R, K2)
        HSIC = torch.trace(torch.mm(RK1, RK2))/((dim-1)*(dim-1))       
        return HSIC

    def forward(self):
        loss = Variable(torch.FloatTensor([0]).cuda())
        for i in range(9):
            for j in range(i+1, 10):
                feature1 = self.final_readout[:, i*self.embedding_size : (i+1)*self.embedding_size]
                feature2 = self.final_readout[:, j*self.embedding_size : (j+1)*self.embedding_size]
                loss += self.loss_dependence(feature1, feature2, self.weight * self.weight, self.n).view(1)     
        #loss += 0.005*torch.sum(self.weight*self.weight-self.n)**2
        #loss += 0.005*torch.sum((self.weight*self.weight)**2)
        return loss

    def loss(self, pred, label):
        '''
        loss function
        '''
        #softmax + CE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss

class DiffPool(nn.Module):
    """
    DiffPool Fuse
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim,
                 label_dim, activation, n_layers, dropout,
                 n_pooling, linkpred, batch_size, aggregator_type,
                 assign_dim, pool_ratio, assign_num, cat=False):
        super(DiffPool, self).__init__()
        self.link_pred = linkpred
        self.concat = cat
        self.n_pooling = n_pooling
        self.batch_size = batch_size
        self.link_pred_loss = []
        self.entropy_loss = []

        # list of GNN modules before the first diffpool operation
        self.gc_before_pool = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()

        # list of list of GNN modules, each list after one diffpool operation
        self.gc_after_pool = nn.ModuleList()
        self.assign_dim = assign_dim
        self.bn = True
        self.num_aggs = 1

        # constructing layers
        # layers before diffpool
        #assert n_layers >= 3, "n_layers too few"
        self.gc_before_pool.append(
            GraphSageLayer(
                input_dim,
                hidden_dim,
                activation,
                dropout,
                aggregator_type,
                self.bn))
        for _ in range(n_layers - 2):
            self.gc_before_pool.append(
                GraphSageLayer(
                    hidden_dim,
                    hidden_dim,
                    activation,
                    dropout,
                    aggregator_type,
                    self.bn))
        self.gc_before_pool.append(
            GraphSageLayer(
                hidden_dim,
                embedding_dim,
                None,
                dropout,
                aggregator_type))

        assign_dims = []
        assign_dims.append(self.assign_dim)
        if self.concat:
            # diffpool layer receive pool_emedding_dim node feature tensor
            # and return pool_embedding_dim node embedding
            pool_embedding_dim = hidden_dim * (n_layers - 1) + embedding_dim
        else:

            pool_embedding_dim = embedding_dim

        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            pool_embedding_dim,
            self.assign_dim,
            hidden_dim,
            activation,
            dropout,
            aggregator_type,
            self.link_pred)
        gc_after_per_pool = nn.ModuleList()

        for _ in range(n_layers - 1):
            gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, hidden_dim))
        gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, embedding_dim))
        self.gc_after_pool.append(gc_after_per_pool)

        self.assign_dim = int(self.assign_dim * pool_ratio)
        # each pooling module
        for _ in range(n_pooling - 1):
            self.diffpool_layers.append(
                BatchedDiffPool(
                    pool_embedding_dim,
                    self.assign_dim,
                    hidden_dim,
                    self.link_pred))
            gc_after_per_pool = nn.ModuleList()
            for _ in range(n_layers - 1):
                gc_after_per_pool.append(
                    BatchedGraphSAGE(
                        hidden_dim, hidden_dim))
            gc_after_per_pool.append(
                BatchedGraphSAGE(
                    hidden_dim, embedding_dim))
            self.gc_after_pool.append(gc_after_per_pool)
            assign_dims.append(self.assign_dim)
            self.assign_dim = int(self.assign_dim * pool_ratio)

        # predicting layer
        if self.concat:
            self.pred_input_dim = pool_embedding_dim * \
                self.num_aggs * (n_pooling + 1)
        else:
            self.pred_input_dim = embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim*assign_num, label_dim)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data,
                                                     gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def gcn_forward(self, g, h, gc_layers, cat=False):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h)
            block_readout.append(h)
        h = gc_layers[-1](g, h)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def forward(self, g):
        self.link_pred_loss = []
        self.entropy_loss = []
        h = g.ndata['feat']
        # node feature for assignment matrix computation is the same as the
        # original node feature
        h_a = h

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gcn_forward(g, h, self.gc_before_pool, self.concat)

        g.ndata['h'] = g_embedding

        readout = dgl.sum_nodes(g, 'h')
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, 'h')
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / self.batch_size)

        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.gcn_forward_tensorized(
            h, adj, self.gc_after_pool[0], self.concat)
        readout = torch.reshape(h, (h.shape[0], -1))
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(
                h, adj, self.gc_after_pool[i + 1], self.concat)
            readout = torch.cat(h, dim=0)
            out_all.append(readout)
            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)
        if self.concat or self.num_aggs > 1:
            final_readout = torch.cat(out_all, dim=1)
        else:
            final_readout = readout
        ypred = self.pred_layer(final_readout)
        return ypred, final_readout


    def cross_entropy_with_weights(logits, target, weights=None):
        assert logits.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1
        loss = log_sum_exp(logits) - class_select(logits, target)
        if weights is not None:
            # loss.size() = [N]. Assert weights has the same shape
            assert list(loss.size()) == list(weights.size())
            # Weight the loss
            loss = loss * weights
        return loss
    def weighted_loss(self, pred, label, weights):
        '''
        loss function
        '''
        #softmax + CE
        criterion = CrossEntropyLoss(aggregate='mean')
        loss = criterion(pred, label, weights)
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss
    def loss(self, pred, label):
        '''
        loss function
        '''
        #softmax + CE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss
