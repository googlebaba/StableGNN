import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
from scipy.linalg import block_diag

import dgl

from .dgl_layers import GraphSage, GraphSageLayer, DiffPoolBatchedGraphLayer
from .tensorized_layers import *
from .model_utils import batch2tensor
import time

from .weighted_loss import CrossEntropyLoss

from torch.autograd import Variable

softmax = nn.Softmax(0)
class HSIC_weight(nn.Module):
    """
    DiffPool Fuse
    """

    def __init__(self, cfeatures, pre_features,  pre_weight, embedding_size, n, assign_num, cat=False, first= False):
        super(HSIC_weight, self).__init__()
        self.embedding_size = embedding_size
        self.n = n
        self.assign_num = assign_num
        self.pre_weight = pre_weight
        cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
        cfeaturec.data.copy_(cfeatures.data)
        if True:
            self.all_features = cfeaturec
        else:
            self.all_features = torch.cat([cfeaturec, pre_features.detach()], dim=0)

        self.weights = nn.Parameter(torch.ones((n, 1)))
        #print("para", sample_weight)
        #self.params = nn.ParameterList([sample_weight])
        
        
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

    def _kernel(self, X, sigma=1.0):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX
    def biased_estimator(self, input1, input2, sample_weights):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
       # print("sample_weights", sample_weights)
       # print("input1", input1)
        weighted_input1 = sample_weights * input1
       # print("weighted_input1", weighted_input1)
        weighted_input2 = sample_weights * input2
        K = self._kernel(weighted_input1)
        L = self._kernel(weighted_input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)


    def forward(self, lambdap, global_epoch, lambda_decay_rate, lambda_decay_epoch, min_lambda_times, first):
        if True:
            self.all_weights = self.weights
        else:
            self.all_weights = torch.cat((self.weights, self.pre_weight.detach()), dim=0)
        lossb = Variable(torch.FloatTensor([0]).cuda())
        for i in range(self.assign_num-1):
            for j in range(i+1, self.assign_num):
                feature1 = self.all_features[:, i*self.embedding_size : (i+1)*self.embedding_size]
                feature2 = self.all_features[:, j*self.embedding_size : (j+1)*self.embedding_size]
                #lossb += self.loss_dependence(feature1, feature2, softmax(self.all_weights), self.all_features.size(0)).view(1)
                lossb += self.biased_estimator(feature1, feature2, softmax(self.all_weights)).view(1)
        #lossq = (torch.sum(self.weights*self.weights)-self.n)**2 
        lossp = softmax(self.weights).pow(2).sum()
        lambdap = lambdap * max((lambda_decay_rate ** (global_epoch // lambda_decay_epoch)),
                                     min_lambda_times)
        lossg = 1e+3*lossb / lambdap + lossp
        return lossg, lossb

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

    def __init__(self, net_params):
        super(DiffPool, self).__init__()
        self.link_pred = net_params['linkpred']
        self.concat = net_params['cat']
        n_pooling = net_params['num_pool']
        self.batch_size = net_params['batch_size']
        self.link_pred_loss = []
        self.entropy_loss = []

        # list of GNN modules before the first diffpool operation
        self.gc_before_pool = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()

        # list of list of GNN modules, each list after one diffpool operation
        self.gc_after_pool = nn.ModuleList()
        self.assign_dim = net_params['assign_dim']
        self.bn = True
        self.assign_num = net_params['assign_num']
        self.num_aggs = 1
        label_dim = net_params['n_classes']
        input_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']

        embedding_dim = net_params['hidden_dim']
        activation = F.relu 
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['gnn_per_block']
        

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

        #self.assign_dim = int(self.assign_dim * pool_ratio)
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
        self.pred_layer = nn.Linear(self.pred_input_dim*self.assign_num, label_dim)

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
        #h = self.gcn_forward_tensorized(
        #    h, adj, self.gc_after_pool[0], self.concat)
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
        return ypred, readout
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
