# StableGNN
StableGNN-Generalizing Graph Neural Networks on Out-Of-Distribution Graphs

Paper:https://arxiv.org/pdf/2111.10657.pdf

Connection

Shaohua Fan: fanshaohua@bupt.cn

Note that

If you want to use stableGNN on other non-molecule datasets, replace https://github.com/googlebaba/StableGNN/blob/c7ca1a3dca805ad1238dc452b1a225216014323a/nets/superpixels_graph_classification/diffpool_net.py#L176 with h= self.embedding_h(h)

And finetune lrbl in .sh may get better results.
