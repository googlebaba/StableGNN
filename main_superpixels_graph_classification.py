




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self






"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device










"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def create_filename(save_dir, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    filename = os.path.join(save_dir, 'MNIST_GCN_0.999')
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + "_undirected.pth.tar"

def save_checkpoint(model, optimizer, ckptdir, model_name, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.

    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    filename = create_filename(ckptdir, isbest, num_epochs=num_epochs)
    torch.save(
        {
            "epoch": num_epochs,
            "model_type": model_name,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        filename,
    )
def eval(model, device, loader, evaluator, Acc_evaluator, F1_evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch_graphs = batch[0].to(device)
        if batch_graphs.ndata['feat'].size(0) == 1:
            pass
        else:
            with torch.no_grad():
                batch_x = batch_graphs.ndata['feat'].float().to(device)  # num x feat
                batch_e = batch_graphs.edata['feat'].to(device)
                pred, embed = model.forward(batch_graphs, batch_x, batch_e)

            y_true.append(batch[1].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def train_val_pipeline(MODEL_NAME, DATASET_NAME, dataset, params, net_params, dirs, args, out_dir):
    t0 = time.time()
    per_epoch_time = []
        
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
#    for i in range(len(trainset)):
#        print("num of nodes:", dataset.train[i][0].number_of_nodes())
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format("modelcular", MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    #print("Training Graphs: ", len(trainset))
    #print("Validation Graphs: ", len(valset))
    #print("Test Graphs: ", len(biased_testset))
    print("Number of Classes: ", net_params['n_classes'])

    
    split_idx = dataset.get_idx_split()



    model = gnn_model(MODEL_NAME, net_params)
    #model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'])
   
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], [] 
    
    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # import train functions specific for WL-GNNs
        from train.train_superpixels_graph_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network

        train_loader = DataLoader(trainset, shuffle=True, collate_fn=dataset.collate_dense_gnn)
        val_loader = DataLoader(valset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
        biased_test_loader = DataLoader(biased_testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
        unbiased_test_loader = DataLoader(unbiased_testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)

    else:
        # import train functions for all other GCNs
        from train.train_superpixels_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network, train_epoch_sparse_HSIC as train_epoch_HSIC

        ### automatic evaluator. takes dataset name as input


    train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=net_params['batch_size'], shuffle=True, num_workers = 0, drop_last=drop_last)
    valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=net_params['batch_size'], shuffle=True, num_workers = 0, drop_last=drop_last)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=net_params['batch_size'], shuffle=True, num_workers = 0, drop_last=drop_last)


    evaluator = Evaluator(DATASET_NAME)
    Acc_evaluator = Evaluator("ogbg-ppa")
    F1_evaluator = Evaluator("ogbg-ppa")
    #F1_evaluator.eval_metric = "F1"

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print("pretrain")
        valid_curve = []
        test_curve = []
        train_curve = []

        Acc_valid_curve = []
        Acc_test_curve = []
        Acc_train_curve = []

        F1_valid_curve = []
        F1_test_curve = []
        F1_train_curve = []

        best_val = 0
        save_model=True
        #if 'pretrain' in params:
        #    pretrain_epoch = 0
        #else:
        pretrain_epoch = 100
        model = model.to(device)

        with tqdm(range(pretrain_epoch)) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                else:   # for all other models common train function
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, task_type=dataset.task_type)
                train_perf = eval(model, device, train_loader, evaluator, Acc_evaluator, F1_evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator, Acc_evaluator, F1_evaluator)
                test_perf = eval(model, device, test_loader, evaluator, Acc_evaluator, F1_evaluator)


                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

                train_curve.append(train_perf[dataset.eval_metric])
                valid_curve.append(valid_perf[dataset.eval_metric])
                test_curve.append(test_perf[dataset.eval_metric])
                

                if save_model:
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    if valid_perf[dataset.eval_metric] > best_val:
                        torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_best_mocular"))
                        best_val = valid_perf[dataset.eval_metric]


        if pretrain_epoch > 0: 
            if 'classification' in dataset.task_type:
                best_val_epoch = np.argmax(np.array(valid_curve))
                best_train = max(train_curve)
            else:
                best_val_epoch = np.argmin(np.array(valid_curve))
                best_train = min(train_curve)
            print('Finished training!')
            print('Best Auc validation score: {}, Test score: {}'.format(valid_curve[best_val_epoch], test_curve[best_val_epoch]))
            best_val_base = valid_curve[best_val_epoch]
            best_test_base = test_curve[best_val_epoch]
        else:
            pass
            #pretrain_dir = out_dir + "checkpoints/"+ params["pretrain"]
            #ckpt_dir = os.path.join(pretrain_dir, "RUN_")


        model.train()
        model = model.to(device)
        #model.load_state_dict(torch.load('{}.pkl'.format(ckpt_dir + "/epoch_best_mocular")))
                        
        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)


    
        print("ourmodel start!")
        train_curve = []
        valid_curve = []
        test_curve = []
        with tqdm(range(100)) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                else:   # for all other models common train function

                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch_HSIC(model, optimizer, net_params['hidden_dim'], net_params["assign_num"], device, train_loader, epoch, params["lrbl"], params["lambdap"], args.lambda_decay_rate, args.lambda_decay_epoch, args.min_lambda_times, task_type=dataset.task_type)

                train_perf = eval(model, device, train_loader, evaluator, Acc_evaluator, F1_evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator, Acc_evaluator, F1_evaluator)
                test_perf = eval(model, device, test_loader, evaluator, Acc_evaluator, F1_evaluator)


                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
                train_curve.append(train_perf[dataset.eval_metric])
                valid_curve.append(valid_perf[dataset.eval_metric])
                test_curve.append(test_perf[dataset.eval_metric])
                

        if 'classification' in dataset.task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)

        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)


        print('Finished training!')
        print('Best Auc validation score: {}, Test score: {}'.format(valid_curve[best_val_epoch], test_curve[best_val_epoch]))

        best_val_our = valid_curve[best_val_epoch]
        best_test_our = test_curve[best_val_epoch]
        

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nbest_validation_base: {:.4f}\n\nbest_test_base: {:.4f}\n\nbest_validation_our: {:.4f}\n\nbest_test_base: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                   best_val_base*100,  best_test_base*100, best_val_our*100, best_test_our*100, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
 

def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--train_str', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
    parser.add_argument('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
    parser.add_argument('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')
    parser.add_argument('--lambdap', type = float, default = 2.0, help = 'weight decay for weight1 ')
    parser.add_argument('--lrbl', type = float, default = 1.0, help = 'weight decay for weight1 ')
    parser.add_argument('--pretrain', help = 'weight decay for weight1 ')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    dataset = DglGraphPropPredDataset(name = args.dataset)

    print("data 0 ", dataset[0])

    #dataset = LoadData(DATASET_NAME, args.train_str, '1_biased', '0.1_unbiased')
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.lrbl is not None:
        params['lrbl'] = float(args.lrbl)
    if args.lambdap is not None:
        params['lambdap'] = float(args.lambdap)
    if args.pretrain is not None:
        params['pretrain'] = args.pretrain

    # network parameters
    net_params = config['net_params']

    net_params['in_dim'] = dataset[0][0].ndata['feat'].size(1)
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)


    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
        
    # Superpixels
    #net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    #net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    net_params['n_classes'] = dataset.num_tasks

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        max_num_nodes_train = max([dataset[i][0].number_of_nodes() for i in range(len(dataset))])
        max_num_node = max_num_nodes_train
        #net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']
        #net_params["assign_num"] = int(max_num_node * net_params['pool_ratio'])

        net_params["assign_num"] = 7
        net_params['assign_dim'] = net_params["assign_num"] * net_params['batch_size']
    else:
        net_params["assign_num"] = 0
        net_params["pool_ratio"] = 0
        
        
    if MODEL_NAME == 'RingGNN':
        num_nodes_train = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))]
        num_nodes = num_nodes_train + num_nodes_test
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
        
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') +'_'+str(args.train_str)+"_"+str(params['seed'])
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')  +'_'+str(args.train_str)+"_"+str(params['seed'])
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')  +'_'+str(args.train_str) + '_assign_ratio_'+str(net_params['pool_ratio']) +"_"+str(params['seed'])
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') +'_'+str(args.train_str) +"_"+str(params['seed'])
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, args.dataset, dataset, params, net_params, dirs, args, out_dir)

    
    
    
main()    
















