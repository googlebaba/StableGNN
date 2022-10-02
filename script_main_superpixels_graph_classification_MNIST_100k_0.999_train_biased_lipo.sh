#!/bin/bash


############
# Usage
############

# bash script_main_superpixels_graph_classification_MNIST_100k.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GatedGCN
#GAT
#MoNet
#GIN
#3WLGNN
#RingGNN



############
# MNIST - 4 RUNS  
############

seed0=41
seed1=95
seed2=15
seed3=38
seed4=52
train_str=MNIST_BIASED_0.995
test_str=1_biased
code=main_superpixels_graph_classification.py

str1="benchmark_mollipo" 
tmux new -s $str1 -d
tmux send-keys "source activate benchmark_gnn" C-m
#dataset=ogbg-molesol
#dataset=ogbg-molfreesolv

dataset=ogbg-mollipo
lrbl=1
lambdap=2


tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --train_str $train_str --seed $seed0 --lrbl $lrbl --lambdap $lambdap --config 'configs/superpixels_graph_classification_DiffPool_Mollipo_100k.json' &
#python $code --dataset $dataset --gpu_id 1 --train_str $train_str --seed $seed1 --lrbl $lrbl --lambdap $lambdap --config 'configs/superpixels_graph_classification_DiffPool_Mollipo_100k.json' &
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed2 --lrbl $lrbl --lambdap $lambdap --config 'configs/superpixels_graph_classification_DiffPool_Mollipo_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --seed $seed3 --lrbl $lrbl --lambdap $lambdap --config 'configs/superpixels_graph_classification_DiffPool_Mollipo_100k.json' &
#python $code --dataset $dataset --gpu_id 0 --train_str $train_str --seed $seed4 --lrbl $lrbl --lambdap $lambdap --config 'configs/superpixels_graph_classification_DiffPool_Mollipo_100k.json' &
wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed0 --config 'configs/superpixels_graph_classification_GraphSage_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --seed $seed1 --config 'configs/superpixels_graph_classification_GraphSage_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 4 --train_str $train_str --seed $seed2 --config 'configs/superpixels_graph_classification_GraphSage_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 5 --train_str $train_str --seed $seed3 --config 'configs/superpixels_graph_classification_GraphSage_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --seed $seed4 --config 'configs/superpixels_graph_classification_GraphSage_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed0 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --seed $seed1 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST_100k.json' &
##python $code --dataset $dataset --gpu_id 7 --train_str $train_str --seed $seed2 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 8 --train_str $train_str --seed $seed3 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed0 --config 'configs/superpixels_graph_classification_GAT_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --seed $seed1 --config 'configs/superpixels_graph_classification_GAT_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 7 --train_str $train_str --seed $seed2 --config 'configs/superpixels_graph_classification_GAT_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 8 --train_str $train_str --seed $seed3 --config 'configs/superpixels_graph_classification_GAT_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed0 --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed1 --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 7 --train_str $train_str --seed $seed2 --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 8 --train_str $train_str --seed $seed3 --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 7 --train_str $train_str --test_str $test_str --seed $seed0 --config 'configs/superpixels_graph_classification_3WLGNN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 8 --train_str $train_str --test_str $test_str --seed $seed1 --config 'configs/superpixels_graph_classification_3WLGNN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --test_str $test_str --seed $seed2 --config 'configs/superpixels_graph_classification_3WLGNN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --test_str $test_str --seed $seed3 --config 'configs/superpixels_graph_classification_3WLGNN_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 7 --train_str $train_str --test_str $test_str --seed $seed0 --config 'configs/superpixels_graph_classification_RingGNN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 8 --train_str $train_str --test_str $test_str --seed $seed1 --config 'configs/superpixels_graph_classification_RingGNN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --test_str $test_str --seed $seed2 --config 'configs/superpixels_graph_classification_RingGNN_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --test_str $test_str --seed $seed3 --config 'configs/superpixels_graph_classification_RingGNN_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --train_str $train_str --seed $seed0 --config 'configs/superpixels_graph_classification_MoNet_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 3 --train_str $train_str --seed $seed1 --config 'configs/superpixels_graph_classification_MoNet_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 7 --train_str $train_str --seed $seed2 --config 'configs/superpixels_graph_classification_MoNet_MNIST_100k.json' &
#python $code --dataset $dataset --gpu_id 8 --train_str $train_str --seed $seed3 --config 'configs/superpixels_graph_classification_MoNet_MNIST_100k.json' &
#wait" C-m
#tmux send-keys "tmux kill-session -t $str1" C-m








