repository of paper 《DGC: Dual-Graph Attention Network with Contrastive Representation Generation for Emotion Recognition in Conversations》


## Requirements

Python  3.8(ubuntu20.04)
CUDA  11.3


## Run Steps

1. Please download the three public benchmark ERC datasets 
2. Run the CRGC module in data folder:
3. Run the main.py in root.

## Command
python main.py --dataset_name EmoryNLP --batch_size 32 --epochs 150 --lr 1e-4 --do_CA 1 --diff_rating 0.4 --dropout 0.2 --gnn_layers 4 --diff_layers 3

python main.py --dataset_name MELD --batch_size 32 --epochs 150 --lr 5e-6 --do_CA 1 --diff_rating 0.1 --dropout 0.4 --gnn_layers 1 --diff_layers 4

python main.py --dataset_name IEMOCAP --batch_size 8 --epochs 150 --lr 1e-4 --do_CA 1 --diff_rating 0.4 --dropout 0.4 --gnn_layers 1 --diff_layers 5



