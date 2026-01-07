repository of paper 《DGC: Dual-Graph Attention Network with Contrastive Representation Generation for Emotion Recognition in Conversations》



## Requirements

Python  3.8(ubuntu20.04)
CUDA  11.3



## Run Steps

1. Please download the three public benchmark ERC datasets
2. Run the CRGC module in data folder:
3. Run the main.py in root.

## Command

python main.py --dataset\_name EmoryNLP --batch\_size 32 --epochs 150 --lr 1e-4 --do\_CA 1 --diff\_rating 0.4 --dropout 0.2 --gnn\_layers 4 --diff\_layers 3

python main.py --dataset\_name MELD --batch\_size 32 --epochs 150 --lr 5e-6 --do\_CA 1 --diff\_rating 0.1 --dropout 0.4 --gnn\_layers 1 --diff\_layers 4

python main.py --dataset\_name IEMOCAP --batch\_size 8 --epochs 150 --lr 1e-4 --do\_CA 1 --diff\_rating 0.4 --dropout 0.4 --gnn\_layers 1 --diff\_layers 5





\# We are in the process of polishing our code. File uploads and detailed documentation will be completed in subsequent commits.

