###### repository of paper 《DGC: Dual-Graph Attention Network with Contrastive Representation Generation for Emotion Recognition in Conversations》



###### &nbsp;We are in the process of polishing our code. File uploads and detailed documentation will be completed in subsequent commits.



## Requirements

Python  3.8(ubuntu20.04)
CUDA  11.3

matplotlib                3.9.4                    pypi\_0    pypi

numpy                     1.23.5                   pypi\_0    pypi

pandas                    2.2.3                    pypi\_0    pypi

python                    3.9.20               he870216\_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

scikit-learn              1.5.2                    pypi\_0    pypi

scipy                     1.13.1                   pypi\_0    pypi

seaborn                   0.13.2                   pypi\_0    pypi

tokenizers                0.21.0                   pypi\_0    pypi

torch                     1.13.1+cu117             pypi\_0    pypi

torchaudio                0.13.1+cu117             pypi\_0    pypi

torchvision               0.14.1+cu117             pypi\_0    pypi

tqdm                      4.67.1                   pypi\_0    pypi

transformers              4.47.0                   pypi\_0    pypi

vocab                     0.0.5                    pypi\_0    pypi



## Run Steps

1. Please download the three public benchmark ERC datasets
2. Run the CRGC module in data folder:
3. Run the main.py in root.

## Command

python main.py --dataset\_name EmoryNLP --batch\_size 32 --epochs 150 --lr 1e-4 --do\_CA 1 --diff\_rating 0.4 --dropout 0.2 --gnn\_layers 4 --diff\_layers 3

python main.py --dataset\_name MELD --batch\_size 32 --epochs 150 --lr 5e-6 --do\_CA 1 --diff\_rating 0.1 --dropout 0.4 --gnn\_layers 1 --diff\_layers 4

python main.py --dataset\_name IEMOCAP --batch\_size 8 --epochs 150 --lr 1e-4 --do\_CA 1 --diff\_rating 0.4 --dropout 0.4 --gnn\_layers 1 --diff\_layers 5

