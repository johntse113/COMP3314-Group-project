# COMP3314-Group-project
This repo is for the group project of COMP3314.
This undergraduate project aims to reproduce the experiment results in a research paper called "LoRANN: Low-Rank Matrix Factorization for
Approximate Nearest Neighbor Search" from 38th Conference on Neural Information Processing Systems (NeurIPS 2024): https://arxiv.org/abs/2410.18926

# Prerequisites 
Please install the following libraries according to their Github repos:
ScaNN:
https://github.com/google-research/google-research/tree/master/scann
Glass:
https://github.com/zilliztech/pyglass
Faiss-CPU:
https://github.com/facebookresearch/faiss
LoRANN:
https://github.com/ejaasaari/lorann

# How to run
Run the code directly using Python, and add one of the available dataset names as an argument.
'''
python3 main.py <dataset>
'''
e.g. 
'''
python3 main.py sift-128-euclidean
'''

All available dataset names:
'''
sift-128-euclidean
gist-960-euclidean
glove-25-angular
glove-50-angular
glove-100-angular
glove-200-angular
kosarak-jaccard
movielens10m-jaccard
nytimes-256-angular
lastfm-64-dot
deep-image-96-angular
fashion-mnist-784-euclidean
mnist-784-euclidean
coco-i2i-512-angular
coco-t2i-512-angular
'''

The datasets are from the ANN Benchmarks repo:
https://github.com/erikbern/ann-benchmarks
