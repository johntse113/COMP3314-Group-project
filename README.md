# COMP3314-Group-project
This repo is for the group project of COMP3314.<br />
This undergraduate project aims to reproduce the experiment results in a research paper called "LoRANN: Low-Rank Matrix Factorization for
Approximate Nearest Neighbor Search" from 38th Conference on Neural Information Processing Systems (NeurIPS 2024): https://arxiv.org/abs/2410.18926

# Prerequisites 
Please install the following libraries according to their Github repos:<br />
ScaNN:<br />
https://github.com/google-research/google-research/tree/master/scann<br />
Glass:<br />
https://github.com/zilliztech/pyglass<br />
Faiss-CPU:<br />
https://github.com/facebookresearch/faiss<br />
LoRANN:<br />
https://github.com/ejaasaari/lorann<br />

# How to run
Run the code directly using Python, and add one of the available dataset names as an argument.<br />
```
python3 run.py <dataset>
```
e.g. <br />
```
python3 run.py sift-128-euclidean
```

All available dataset names:<br />
```
sift-128-euclidean
gist-960-euclidean
glove-100-angular
glove-200-angular
nytimes-256-angular
lastfm-64-dot
deep-image-96-angular
fashion-mnist-784-euclidean
mnist-784-euclidean
coco-i2i-512-angular
coco-t2i-512-angular
```
The datasets are from the ANN Benchmarks repo:<br />
https://github.com/erikbern/ann-benchmarks
<br />
Please note that LoRANN only supports L2 or Inner Product distances. Although 2 of the datasets from the ANN Benchmarks repo: MovieLens-10M and Kosarak, can be used in the program, they are not supported by LoRANN (and also Faiss) since they use Jaccard. The results produced by the program using these 2 datasets will not be accurate. In addition, LoRANN also requires the datasets to be at least 64 in dimension, so GloVe-25 and GloVe-50 from the ANN Benchmarks repo cannot be used.<br />


