# t-SNE pytorch Implementation with CUDA
CUDA-accelerated PyTorch implementation of the t-stochastic neighbor embedding algorithm described in [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf). 

## Installation

Requires Python 3.7

### Install via Pip

```bash
pip3 install tsne-torch
```

### Install from Source

```bash
git clone https://github.com/palle-k/tsne-pytorch.git
cd tsne-pytorch
python3 setup.py install
```

## Usage

```python
from tsne_torch import TorchTSNE as TSNE

X = ...  # shape (n_samples, d)
X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(X)  # returns shape (n_samples, 2)
```

## Command-Line Usage

```bash
python3 -m tsne_torch --xfile <path> --yfile <path>
```

## Example

This is our result compare to result of python implementation.
* PyTorch result

![pytorch result](images/pytorch.png)
* python result

![python result](images/python.png)

## Credit
This code highly inspired by 
* author's python implementation code [here](https://lvdmaaten.github.io/tsne/).
