# t-SNE pytorch Implementation with CUDA

A [pytorch](https://pytorch.org) implementation of the tSNE algorithm described in [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) with CUDA acceleration.

## Installation

Clone this repository and install the required packages using [uv](https://docs.astral.sh/uv) as follows:

```bash
uv sync
```

This will install the required packages defined in the `pyproject.toml` file (or if available, `uv.lock`).

To allow for importing `TorchTSNE` in other projects, add the path of the repository root to your PYTHONPATH as follows:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## How to use it

Just download the repository, and the unzip mnist2500_X.zip or put feature file and labels file with code.

Then run `tsne_torch.py` given the desired inputs (and the `-v` verbose option to see what's happening under the hood):

```bash
python tsne_torch.py -x mnist2500_X.txt -y mnist2500_labels.txt -v
```

The script will automatically detect if a CUDA enabled GPU is available and make use of it.

***Note:*** The input data should be normalized to the range [0.0, 1.0], otherwise you may get the 'nan' result.


## Credit

This code is inspired by https://lvdmaaten.github.io/tsne
