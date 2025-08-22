#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
tsne-torch: a tSNE implementation based on pytorch using cuda.

Based on tsne-pytorch created by Xiao Li on 23-03-2020.
Copyright (c) 2020, 2021, 2025 Xiao Li, Palle Klewitz, Alex Müller.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def h_beta_torch(D: torch.Tensor, beta: float = 1.0, device: str = "cuda"):
    D = D.to(device)
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    return H, P / sumP


def x2p_torch(
    X: torch.Tensor, tol: float = 1e-5, perplexity: float = 30.0, verbose: bool = False
):
    """
    Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = X.shape[0]
    sum_X = torch.sum(X * X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X).to(device)

    P = torch.zeros(n, n).to(device)
    beta = torch.ones(n, 1).to(device)
    logU = torch.log(torch.tensor([perplexity])).to(device)
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    bar = range(n)
    if verbose:
        bar = tqdm(bar)

    for i in bar:
        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin, betamax = None, None

        Di = D[i, n_list[0:i] + n_list[i + 1 : n]]
        (H, thisP) = h_beta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = h_beta_torch(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i] + n_list[i + 1 : n]] = thisP

    return P


def pca_torch(X, no_dims=2, device="cuda"):
    X = X.to(device)
    _, _, V = torch.pca_lowrank(X)
    return torch.matmul(X, V[:, :no_dims].to(device))


def tsne(
    X,
    no_dims=2,
    perplexity=30.0,
    max_iter=1000,
    iter_explore=100,
    initial_dims=50,
    early_exager=12.0,
    lr=500.0,
    momentum_init=0.5,
    momentum_final=0.8,
    seed=42,
    verbose=False,
):
    """
    Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if verbose:
        print(f"Using random seed {seed}")
        print(f"Using device {device}")
        print("Running initial PCA...")

    X = pca_torch(X, initial_dims, device=device)
    n = X.shape[0]
    min_gain = 0.01
    Y = torch.randn(n, no_dims).to(device)
    dY = torch.zeros(n, no_dims).to(device)
    iY = torch.zeros(n, no_dims).to(device)
    gains = torch.ones(n, no_dims).to(device)

    # Compute P-values
    if verbose:
        print("Computing initial p-values...")
    P = x2p_torch(X, 1e-5, perplexity).to(device)
    P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0, out=P)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * early_exager  # early exaggeration
    P = torch.max(P, torch.tensor([1e-21]).to(device))

    # Run iterations
    if verbose:
        print("Fitting tSNE...")
    bar = range(max_iter)
    if verbose:
        bar = tqdm(bar)
    for i in bar:
        # Compute pairwise affinities
        sum_Y = torch.sum(Y * Y, 1)
        num = -2.0 * torch.mm(Y, Y.t())
        num = 1.0 / (1.0 + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.0
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]).to(device))

        # Compute gradient
        PQ = P - Q
        dY = torch.sum(
            (PQ * num).unsqueeze(1).repeat(1, no_dims, 1).transpose(2, 1)
            * (Y.unsqueeze(1) - Y),
            dim=1,
        )

        # Perform the update
        if i < iter_explore:
            momentum = momentum_init
        else:
            momentum = momentum_final
        if verbose and i == iter_explore:
            print(f"Switching momentum to {momentum}")

        gains = (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)).float() + (gains * 0.8) * (
            (dY > 0.0) == (iY > 0.0)
        ).float()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - lr * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if verbose:
            C = torch.sum(P * torch.log(P / Q))
            bar.set_description(f"Error: {C.cpu().item():.3f}")

        # finish exageration
        if i == iter_explore:
            P = P / early_exager

    # Return solution
    return Y if device == "cpu" else Y.cpu().numpy()


class TorchTSNE:

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        max_iter=1000,
        iter_explore=100,
        initial_dims=50,
        early_exager=12.0,
        lr=500.0,
        momentum_init=0.5,
        momentum_final=0.8,
        random_state=42,
    ):
        self._embedding = None
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.iter_explore = iter_explore
        self.initial_dims = initial_dims
        self.early_exager = early_exager
        self.lr = lr
        self.momentum_init = momentum_init
        self.momentum_final = momentum_final
        self.random_state = random_state

    def fit_transform(self, x, verbose=False):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            self._embedding = tsne(
                x,
                max_iter=self.max_iter,
                iter_explore=self.iter_explore,
                perplexity=self.perplexity,
                lr=self.lr,
                early_exager=self.early_exager,
                seed=self.random_state,
                verbose=verbose,
            )
        return self._embedding


def main(xfile, yfile, iter, perplex, lr, exager, seed, verbose):
    X = torch.Tensor(np.loadtxt(xfile))
    labels = np.loadtxt(yfile).tolist()

    # confirm that x file get same number point than label file
    assert len(X) == len(labels), "different number of datapoints and labels"

    with torch.no_grad():
        Y = tsne(
            X,
            max_iter=iter,
            iter_explore=iter // 10,
            perplexity=perplex,
            lr=lr,
            early_exager=exager,
            seed=seed,
            verbose=verbose,
        )

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("images/pytorch.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--xfile",
        type=str,
        default="data/mnist2500_X.txt",
        help="file name of feature stored",
    )
    parser.add_argument(
        "-y",
        "--yfile",
        type=str,
        default="data/mnist2500_labels.txt",
        help="file name of label stored",
    )
    parser.add_argument(
        "-i", "--iter", type=int, help="number of iterations", default=1000
    )
    parser.add_argument("-p", "--perplex", type=float, help="perplexity", default=30.0)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=500.0)
    parser.add_argument(
        "-e", "--exager", type=float, help="early exageration", default=4.0
    )
    parser.add_argument("-s", "--seed", type=int, help="random seed", default=42)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose output"
    )

    args = parser.parse_args()
    main(
        args.xfile,
        args.yfile,
        args.iter,
        args.perplex,
        args.lr,
        args.exager,
        args.seed,
        args.verbose,
    )
