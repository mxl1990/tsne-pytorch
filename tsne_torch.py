#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch



def Hbeta_torch(D, beta=1.0):
    P = (-D).exp() * beta
    SP = P.sum()
    H = SP.log() + beta * (D * P).sum() / SP
    P /= SP
    return H, P


def x2p_torch(X:torch.FloatTensor, tol=1e-5, perplexity=30.0, verbose=False):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    if verbose:
        print("Computing pairwise distances")
    M = X.size(0)

    SSX = (X**2).sum(1)
    D = ((X @ X.T) * -2 + SSX).T + SSX

    P = X.new_zeros(M, M)
    beta = X.new_ones(M, 1)
    logU = torch.tensor([perplexity]).type_as(X).log()
    indices = torch.arange(M)

    # Loop over all datapoints
    for i in range(M):

        # Print progress
        if i % 500 == 0 and verbose:
            print(f"Computing P-values for point {i} of {M}")

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, torch.cat([indices[:i], indices[i+1:M]])]
        H, thisP = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] *= 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] /= 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            H, thisP = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, torch.cat([indices[:i], indices[i+1:M]])] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, k=50, device="cuda"):
    assert X.ndim == 2, "X should be a 2D array"
    *_, V = torch.pca_lowrank(X)
    return X @ V[:, :k]


@torch.no_grad()
def tsne(
    X:torch.FloatTensor, 
    k:int=2, 
    initial_dims:int=50, 
    perplexity:float=30.0, 
    max_iter:int=1000,
    initial_momentum:float=0.5,
    final_momentum:float=0.8,
    eta:float=500,
    min_gain:float=0.01,
    eps:float=1e-12,
    device="cuda", 
    verbose=False
) -> torch.FloatTensor:
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to k dimensions. Usage:
        `Y = tsne.tsne(X, k, perplexity)`
    """

    # Check inputs
    if not isinstance(k, int):
        raise TypeError("k is expected to be an integer")
    if not isinstance(initial_dims, int):
        raise TypeError("initial_dims is expected to be an integer")

    # Initialize variables
    if X.size(1) > initial_dims:
        if verbose:
            print("Preprocessing the data using PCA")
        X = pca_torch(X, initial_dims)

    M = X.size(0)
    X = X.to(device)
    Y = torch.randn(M, k).to(device)
    dY = torch.zeros_like(Y)
    iY = torch.zeros_like(Y)
    gains = torch.ones_like(Y)
    eps = torch.tensor([eps]).type_as(X)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity, verbose)
    P += P.T.clone()
    P /= P.sum()
    P *= 4      # early exaggeration
    P = torch.max(P, eps)

    # Run iterations
    for iter in range(1, max_iter + 1):

        # Compute pairwise affinities
        SSY = (Y ** 2).sum(1)
        num = -2. * (Y @ Y.T)
        num = (1 + (num + SSY).T + SSY) ** -1
        num[torch.arange(M), torch.arange(M)] = 0.
        Q = num / num.sum()
        Q = torch.max(Q, eps)

        # Compute gradient
        PQ = P - Q
        for i in range(M):
            dY[i, :] = ((PQ[:, i] * num[:, i]).repeat(k, 1).T * (Y[i, :] - Y)).sum(0)

        # Perform the update
        if iter <= 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        mask = (dY > 0) == (iY > 0)
        gains = (gains * 0.2) * (~mask).float() + (gains * 0.8) * mask.float()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * gains * dY
        Y += iY
        Y -= Y.mean(0)

        # Compute current value of cost function
        if iter % 10 == 0:
            C = (P * (P / Q).log()).sum()
            if verbose:
                print(f"Iteration {iter}: error is {C}")

        # Stop lying about P-values
        if iter == 100:
            P /= 4

    # Return solution
    return Y.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="test_data/iris_features.npy", help="features file")
    parser.add_argument("--labels", type=str, default="test_data/iris_labels.npy", help="label file")
    parser.add_argument("--ppl", type=float, default=20., help="Perplexity")
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda")
    parser.add_argument("--output_file", type=str, default=None, help="output plot file")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available else "cpu")
    X = torch.from_numpy(np.loadtxt(args.features))
    labels = np.loadtxt(args.labels)

    # confirm that x file get same number point than label file
    assert len(X) == len(labels)

    Y = tsne(X, 2, 50, args.ppl, device=device).numpy()

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    (args.output_file and plt.savefig(args.output_file)) or plt.show()
