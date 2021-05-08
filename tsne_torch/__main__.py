from tsne_torch import TorchTSNE


def main(xfile: str, yfile: str):
    import torch
    import numpy as np
    from matplotlib import pyplot

    X = np.loadtxt(xfile)
    X = torch.Tensor(X)
    labels = np.loadtxt(yfile).tolist()

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0]) == len(X[:, 1]))
    assert(len(X) == len(labels))

    Y = TorchTSNE(n_components=2, perplexity=20.0, initial_dims=50, verbose=True).fit_transform(X)
    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pyplot.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xfile", type=str, default="mnist2500_X.txt", help="file name of feature vectors")
    parser.add_argument("--yfile", type=str, default="mnist2500_labels.txt", help="file name of corresponding labels")

    opt = parser.parse_args()
    xfile = opt.xfile
    yfile = opt.yfile

    main(xfile, yfile)
