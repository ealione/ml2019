import torch
import numpy as np


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def zca_whitening_matrix(x):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix

    USAGE:
    >> X = np.array([[0, 2, 2], [1, 1, 0], [2, 0, 1], [1, 3, 5], [10, 10, 10] ]) # Input: X [5 x 3] matrix
    >> ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
    >> ZCAMatrix # [5 x 5] matrix
    >> xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
    >> xZCAMatrix # [5 x 3] matrix

    https://stackoverflow.com/questions/57709758/using-transforms-lineartransformation-to-apply-whitening-in-pytorch
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(x, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    u, s, v = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + epsilon)), u.T))  # [M x M]
    return ZCAMatrix


def load_whitened_dataset(path, train=True):
    with np.load(path) as data:
        if train:
            x, y = tuple(data[k] for k in ('train_x', 'train_y'))
        else:
            x, y = tuple(data[k] for k in ('test_x', 'test_y'))
        return np.float32(x)[:300], np.int64(y)[:300]
