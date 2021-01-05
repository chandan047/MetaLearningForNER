import copy
import numpy as np

import pdb

import torch
import torch.nn as nn
from sklearn import metrics
from seqeval import metrics as seq_metrics

def calculate_seqeval_metrics(predictions, labels, tags=None, binary=False):
    if tags is not None:
        map2label = {v:k for k,v in tags.items()}
        # pdb.set_trace()
        for i in range(len(predictions)):
            predictions[i] = [map2label[v] for v in predictions[i]]
            labels[i] = [map2label[v] for v in labels[i]]
    
    accuracy = seq_metrics.accuracy_score(labels, predictions)
    precision = seq_metrics.precision_score(labels, predictions)
    recall = seq_metrics.recall_score(labels, predictions)
    f1_score = seq_metrics.f1_score(labels, predictions)
    return accuracy, precision, recall, f1_score


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = torch.stack(predictions).cpu().numpy()
    labels = torch.stack(labels).cpu().numpy()
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels)
    return accuracy, precision, recall, f1_score


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def replicate_model_to_gpus(model, device_ids):
    replica_models = [model] + [copy.deepcopy(model).to(device) for device in device_ids[1:]]
    for rm in replica_models[1:]:
        rm.device = next(rm.parameters()).device
    return replica_models


class EuclideanDistance(nn.Module):
    """Implement a EuclideanDistance object."""

    def forward(self, mat_1, mat_2):
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.
        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)
        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)
        """
        _dist = [torch.sum((mat_1 - mat_2[i])**2, dim=1) for i in range(mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return dist


class EuclideanMean(nn.Module):
    """Implement a EuclideanMean object."""

    def forward(self, data):
        """Performs a forward pass through the network.
        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor
        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor
        """
        return data.mean(0)
    

EPSILON = 1e-5


def arccosh(x):
    """Compute the arcosh, numerically stable."""
    x = torch.clamp(x, min=1 + EPSILON)
    a = torch.log(x)
    b = torch.log1p(torch.sqrt(x * x - 1) / x)
    return a + b


def mdot(x, y):
    """Compute the inner product."""
    m = x.new_ones(1, x.size(1))
    m[0, 0] = -1
    return torch.sum(m * x * y, 1, keepdim=True)


def dist(x, y):
    """Get the hyperbolic distance between x and y."""
    return arccosh(-mdot(x, y))


def project(x):
    """Project onto the hyeprboloid embedded in in n+1 dimensions."""
    return torch.cat([torch.sqrt(1.0 + torch.sum(x * x, 1, keepdim=True)), x], 1)


def log_map(x, y):
    """Perform the log step."""
    d = dist(x, y)
    return (d / torch.sinh(d)) * (y - torch.cosh(d) * x)


def norm(x):
    """Compute the norm"""
    n = torch.sqrt(torch.abs(mdot(x, x)))
    return n


def exp_map(x, y):
    """Perform the exp step."""
    n = torch.clamp(norm(y), min=EPSILON)
    return torch.cosh(n) * x + (torch.sinh(n) / n) * y


def loss(x, y):
    """Get the loss for the optimizer."""
    return torch.sum(dist(x, y)**2)


class HyperbolicDistance(nn.Module):
    """Implement a HyperbolicDistance object.
    """

    def forward(self, mat_1, mat_2):
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.
        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)
        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)
        """
        # Get projected 1st dimension
        mat_1_x_0 = torch.sqrt(1 + mat_1.pow(2).sum(dim=1, keepdim=True))
        mat_2_x_0 = torch.sqrt(1 + mat_2.pow(2).sum(dim=1, keepdim=True))

        # Compute bilinear form
        left = mat_1_x_0.mm(mat_2_x_0.t())  # n_1 x n_2
        right = mat_1[:, 1:].mm(mat_2[:, 1:].t())  # n_1 x n_2

        # Arcosh
        return arccosh(left - right).pow(2)


class HyperbolicMean(nn.Module):
    """Compute the mean point in the hyperboloid model."""

    def forward(self, data):
        """Performs a forward pass through the network.
        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor
        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor
        """
        n_iter = 5 if self.training else 100

        # Project the input data to n+1 dimensions
        projected = project(data)

        mean = torch.mean(projected, 0, keepdim=True)
        mean = mean / norm(mean)

        r = 1e-2
        for i in range(n_iter):
            g = -2 * torch.mean(log_map(mean, projected), 0, keepdim=True)
            mean = exp_map(mean, -r * g)
            mean = mean / norm(mean)

        # The first dimension, is recomputed in the distance module
        return mean.squeeze()[1:]
