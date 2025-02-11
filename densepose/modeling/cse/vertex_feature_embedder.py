# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import torch
from torch import nn

from detectron2.utils.file_io import PathManager

from .utils import normalize_embeddings


class VertexFeatureEmbedder(nn.Module):
    """
    Class responsible for embedding vertex features. Mapping from
    feature space to the embedding space is a tensor of size [K, D], where
        K = number of dimensions in the feature space
        D = number of dimensions in the embedding space
    Vertex features is a tensor of size [N, K], where
        N = number of vertices
        K = number of dimensions in the feature space
    Vertex embeddings are computed as F * E = tensor of size [N, D]
    """

    def __init__(
        self, num_vertices: int, feature_dim: int, embed_dim: int
    ):
        """
        Initialize embedder, set random embeddings

        Args:
            num_vertices (int): number of vertices to embed
            feature_dim (int): number of dimensions in the feature space
            embed_dim (int): number of dimensions in the embedding space
            train_features (bool): determines whether vertex features should
                be trained (default: False)
        """
        super(VertexFeatureEmbedder, self).__init__()

        self.register_buffer("features", torch.Tensor(num_vertices, feature_dim))
        self.embeddings = nn.Parameter(torch.Tensor(feature_dim, embed_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.embeddings.zero_()

    def forward(self) -> torch.Tensor:
        """
        Produce vertex embeddings, a tensor of shape [N, D] where:
            N = number of vertices
            D = number of dimensions in the embedding space

        Return:
           Full vertex embeddings, a tensor of shape [N, D]
        """
        return normalize_embeddings(torch.mm(self.features, self.embeddings))