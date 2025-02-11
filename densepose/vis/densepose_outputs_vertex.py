# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# densepose_outputs_vertex-og.py

import json
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import cv2
import torch

from detectron2.utils.file_io import PathManager

from densepose.modeling import build_densepose_embedder
from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES

from ..data.utils import get_class_to_mesh_name_mapping
from ..structures import DensePoseEmbeddingPredictorOutput
from ..structures.mesh import create_mesh
from .base import Boxes, Image, MatrixVisualizer


@lru_cache()
def get_xyz_vertex_embedding(mesh_name: str, device: torch.device):
    if mesh_name == "smpl_27554":
        embed_path = PathManager.get_local_path(
            "https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"
        )
        embed_map, _ = np.load(embed_path, allow_pickle=True)
        embed_map = torch.tensor(embed_map).float()[:, 0]
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
    else:
        mesh = create_mesh(mesh_name, device)
        embed_map = mesh.vertices.sum(dim=1)
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
        embed_map = embed_map**2
    return embed_map


class DensePoseOutputsVertexVisualizer(object):
    def __init__(
        self,
        cfg,
        inplace=True,
        cmap=cv2.COLORMAP_JET,
        alpha=0.7,
        device="cuda",
        default_class=0,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.embedder = build_densepose_embedder(cfg)
        self.device = torch.device(device)
        self.default_class = default_class

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
            if self.embedder.has_embeddings(mesh_name)
        }

    def visualize(
        self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[
            Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]
        ],
    ) -> Image:
        if outputs_boxes_xywh_classes[0] is None:
            return image_bgr

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(
            outputs_boxes_xywh_classes
        )

        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().tolist()
            mesh_name = self.class_to_mesh_name[pred_classes[n]]
            closest_vertices, mask = get_closest_vertices_mask_from_ES(
                E[[n]],
                S[[n]],
                h,
                w,
                self.mesh_vertex_embeddings[mesh_name],
                self.device,
            )
            embed_map = get_xyz_vertex_embedding(mesh_name, self.device)
            ## device mismatch on GPU and CPU
            device = embed_map.device  # Get the device of embed_map
            closest_vertices = closest_vertices.to(device)  # Move closest_vertices to the same device
            vis = (embed_map[closest_vertices].clip(0, 1) * 255.0).cpu().numpy()
            mask_numpy = mask.cpu().numpy().astype(dtype=np.uint8)
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask_numpy, vis, [x, y, w, h])

        return image_bgr

    def extract_and_check_outputs_and_boxes(self, outputs_boxes_xywh_classes):

        densepose_output, bboxes_xywh, pred_classes = outputs_boxes_xywh_classes

        if pred_classes is None:
            pred_classes = [self.default_class] * len(bboxes_xywh)

        assert isinstance(
            densepose_output, DensePoseEmbeddingPredictorOutput
        ), "DensePoseEmbeddingPredictorOutput expected, {} encountered".format(
            type(densepose_output)
        )

        S = densepose_output.coarse_segm
        E = densepose_output.embedding
        N = S.size(0)
        assert N == E.size(
            0
        ), "CSE coarse_segm {} and embeddings {}" " should have equal first dim size".format(
            S.size(), E.size()
        )
        assert N == len(
            bboxes_xywh
        ), "number of bounding boxes {}" " should be equal to first dim size of outputs {}".format(
            len(bboxes_xywh), N
        )
        assert N == len(pred_classes), (
            "number of predicted classes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )

        return S, E, N, bboxes_xywh, pred_classes
