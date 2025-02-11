# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import List, Optional, Sequence, Tuple
import torch

from detectron2.structures.instances import Instances


from densepose.structures import DensePoseEmbeddingPredictorOutput


from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import DensePoseOutputsVertexVisualizer

from .base import CompoundVisualizer

Scores = Sequence[float]


def extract_scores_from_instances(instances: Instances, select=None):
    if instances.has("scores"):
        return instances.scores if select is None else instances.scores[select]
    return None


def extract_boxes_xywh_from_instances(instances: Instances, select=None):
    if instances.has("pred_boxes"):
        boxes_xywh = instances.pred_boxes.tensor.clone()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        return boxes_xywh if select is None else boxes_xywh[select]
    return None


def create_extractor(visualizer: object):
    """
    Create an extractor for the provided visualizer
    """
    if isinstance(visualizer, CompoundVisualizer):
        extractors = [create_extractor(v) for v in visualizer.visualizers]
        return CompoundExtractor(extractors)
    elif isinstance(visualizer, ScoredBoundingBoxVisualizer):
        return CompoundExtractor([extract_boxes_xywh_from_instances, extract_scores_from_instances])
    elif isinstance(visualizer, DensePoseOutputsVertexVisualizer):
        return DensePoseOutputsExtractor()
    else:
        logger = logging.getLogger(__name__)
        logger.error(f"Could not create extractor for {visualizer}")
        return None

class DensePoseOutputsExtractor(object):
    """
    Extracts DensePose result from instances
    """

    def __call__(
        self,
        instances: Instances,
        select=None,
    ) -> Tuple[
        Optional[DensePoseEmbeddingPredictorOutput], Optional[torch.Tensor], Optional[List[int]]
    ]:
        if not (instances.has("pred_densepose") and instances.has("pred_boxes")):
            return None, None, None
        dpout = instances.pred_densepose
        boxes_xyxy = instances.pred_boxes
        boxes_xywh = extract_boxes_xywh_from_instances(instances)

        if instances.has("pred_classes"):
            classes = instances.pred_classes.tolist()
        else:
            classes = None

        if select is not None:
            dpout = dpout[select]
            boxes_xyxy = boxes_xyxy[select]
            if classes is not None:
                classes = classes[select]

        return dpout, boxes_xywh, classes


class CompoundExtractor(object):
    """
    Extracts data for CompoundVisualizer
    """

    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self, instances: Instances, select=None):
        datas = []
        for extractor in self.extractors:
            data = extractor(instances, select)
            datas.append(data)
        return datas
