# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# pyre-ignore-all-errors

from detectron2.config import CfgNode as CN

# needed 
def add_dataset_category_config(cfg: CN) -> None:
    """
    Add config for additional category-related dataset options
     - category whitelisting
     - category mapping
    """
    _C = cfg
    # _C.DATASETS.CATEGORY_MAPS = CN(new_allowed=True)
    # _C.DATASETS.WHITELISTED_CATEGORIES = CN(new_allowed=True)
    # class to mesh mapping
    _C.DATASETS.CLASS_TO_MESH_NAME_MAPPING = CN(new_allowed=True)


def add_densepose_head_cse_config(cfg: CN) -> None:
    """
    Add configuration options for Continuous Surface Embeddings (CSE)
    """
    _C = cfg
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE = CN()
    # Dimensionality D of the embedding space
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE = 16
    # Embedder specifications for various mesh IDs
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS = CN(new_allowed=True)


def add_densepose_head_config(cfg: CN) -> None:
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.DENSEPOSE_ON = True

    _C.MODEL.ROI_DENSEPOSE_HEAD = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.NAME = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS = 8
    # Number of parts used for point labels
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES = 24
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL = 4
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM = 512
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL = 3
    _C.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE = 112
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION = 28
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS = 2  # 15 or 2

    # For Decoder # need ROi decoder for color -yy
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON = True # show color
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE = 4

    _C.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR_NAME = "DensePoseEmbeddingPredictor" #"DensePoseChartWithConfidencePredictor"
   

    add_densepose_head_cse_config(cfg)


def add_densepose_config(cfg: CN) -> None:
    add_densepose_head_config(cfg)
    add_dataset_category_config(cfg)

