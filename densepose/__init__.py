# Copyright (c) Facebook, Inc. and its affiliates.

from .config import (
    add_densepose_config,
    add_densepose_head_config,
    add_dataset_category_config
)

from .modeling.roi_heads import DensePoseROIHeads


