# Copyright (c) Facebook, Inc. and its affiliates.


from .inference import densepose_inference
from .utils import initialize_module_params
from .build import (
    build_densepose_embedder,
    build_densepose_head,
    build_densepose_predictor,
)
