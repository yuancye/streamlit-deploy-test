from detectron2.layers import ShapeSpec

from .anchor_generator import build_anchor_generator, ANCHOR_GENERATOR_REGISTRY
from .backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_backbone,
    build_resnet_backbone,
    make_stage,
)
from .meta_arch import (
    META_ARCH_REGISTRY,
    GeneralizedRCNN,
    build_model,
)
from .postprocessing import detector_postprocess
from .proposal_generator import (
    PROPOSAL_GENERATOR_REGISTRY,
    build_proposal_generator,
    RPN_HEAD_REGISTRY,
    build_rpn_head,
)
from .roi_heads import (
    ROI_BOX_HEAD_REGISTRY,
    ROI_HEADS_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    FastRCNNOutputLayers,
    build_box_head,
    build_roi_heads,
)

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
