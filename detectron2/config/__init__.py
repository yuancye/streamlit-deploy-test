from .compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable

__all__ = [
    "CfgNode",
    "get_cfg",
    "configurable",
]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
