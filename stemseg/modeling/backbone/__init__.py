from .backbone import build_resnet_fpn_backbone

from stemseg.utils.global_registry import GlobalRegistry

BACKBONE_REGISTRY = GlobalRegistry.get("Backbone")

BACKBONE_REGISTRY.add("R-50-FPN", build_resnet_fpn_backbone)
BACKBONE_REGISTRY.add("R-101-FPN", build_resnet_fpn_backbone)
BACKBONE_REGISTRY.add("X-101-FPN", build_resnet_fpn_backbone)

__all__ = [
    BACKBONE_REGISTRY,
    "build_resnet_fpn_backbone"
]
