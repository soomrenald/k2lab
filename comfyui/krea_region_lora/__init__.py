from .nodes import (
    K2BBoxToRegionalMask,
    K2RegionalAttentionLoRASampler,
    K2RegionalCharacterLoRA,
    K2RegionalDecodeComposite,
    K2RegionalLayerLoRAApply,
    K2RegionalLoRAStack3,
)
from .full_nodes import (
    K2ApplyLoRAToRegions,
    K2ApplyProjector,
    K2CompileRegionalPrompts,
    K2FaceDetailer,
    K2LoRAReferenceNode,
    K2PostUpscaler,
    K2ProjectorControls,
    K2PromptEmphasisText,
    K2RegionEditor,
    K2RegionFromMask,
    K2RegionLabApp,
    K2RegionLabSampler,
    K2RegionalControls,
    K2RegionStack,
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "K2BBoxToRegionalMask": K2BBoxToRegionalMask,
    "K2RegionalCharacterLoRA": K2RegionalCharacterLoRA,
    "K2RegionalLoRAStack3": K2RegionalLoRAStack3,
    "K2RegionalLayerLoRAApply": K2RegionalLayerLoRAApply,
    "K2RegionalAttentionLoRASampler": K2RegionalAttentionLoRASampler,
    "K2RegionalDecodeComposite": K2RegionalDecodeComposite,
    "K2RegionFromMask": K2RegionFromMask,
    "K2RegionStack": K2RegionStack,
    "K2RegionEditor": K2RegionEditor,
    "K2CompileRegionalPrompts": K2CompileRegionalPrompts,
    "K2PromptEmphasisText": K2PromptEmphasisText,
    "K2RegionalControls": K2RegionalControls,
    "K2ProjectorControls": K2ProjectorControls,
    "K2ApplyProjector": K2ApplyProjector,
    "K2LoRAReference": K2LoRAReferenceNode,
    "K2ApplyLoRAToRegions": K2ApplyLoRAToRegions,
    "K2RegionLabSampler": K2RegionLabSampler,
    "K2FaceDetailer": K2FaceDetailer,
    "K2PostUpscaler": K2PostUpscaler,
    "K2RegionLabApp": K2RegionLabApp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K2BBoxToRegionalMask": "K2 BBox To Regional Mask",
    "K2RegionalCharacterLoRA": "K2 Regional Character LoRA",
    "K2RegionalLoRAStack3": "K2 Regional LoRA Stack 3",
    "K2RegionalLayerLoRAApply": "K2 Regional Layer LoRA Apply",
    "K2RegionalAttentionLoRASampler": "K2 Regional Attention LoRA Sampler",
    "K2RegionalDecodeComposite": "K2 Regional Decode Composite",
    "K2RegionFromMask": "K2 Region From Mask",
    "K2RegionStack": "K2 Region Stack",
    "K2RegionEditor": "K2 Region Editor",
    "K2CompileRegionalPrompts": "K2 Compile Regional Prompts",
    "K2PromptEmphasisText": "K2 Prompt Emphasis",
    "K2RegionalControls": "K2 Regional Controls",
    "K2ProjectorControls": "K2 Projector Controls",
    "K2ApplyProjector": "K2 Apply Projector",
    "K2LoRAReference": "K2 LoRA Reference",
    "K2ApplyLoRAToRegions": "K2 Apply LoRA To Regions",
    "K2RegionLabSampler": "K2 Region Lab Sampler",
    "K2FaceDetailer": "K2 Face Detailer",
    "K2PostUpscaler": "K2 Post Upscaler",
    "K2RegionLabApp": "K2 Region Lab App",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
