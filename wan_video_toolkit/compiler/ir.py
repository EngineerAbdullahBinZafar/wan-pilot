"""
Canonical Generation Pipeline IR (Intermediate Representation) for Wan-Video.
Specifies device-agnostic, frontend-agnostic execution graphs across Wan2.1 and Wan2.2 MoE models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class TargetFormat(str, Enum):
    CLI_BASH = "cli_bash"
    CLI_POWERSHELL = "cli_powershell"
    COMFYUI_PROMPT = "comfyui_prompt"
    COMFYUI_CANVAS = "comfyui_canvas"
    DIFFUSERS = "diffusers"


RESOLUTION_MAP = {
    "480p": (832, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


class RecommendedFlags(list):
    """List that matches both exact items and flag prefixes (e.g. '--precision' matches '--precision fp8')."""
    def __contains__(self, item: Any) -> bool:
        if super().__contains__(item):
            return True
        if isinstance(item, str):
            for existing in self:
                if isinstance(existing, str) and (existing.startswith(item) or item in existing):
                    return True
        return False


@dataclass
class WanGenerationPipelineIR:
    prompt: str
    negative_prompt: str = ""
    model: str = "wan2.2_moe_14b"
    resolution: str = "720p"
    width: int = 1280
    height: int = 720
    num_frames: int = 81
    fps: int = 16
    steps: int = 30
    guidance: float = 6.0
    seed: int = -1
    input_image: Optional[str] = None
    image_path: Optional[str] = None
    task: str = "t2v"
    offload_profile: str = "balanced"
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt or not str(self.prompt).strip():
            raise ValueError("Prompt cannot be empty")
        if self.steps <= 0:
            raise ValueError("Steps must be positive")
        if self.guidance < 0:
            raise ValueError("Guidance must be non-negative")

        # Sync image_path and input_image
        if self.image_path and not self.input_image:
            self.input_image = self.image_path
        elif self.input_image and not self.image_path:
            self.image_path = self.input_image

        if self.input_image:
            self.task = "i2v"

        # Apply resolution presets if dimensions are default
        if self.resolution in RESOLUTION_MAP:
            preset_w, preset_h = RESOLUTION_MAP[self.resolution]
            # Only apply if width/height are at standard 720p defaults and resolution is different
            if self.resolution != "720p" and self.width == 1280 and self.height == 720:
                self.width = preset_w
                self.height = preset_h

    def is_temporal_frame_valid(self) -> bool:
        """Wan temporal VAE requires F = 4k + 1 (e.g. 17, 33, 49, 65, 81)."""
        return (self.num_frames - 1) % 4 == 0

    def are_dimensions_valid(self) -> bool:
        """Wan spatial patchification requires width and height divisible by 16."""
        return self.width % 16 == 0 and self.height % 16 == 0

    def get_recommended_flags(self) -> RecommendedFlags:
        """Translates offload profile into CLI flags."""
        if self.offload_profile in ("balanced", "low_vram"):
            return RecommendedFlags([
                "--precision fp8",
                "--expert_offload True",
                "--t5_cpu True",
                "--tiled_vae True",
            ])
        elif self.offload_profile == "aggressive_cpu_offload":
            return RecommendedFlags([
                "--precision fp8",
                "--expert_offload True",
                "--t5_cpu True",
                "--tiled_vae True",
                "--dit_cpu_offload True",
            ])
        else:
            return RecommendedFlags(["--precision bf16", "--expert_offload False"])

    def to_dict(self) -> Dict[str, Any]:
        """Serializes IR into a clean JSON-friendly dictionary."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model": self.model,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "steps": self.steps,
            "guidance": self.guidance,
            "seed": self.seed,
            "input_image": self.input_image,
            "image_path": self.image_path,
            "task": self.task,
            "offload_profile": self.offload_profile,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WanGenerationPipelineIR":
        return cls(**data)
