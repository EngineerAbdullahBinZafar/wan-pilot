"""
Analytical VRAM and System Memory Estimation Engine for Wan-Video models.
Supports Wan2.1 (1.3B, 14B) and Wan2.2 MoE dual-expert architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


# Architecture constant parameters
WAN_MODEL_SPECS = {
    "wan2.1_1.3b": {
        "dit_params": 1_332_602_880,
        "total_params": 1_332_602_880,
        "active_params": 1_332_602_880,
        "hidden_dim": 1536,
        "layers": 30,
        "heads": 12,
        "head_dim": 128,
        "is_moe": False,
    },
    "wan2.1_14b": {
        "dit_params": 14_249_779_200,
        "total_params": 14_249_779_200,
        "active_params": 14_249_779_200,
        "hidden_dim": 5120,
        "layers": 40,
        "heads": 40,
        "head_dim": 128,
        "is_moe": False,
    },
    "wan2.2_moe_14b": {
        "dit_params": 14_249_779_200,  # Active expert
        "total_params": 27_000_000_000,  # Combined dual-expert repository (~27B)
        "active_params": 14_249_779_200,
        "hidden_dim": 5120,
        "layers": 40,
        "heads": 40,
        "head_dim": 128,
        "is_moe": True,
    },
}

PRECISION_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.55,
}

RESOLUTION_PRESETS = {
    "480p": (832, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

UMT5_PARAMS = 4_700_000_000  # ~4.7B params


@dataclass
class MemoryEstimateResult:
    model_variant: str
    resolution: str
    width: int
    height: int
    num_frames: int
    precision: str
    bytes_per_param: float
    active_dit_params: int
    total_dit_params: int
    active_dit_weights_gb: float
    dormant_expert_weights_gb: float
    umt5_weights_vram_gb: float
    umt5_weights_ram_gb: float
    latent_tokens: int
    activation_and_kv_cache_gb: float
    vae_peak_gb: float
    cuda_overhead_gb: float
    peak_vram_gb: float
    peak_vram_without_moe_offload_gb: float
    system_ram_gb: float
    fits_24gb_gpu: bool
    fits_16gb_gpu: bool
    fits_12gb_gpu: bool
    oom_risk_on_24gb: str
    memory_breakdown_gb: Dict[str, float] = field(default_factory=dict)
    totals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration": {
                "model_variant": self.model_variant,
                "resolution": f"{self.width}x{self.height}",
                "num_frames": self.num_frames,
                "precision": self.precision,
                "bytes_per_param": self.bytes_per_param,
                "active_params": self.active_dit_params,
                "total_params": self.total_dit_params,
            },
            "memory_breakdown_gb": self.memory_breakdown_gb,
            "totals": self.totals,
            "concurrent_comparison": {
                "peak_vram_without_moe_offload_gb": self.peak_vram_without_moe_offload_gb,
                "oom_risk_on_24gb": self.oom_risk_on_24gb,
            },
        }


def parse_resolution(resolution: str) -> Tuple[int, int]:
    """Resolves standard preset names ('720p') or dimension strings ('1280x720', '1280*720')."""
    res_lower = resolution.lower().strip()
    if res_lower in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[res_lower]
    
    for sep in ["x", "*"]:
        if sep in res_lower:
            parts = res_lower.split(sep)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])
                
    # Fallback default
    return RESOLUTION_PRESETS["720p"]


def estimate_memory(
    model_variant: str = "wan2.2_moe_14b",
    resolution: str = "720p",
    num_frames: int = 81,
    precision: str = "fp8",
    moe_offload: bool = True,
    t5_cpu: bool = True,
    tiled_vae: bool = True,
) -> MemoryEstimateResult:
    """
    Computes exact analytical memory requirements across weights, context tokens,
    activation caches, and 3D VAE decoding for Wan-Video models.

    Args:
        model_variant: 'wan2.1_1.3b', 'wan2.1_14b', or 'wan2.2_moe_14b'.
        resolution: '480p', '720p', '1080p' or custom 'WxH'.
        num_frames: Sequence frame count (standard is 81 frames, 4k+1).
        precision: 'fp32', 'bf16', 'fp16', 'fp8', 'int8', or 'int4'.
        moe_offload: If True, uses sequential expert swapping for Wan2.2 MoE.
        t5_cpu: If True, offloads UMT5-XXL text encoder to system RAM.
        tiled_vae: If True, uses bounded tiled VAE decoding (~1.8GB peak).
    """
    model_key = model_variant.lower().strip()
    if model_key not in WAN_MODEL_SPECS:
        raise ValueError(f"Unknown model variant: {model_variant}. Supported: {list(WAN_MODEL_SPECS.keys())}")

    spec = WAN_MODEL_SPECS[model_key]
    width, height = parse_resolution(resolution)
    bp = PRECISION_BYTES.get(precision.lower(), 1.0)

    # 1. DiT Weight Memory
    active_params = spec["active_params"]
    total_params = spec["total_params"]
    is_moe = spec["is_moe"]

    # Active weights on GPU
    active_dit_weights_gb = round((active_params * bp) / (1024 ** 3), 2)

    # In MoE without sequential offload, both experts are pinned in VRAM concurrently
    if is_moe and not moe_offload:
        dormant_expert_params = total_params - active_params
        dormant_expert_weights_gb = round((dormant_expert_params * bp) / (1024 ** 3), 2)
    else:
        dormant_expert_weights_gb = 0.0

    # 2. UMT5-XXL Text Encoder Memory
    # UMT5 is ~4.7B params, standardly resident in bf16 (2.0 bytes/param) ~9.4 GB
    t5_bp = 2.0 if precision in ["fp32", "bf16", "fp16"] else 1.0
    umt5_size_gb = round((UMT5_PARAMS * 2.0) / (1024 ** 3), 2)
    if t5_cpu:
        umt5_weights_vram_gb = 0.0
        umt5_weights_ram_gb = umt5_size_gb
    else:
        umt5_weights_vram_gb = round((UMT5_PARAMS * t5_bp) / (1024 ** 3), 2)
        umt5_weights_ram_gb = 0.0

    # 3. Latent Tokens and Activation Cache
    # Temporal: 4x causal compression -> F_lat = 1 + (F - 1) // 4
    # Spatial: 8x spatial downscale and 2x patchification -> 16x downscale
    f_lat = 1 + max(0, (num_frames - 1)) // 4
    tokens_spatial = (width // 16) * (height // 16)
    latent_tokens = f_lat * tokens_spatial

    # Activation memory: kappa * L_seq * D * bp / (1024^3)
    kappa = 4.8  # activation factor accounting for SwiGLU FFN expansion and norms
    hidden_dim = spec["hidden_dim"]
    activation_gb = round((kappa * latent_tokens * hidden_dim * bp) / (1024 ** 3), 2)

    # 4. 3D VAE Peak Memory
    if tiled_vae:
        vae_peak_gb = 1.8
    else:
        # Untiled 3D causal VAE deconvolution memory spike (intermediate activations spike to ~8.7GB at 720p)
        vae_peak_gb = round((125.0 * num_frames * width * height) / (1024 ** 3), 2)

    # 5. Allocator & CUDA Context Overhead
    weights_on_gpu = active_dit_weights_gb + dormant_expert_weights_gb + umt5_weights_vram_gb
    cuda_overhead_gb = round(max(0.8, 0.15 * (weights_on_gpu + activation_gb) + 0.65), 2)

    # Total Peak VRAM with configured settings
    peak_vram_gb = round(weights_on_gpu + activation_gb + vae_peak_gb + cuda_overhead_gb, 2)

    # Concurrent MoE comparison (what happens if moe_offload=False and t5_cpu=False)
    if is_moe:
        concurrent_weights_gb = round((total_params * bp) / (1024 ** 3), 2) + (umt5_weights_vram_gb if not t5_cpu else round((UMT5_PARAMS * t5_bp) / (1024 ** 3), 2))
        concurrent_overhead = round(max(0.8, 0.15 * (concurrent_weights_gb + activation_gb) + 0.65), 2)
        peak_vram_without_moe_offload_gb = round(
            concurrent_weights_gb + activation_gb + vae_peak_gb + concurrent_overhead, 2
        )
    else:
        peak_vram_without_moe_offload_gb = peak_vram_gb

    # System RAM consumption
    system_ram_gb = round(umt5_weights_ram_gb + (active_dit_weights_gb if moe_offload and is_moe else 0.0) + 4.8, 2)

    # Feasibility evaluation
    fits_24gb_gpu = peak_vram_gb <= 23.5
    fits_16gb_gpu = peak_vram_gb <= 15.5
    fits_12gb_gpu = peak_vram_gb <= 11.5

    if is_moe and not moe_offload and peak_vram_without_moe_offload_gb > 24.0:
        oom_risk_on_24gb = "100% Guaranteed OOM without sequential offload"
    elif peak_vram_gb > 24.0:
        oom_risk_on_24gb = f"Exceeds 24GB capacity ({peak_vram_gb} GB required)"
    else:
        oom_risk_on_24gb = "Safe on 24GB GPUs (RTX 3090/4090)"

    breakdown = {
        "active_dit_weights": active_dit_weights_gb,
        "dormant_expert_weights_in_vram": dormant_expert_weights_gb,
        "umt5_text_encoder_vram": umt5_weights_vram_gb,
        "umt5_text_encoder_ram": umt5_weights_ram_gb,
        "activation_and_kv_cache": activation_gb,
        "tiled_vae_peak": vae_peak_gb,
        "cuda_context_overhead": cuda_overhead_gb,
    }

    totals = {
        "peak_vram_gb": peak_vram_gb,
        "system_ram_gb": system_ram_gb,
        "fits_24gb_gpu": fits_24gb_gpu,
        "fits_16gb_gpu": fits_16gb_gpu,
        "fits_12gb_gpu": fits_12gb_gpu,
    }

    return MemoryEstimateResult(
        model_variant=model_variant,
        resolution=f"{width}x{height}",
        width=width,
        height=height,
        num_frames=num_frames,
        precision=precision,
        bytes_per_param=bp,
        active_dit_params=active_params,
        total_dit_params=total_params,
        active_dit_weights_gb=active_dit_weights_gb,
        dormant_expert_weights_gb=dormant_expert_weights_gb,
        umt5_weights_vram_gb=umt5_weights_vram_gb,
        umt5_weights_ram_gb=umt5_weights_ram_gb,
        latent_tokens=latent_tokens,
        activation_and_kv_cache_gb=activation_gb,
        vae_peak_gb=vae_peak_gb,
        cuda_overhead_gb=cuda_overhead_gb,
        peak_vram_gb=peak_vram_gb,
        peak_vram_without_moe_offload_gb=peak_vram_without_moe_offload_gb,
        system_ram_gb=system_ram_gb,
        fits_24gb_gpu=fits_24gb_gpu,
        fits_16gb_gpu=fits_16gb_gpu,
        fits_12gb_gpu=fits_12gb_gpu,
        oom_risk_on_24gb=oom_risk_on_24gb,
        memory_breakdown_gb=breakdown,
        totals=totals,
    )
