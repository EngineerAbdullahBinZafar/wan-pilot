"""
Hardware, VRAM, and Attention Preflight Diagnostics Engine for Wan-Video.
Inspects CUDA/Metal/CPU environments, evaluates attention backends,
detects catastrophic OOM traps, and provides prescriptive flag recommendations.
"""

import os
import sys
import platform
import logging
from typing import Dict, Any, List, Optional

from .memory_estimator import estimate_memory, parse_resolution

logger = logging.getLogger(__name__)


def detect_attention_backends() -> Dict[str, Any]:
    """
    Detects available attention backends: FlashAttention-2, SageAttention,
    PyTorch Native SDPA, and xFormers. Handles Windows MSVC absence gracefully.
    """
    has_flash = False
    has_sage = False
    has_sdpa = False
    has_xformers = False
    flash_err = None
    sage_err = None

    # 1. FlashAttention-2
    try:
        import flash_attn  # noqa: F401
        has_flash = True
    except Exception as e:
        flash_err = str(e)
        has_flash = False

    # 2. SageAttention
    try:
        import sageattention  # noqa: F401
        has_sage = True
    except Exception as e:
        sage_err = str(e)
        has_sage = False

    # 3. PyTorch Native SDPA
    try:
        import torch.nn.functional as F
        has_sdpa = hasattr(F, "scaled_dot_product_attention")
    except Exception:
        has_sdpa = False

    # 4. xFormers
    try:
        import xformers.ops  # noqa: F401
        has_xformers = True
    except Exception:
        has_xformers = False

    # Assess active backend
    available: List[str] = []
    if has_flash:
        available.append("flash_attention_2")
    if has_sage:
        available.append("sage_attention")
    if has_sdpa:
        available.append("sdpa")
    if has_xformers:
        available.append("xformers")

    is_windows = sys.platform == "win32"
    windows_msvc_missing = is_windows and not has_flash and not has_sage

    if has_flash:
        active = "flash_attention_2"
        rec = "FlashAttention-2 active. Highest throughput and optimal memory tiling."
    elif has_sage:
        active = "sage_attention"
        rec = "SageAttention active. High-efficiency quantization attention kernels."
    elif has_sdpa:
        active = "sdpa"
        if windows_msvc_missing:
            rec = "PyTorch native SDPA active. Operates with zero compilation overhead at ~95% FlashAttention-2 speed (Windows MSVC toolchain not required)."
        else:
            rec = "PyTorch native SDPA active. Built-in kernel with optimal cross-platform stability."
    elif has_xformers:
        active = "xformers"
        rec = "xFormers memory-efficient attention active."
    else:
        active = "naive"
        rec = "WARNING: No accelerated attention backend found! Naive attention has quadratic memory scaling and will OOM on high resolutions."

    return {
        "active": active,
        "available": available,
        "flash_attention_2": has_flash,
        "sage_attention": has_sage,
        "sdpa": has_sdpa,
        "xformers": has_xformers,
        "windows_msvc_missing": windows_msvc_missing,
        "recommendation": rec,
        "diagnostics": {
            "flash_attn_error": flash_err,
            "sage_attn_error": sage_err,
        },
    }


def inspect_hardware_environment(device_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspects host OS, Python, PyTorch runtime, CUDA GPUs, and Apple Silicon MPS.
    """
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    python_ver = sys.version.split()[0]
    torch_installed = False
    torch_ver = "not installed"
    cuda_available = False
    mps_available = False
    device_name = "CPU"
    compute_capability = "N/A"
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    allocated_vram_gb = 0.0
    device_count = 0
    unified_memory = False

    try:
        import torch
        torch_installed = True
        torch_ver = getattr(torch, "__version__", "unknown")

        # Check CUDA
        if hasattr(torch, "cuda") and torch.cuda.is_available() and device_override != "cpu":
            cuda_available = True
            device_count = torch.cuda.device_count()
            target_device = 0
            if device_override and device_override.startswith("cuda:"):
                try:
                    target_device = int(device_override.split(":")[1])
                except Exception:
                    target_device = 0

            device_name = torch.cuda.get_device_name(target_device)
            try:
                props = torch.cuda.get_device_properties(target_device)
                compute_capability = f"{props.major}.{props.minor}"
                total_bytes = props.total_memory
            except Exception:
                total_bytes = 0

            try:
                free_bytes, total_b = torch.cuda.mem_get_info(target_device)
                if total_bytes == 0:
                    total_bytes = total_b
                free_vram_gb = round(free_bytes / (1024 ** 3), 2)
            except Exception:
                free_vram_gb = round(total_bytes * 0.9 / (1024 ** 3), 2)

            total_vram_gb = round(total_bytes / (1024 ** 3), 2)
            try:
                allocated_bytes = torch.cuda.memory_allocated(target_device)
                allocated_vram_gb = round(allocated_bytes / (1024 ** 3), 2)
            except Exception:
                allocated_vram_gb = 0.0

        # Check MPS (Apple Silicon)
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and device_override != "cpu":
            mps_available = True
            device_name = "Apple Silicon (MPS)"
            unified_memory = True
            # Query system memory via psutil as unified VRAM
            try:
                import psutil
                vm = psutil.virtual_memory()
                total_vram_gb = round(vm.total / (1024 ** 3), 2)
                free_vram_gb = round(vm.available / (1024 ** 3), 2)
            except Exception:
                total_vram_gb = 16.0
                free_vram_gb = 12.0

        else:
            # CPU fallback: query host RAM
            try:
                import psutil
                vm = psutil.virtual_memory()
                total_vram_gb = round(vm.total / (1024 ** 3), 2)
                free_vram_gb = round(vm.available / (1024 ** 3), 2)
            except Exception:
                total_vram_gb = 16.0
                free_vram_gb = 8.0

    except ImportError:
        pass

    return {
        "os": os_name,
        "python_version": python_ver,
        "torch_installed": torch_installed,
        "torch_version": torch_ver,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device_count": device_count,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb,
        "allocated_vram_gb": allocated_vram_gb,
        "unified_memory": unified_memory,
    }


def diagnose_hardware_preflight(
    model_variant: str = "wan2.2_moe_14b",
    resolution: str = "720p",
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive preflight diagnostics for Wan-Video execution.
    Inspects hardware, assesses attention engines, computes exact mathematical memory headroom,
    detects critical OOM traps, and synthesizes recommended mitigation flags.

    Args:
        model_variant: 'wan2.1_1.3b', 'wan2.1_14b', or 'wan2.2_moe_14b'.
        resolution: '480p', '720p', '1080p', or custom 'WxH'.
        device_override: Optional device string ('cuda:0', 'mps', 'cpu').

    Returns:
        Structured preflight report dictionary.
    """
    hardware = inspect_hardware_environment(device_override)
    attention = detect_attention_backends()

    total_vram = hardware["total_vram_gb"]
    free_vram = hardware["free_vram_gb"]
    is_moe = "moe" in model_variant.lower()
    is_14b = "14b" in model_variant.lower()
    w, h = parse_resolution(resolution)
    is_high_res = (w * h) >= (1280 * 720)

    # Base recommendation flags
    recommended_flags: List[str] = []
    flagged_traps: List[Dict[str, str]] = []

    # 1. Evaluate Traps
    # Trap 1: OOM_TRAP_MOE_CONCURRENT
    if is_moe and free_vram < 32.0:
        flagged_traps.append({
            "trap_id": "OOM_TRAP_MOE_CONCURRENT",
            "severity": "CRITICAL",
            "description": (
                "Wan2.2 MoE contains 27B parameters across high-noise and low-noise experts. "
                "Loading both concurrently requires >32GB in fp8 or >54GB in bf16, exceeding available VRAM."
            ),
            "mitigation": "Enforce sequential expert swapping via '--expert_offload True'.",
        })

    # Trap 2: OOM_TRAP_T5_VRAM
    if (is_14b or is_moe) and free_vram < 24.0:
        flagged_traps.append({
            "trap_id": "OOM_TRAP_T5_VRAM",
            "severity": "HIGH",
            "description": (
                "UMT5-XXL text encoder requires ~9.4GB VRAM in bf16. Keeping it in GPU memory "
                "during diffusion steps starves DiT activation cache."
            ),
            "mitigation": "Offload text encoder to system RAM via '--t5_cpu True'.",
        })

    # Trap 3: OOM_TRAP_DIT_FP32
    if is_14b or is_moe:
        flagged_traps.append({
            "trap_id": "OOM_TRAP_DIT_FP32",
            "severity": "HIGH",
            "description": (
                "Instantiating DiT layers with default float32 precision allocates ~56.8GB VRAM "
                "for a 14B parameter model, immediately triggering CUDA OOM."
            ),
            "mitigation": "Enforce '--precision fp8' or wrap layer construction in dit_dtype_guard(bfloat16).",
        })

    # Trap 4: OOM_TRAP_VAE_DEC_SPIKE
    if is_high_res:
        flagged_traps.append({
            "trap_id": "OOM_TRAP_VAE_DEC_SPIKE",
            "severity": "MEDIUM",
            "description": (
                f"Untiled 3D causal VAE deconvolution spikes up to ~8.7GB at {resolution} resolution, "
                "frequently causing OOM crashes at 99% generation completion."
            ),
            "mitigation": "Enable bounded tiled VAE decoding via '--tiled_vae True'.",
        })

    # Determine recommended configuration flags based on hardware budget
    if total_vram <= 8.0:
        recommended_flags = ["--precision", "fp8", "--offload_model", "True", "--t5_cpu", "True", "--tiled_vae", "True"]
        precision_choice = "fp8"
        moe_offload_choice = True
        t5_cpu_choice = True
        tiled_vae_choice = True
    elif total_vram <= 12.5:
        if is_14b or is_moe:
            recommended_flags = ["--precision", "fp8", "--offload_model", "True", "--t5_cpu", "True", "--tiled_vae", "True"]
            precision_choice = "fp8"
            moe_offload_choice = True
            t5_cpu_choice = True
            tiled_vae_choice = True
        else:
            recommended_flags = ["--precision", "bf16", "--t5_cpu", "True", "--tiled_vae", "True"]
            precision_choice = "bf16"
            moe_offload_choice = True
            t5_cpu_choice = True
            tiled_vae_choice = True
    elif total_vram <= 24.5:
        # 24GB GPUs (RTX 3090, RTX 4090)
        recommended_flags = ["--precision", "fp8", "--expert_offload", "True", "--t5_cpu", "True", "--tiled_vae", "True"]
        precision_choice = "fp8"
        moe_offload_choice = True
        t5_cpu_choice = True
        tiled_vae_choice = True
    else:
        # High VRAM (40GB/80GB Datacenter GPUs)
        recommended_flags = ["--precision", "bf16", "--expert_offload", "True", "--tiled_vae", "True"]
        precision_choice = "bf16"
        moe_offload_choice = True
        t5_cpu_choice = False
        tiled_vae_choice = True

    # Apple Silicon specific guidance
    mps_guidance = None
    if hardware["mps_available"]:
        mps_guidance = {
            "environment_variable": "PYTORCH_ENABLE_MPS_FALLBACK=1",
            "recommended_sampler": "Euler or DPM++ 2M",
            "note": "MPS uses unified memory architecture. Ensure swap space is available for 14B offloading.",
        }

    # Calculate analytical memory requirements with recommended settings
    mem_est = estimate_memory(
        model_variant=model_variant,
        resolution=resolution,
        num_frames=81,
        precision=precision_choice,
        moe_offload=moe_offload_choice,
        t5_cpu=t5_cpu_choice,
        tiled_vae=tiled_vae_choice,
    )

    estimated_peak = mem_est.peak_vram_gb
    headroom = round(free_vram - estimated_peak, 2)
    can_run = headroom >= 0.0 or hardware["unified_memory"] or total_vram >= estimated_peak

    # Status classification
    if headroom >= 2.0 or (hardware["unified_memory"] and free_vram >= 8.0):
        system_status = "READY"
    elif headroom >= 0.0:
        system_status = "WARNING_LOW_HEADROOM"
    else:
        system_status = "INSUFFICIENT_VRAM"

    return {
        "system_status": system_status,
        "hardware": hardware,
        "attention_backend": attention,
        "evaluation": {
            "model_variant": model_variant,
            "resolution": resolution,
            "estimated_peak_vram_gb": estimated_peak,
            "available_vram_gb": free_vram,
            "headroom_gb": headroom,
            "can_run": can_run,
        },
        "flagged_traps": flagged_traps,
        "recommended_flags": recommended_flags,
        "mps_guidance": mps_guidance,
        "memory_breakdown": mem_est.to_dict(),
    }
