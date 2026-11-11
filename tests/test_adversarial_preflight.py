"""
Adversarial Stress Test Suite for Wan-Video Preflight Diagnostics & Memory Estimator.
Authored by: wan_m1_challenger_2 (Empirical Challenger)
"""

import sys
import types
import os
import numpy as np
from PIL import Image
import pytest
from unittest.mock import MagicMock, patch

from wan_video_toolkit.diagnostics.preflight import (
    detect_attention_backends,
    inspect_hardware_environment,
    diagnose_hardware_preflight,
)
from wan_video_toolkit.diagnostics.memory_estimator import (
    estimate_memory,
    parse_resolution,
    MemoryEstimateResult,
    WAN_MODEL_SPECS,
    PRECISION_BYTES,
)


# ============================================================================
# Section 1: Preflight & Attention Fallbacks Stress Tests
# ============================================================================

def test_detect_attention_backends_no_torch(monkeypatch):
    """Verify detect_attention_backends cleanly handles complete absence of torch."""
    orig_import = __import__
    def fake_import(name, *args, **kwargs):
        if name.startswith("torch"):
            raise ImportError(f"No module named '{name}' (simulated missing torch)")
        if name in ("flash_attn", "sageattention", "xformers"):
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = detect_attention_backends()
    assert isinstance(result, dict)
    assert result["active"] == "naive"
    assert result["flash_attention_2"] is False
    assert result["sage_attention"] is False
    assert result["sdpa"] is False
    assert result["xformers"] is False
    assert "WARNING" in result["recommendation"]
    assert len(result["available"]) == 0


def test_detect_attention_backends_torch_no_cuda(monkeypatch):
    """Verify fallback to SDPA when torch is present but CUDA is absent."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Block flash_attn and sageattention
    orig_import = __import__
    def fake_import(name, *args, **kwargs):
        if name in ("flash_attn", "sageattention", "xformers"):
            raise ImportError(f"No module named '{name}'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = detect_attention_backends()
    assert result["sdpa"] is True
    assert result["active"] == "sdpa"


def test_detect_attention_backends_windows_no_msvc(monkeypatch):
    """Simulate Windows host without MSVC compiler tools."""
    monkeypatch.setattr(sys, "platform", "win32")
    orig_import = __import__
    def fake_import(name, *args, **kwargs):
        if name in ("flash_attn", "sageattention"):
            raise ImportError(f"MSVC build failed for {name}")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = detect_attention_backends()
    assert result["windows_msvc_missing"] is True
    assert result["active"] == "sdpa"
    assert "Windows MSVC toolchain not required" in result["recommendation"]


def test_detect_attention_backends_macos_mps(monkeypatch):
    """Simulate macOS environment with Metal Performance Shaders."""
    monkeypatch.setattr(sys, "platform", "darwin")
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    orig_import = __import__
    def fake_import(name, *args, **kwargs):
        if name in ("flash_attn", "sageattention"):
            raise ImportError(f"{name} unsupported on darwin")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = detect_attention_backends()
    assert result["active"] == "sdpa"
    assert result["windows_msvc_missing"] is False


def test_detect_attention_never_raises_on_arbitrary_exceptions(monkeypatch):
    """Ensure detect_attention_backends never raises even if imports raise unexpected exceptions."""
    orig_import = __import__
    def buggy_import(name, *args, **kwargs):
        if name in ("flash_attn", "sageattention", "torch.nn.functional", "xformers.ops"):
            raise RuntimeError("Corrupted DLL or segfault during import")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", buggy_import)
    result = detect_attention_backends()
    assert isinstance(result, dict)
    assert result["active"] in ("sdpa", "naive")


# ============================================================================
# Section 2: Combinatorial Matrix of GPU Sizes & Model Variants
# ============================================================================

GPU_CONFIGS = [
    # (total_gb, free_gb, name, sm)
    (8.0, 7.5, "NVIDIA GeForce RTX 4050", "8.9"),
    (12.0, 11.0, "NVIDIA GeForce RTX 4070", "8.9"),
    (16.0, 15.0, "NVIDIA GeForce RTX 4080", "8.9"),
    (24.0, 22.4, "NVIDIA GeForce RTX 4090", "8.9"),
    (40.0, 38.0, "NVIDIA A100-PCIE-40GB", "8.0"),
    (80.0, 78.0, "NVIDIA A100-SXM4-80GB", "8.0"),
]

MODEL_VARIANTS = ["wan2.1_1.3b", "wan2.1_14b", "wan2.2_moe_14b"]
RESOLUTIONS = ["480p", "720p", "1080p"]


@pytest.mark.parametrize("total_gb,free_gb,name,sm", GPU_CONFIGS)
@pytest.mark.parametrize("model", MODEL_VARIANTS)
def test_preflight_combinatorial_gpu_matrix(monkeypatch, total_gb, free_gb, name, sm, model):
    """
    Stress-tests diagnose_hardware_preflight across 6 GPU tiers x 3 model architectures.
    Validates trap detection invariants, headroom calculation, and recommended flags.
    """
    import torch

    class FakeProps:
        total_memory = int(total_gb * 1024**3)
        major = int(sm.split(".")[0])
        minor = int(sm.split(".")[1])
        multi_processor_count = 64

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: name)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: FakeProps())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
        int(free_gb * 1024**3),
        int(total_gb * 1024**3),
    ))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(0.5 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    report = diagnose_hardware_preflight(model_variant=model, resolution="720p")
    assert report["hardware"]["total_vram_gb"] == pytest.approx(total_gb, rel=0.01)
    assert report["hardware"]["free_vram_gb"] == pytest.approx(free_gb, rel=0.01)

    traps = {t["trap_id"]: t for t in report["flagged_traps"]}

    # Invariant 1: OOM_TRAP_MOE_CONCURRENT
    if model == "wan2.2_moe_14b":
        if free_gb < 32.0:
            assert "OOM_TRAP_MOE_CONCURRENT" in traps, f"Failed to flag MoE trap on {total_gb}GB GPU ({free_gb}GB free)"
            assert traps["OOM_TRAP_MOE_CONCURRENT"]["severity"] == "CRITICAL"
        else:
            assert "OOM_TRAP_MOE_CONCURRENT" not in traps, f"Falsely flagged MoE trap on {total_gb}GB GPU ({free_gb}GB free)"
    else:
        assert "OOM_TRAP_MOE_CONCURRENT" not in traps

    # Invariant 2: OOM_TRAP_T5_VRAM
    if model in ("wan2.1_14b", "wan2.2_moe_14b"):
        if free_gb < 24.0:
            assert "OOM_TRAP_T5_VRAM" in traps, f"Failed to flag T5 trap on {total_gb}GB GPU"
        else:
            assert "OOM_TRAP_T5_VRAM" not in traps, f"Falsely flagged T5 trap on {total_gb}GB GPU"
    else:
        assert "OOM_TRAP_T5_VRAM" not in traps

    # Invariant 3: OOM_TRAP_DIT_FP32
    if model in ("wan2.1_14b", "wan2.2_moe_14b"):
        assert "OOM_TRAP_DIT_FP32" in traps
    else:
        assert "OOM_TRAP_DIT_FP32" not in traps

    # Invariant 4: Headroom calculation
    est_peak = report["evaluation"]["estimated_peak_vram_gb"]
    expected_headroom = round(free_gb - est_peak, 2)
    assert report["evaluation"]["headroom_gb"] == pytest.approx(expected_headroom, abs=0.05)

    # Invariant 5: Status categorization
    if expected_headroom >= 2.0:
        assert report["system_status"] == "READY"
    elif expected_headroom >= 0.0:
        assert report["system_status"] == "WARNING_LOW_HEADROOM"
    else:
        assert report["system_status"] == "INSUFFICIENT_VRAM"


def test_preflight_vae_dec_spike_resolution_boundary(mock_cuda_rtx4090):
    """Verify OOM_TRAP_VAE_DEC_SPIKE triggers strictly at >= 720p (1280x720)."""
    # 480p (832x480 = 399,360) < (1280*720 = 921,600) -> should NOT trigger
    rep_480p = diagnose_hardware_preflight(model_variant="wan2.1_14b", resolution="480p")
    traps_480p = [t["trap_id"] for t in rep_480p["flagged_traps"]]
    assert "OOM_TRAP_VAE_DEC_SPIKE" not in traps_480p

    # 720p (1280x720) -> should trigger
    rep_720p = diagnose_hardware_preflight(model_variant="wan2.1_14b", resolution="720p")
    traps_720p = [t["trap_id"] for t in rep_720p["flagged_traps"]]
    assert "OOM_TRAP_VAE_DEC_SPIKE" in traps_720p

    # 1080p (1920x1080) -> should trigger
    rep_1080p = diagnose_hardware_preflight(model_variant="wan2.1_14b", resolution="1080p")
    traps_1080p = [t["trap_id"] for t in rep_1080p["flagged_traps"]]
    assert "OOM_TRAP_VAE_DEC_SPIKE" in traps_1080p


# ============================================================================
# Section 3: Memory Estimator Real-World Benchmarks Stress Tests
# ============================================================================

def test_survey_benchmark_1_wan2_1_1_3b_480p():
    """
    Survey 1 Table: Wan2.1 1.3B 480p bf16 ~ 8.4 GB.
    Verify that 1.3B 480p bf16 consumes under 12GB and fits 12GB/16GB/24GB GPUs.
    """
    # With t5_cpu=True
    res_cpu = estimate_memory("wan2.1_1.3b", "480p", num_frames=81, precision="bf16", t5_cpu=True, tiled_vae=True)
    assert res_cpu.peak_vram_gb <= 8.4
    assert res_cpu.fits_12gb_gpu is True
    assert res_cpu.fits_24gb_gpu is True

    # With t5_cpu=False (full resident weights)
    res_gpu = estimate_memory("wan2.1_1.3b", "480p", num_frames=81, precision="bf16", t5_cpu=False, tiled_vae=True)
    # Weights: 2.48GB DiT + 8.75GB T5 = 11.23GB
    assert res_gpu.active_dit_weights_gb == 2.48
    assert res_gpu.umt5_weights_vram_gb == 8.75


def test_survey_benchmark_2_wan2_1_14b_720p_fp8():
    """
    Survey 1 Table: Wan2.1 14B 720p fp8 + t5_cpu + tiled_vae ~ 21.8 GB (fits 24GB RTX 3090/4090).
    """
    res = estimate_memory("wan2.1_14b", "720p", num_frames=81, precision="fp8", t5_cpu=True, tiled_vae=True)
    # Peak VRAM must be <= 23.5 GB (safe on 24GB) and > 18.0 GB
    assert 18.0 <= res.peak_vram_gb <= 23.5
    assert res.fits_24gb_gpu is True
    assert res.fits_16gb_gpu is False
    assert res.active_dit_weights_gb == pytest.approx(13.27, abs=0.5)


def test_survey_benchmark_3_wan2_2_moe_14b_720p_fp8():
    """
    Survey 1 Table: Wan2.2 MoE 14B 720p fp8 + expert_offload + t5_cpu + tiled_vae ~ 18.4 - 22.4 GB (fits 24GB).
    """
    res = estimate_memory("wan2.2_moe_14b", "720p", num_frames=81, precision="fp8", moe_offload=True, t5_cpu=True, tiled_vae=True)
    assert 18.0 <= res.peak_vram_gb <= 22.5
    assert res.fits_24gb_gpu is True
    assert res.fits_16gb_gpu is False
    assert res.oom_risk_on_24gb == "Safe on 24GB GPUs (RTX 3090/4090)"


def test_survey_benchmark_4_wan2_2_moe_14b_720p_bf16_concurrent():
    """
    Survey 1 Table: Wan2.2 MoE 14B 720p bf16 concurrent > 60 GB (crashes 24GB GPU).
    """
    res = estimate_memory("wan2.2_moe_14b", "720p", num_frames=81, precision="bf16", moe_offload=False, t5_cpu=False, tiled_vae=True)
    assert res.peak_vram_without_moe_offload_gb > 60.0
    assert res.fits_24gb_gpu is False
    assert "OOM" in res.oom_risk_on_24gb


def test_negative_headroom_oom_scenario(monkeypatch):
    """Stress test negative headroom on constrained GPU (8GB RTX 4050 attempting 14B without offload)."""
    import torch

    class FakeProps8GB:
        total_memory = int(8.0 * 1024**3)
        major = 8
        minor = 9
        multi_processor_count = 32

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 4050 Laptop")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: FakeProps8GB())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
        int(4.0 * 1024**3),  # Only 4GB free
        int(8.0 * 1024**3),
    ))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(4.0 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    report = diagnose_hardware_preflight(model_variant="wan2.1_14b", resolution="720p")
    # Peak for 14B fp8 + offload is ~19.7 GB (or ~13.4 with layer offload)
    headroom = report["evaluation"]["headroom_gb"]
    assert headroom < 0.0, f"Expected negative headroom for 14B on 4GB free VRAM, got {headroom}"
    assert report["system_status"] == "INSUFFICIENT_VRAM"
    assert report["evaluation"]["can_run"] is False


# ============================================================================
# Section 4: Media Validator Stress & Boundary Tests
# ============================================================================

from wan_video_toolkit.diagnostics.media_validator import (
    apply_exif_transposition,
    validate_and_normalize_dimensions,
    load_conditioning_image,
    save_video_robust,
    VideoSaveError,
    dit_dtype_guard,
    assert_model_dtype,
    DtypeTrapError,
    validate_media_inputs,
)


@pytest.mark.parametrize("tag,expected_rotated", [
    (1, False),  # Normal
    (2, False),  # Flipped horizontal
    (3, False),  # 180 deg
    (4, False),  # Flipped vertical
    (5, True),   # Transposed (width/height swapped)
    (6, True),   # 90 CW (width/height swapped)
    (7, True),   # Transverse (width/height swapped)
    (8, True),   # 270 CW (width/height swapped)
])
def test_exif_orientation_all_tags_stress(tmp_path, tag, expected_rotated):
    """Stress test all 8 standard EXIF orientation tags."""
    img_path = tmp_path / f"tag_{tag}.jpg"
    w, h = 160, 80  # Non-square: 160 x 80
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    exif = img.getexif()
    exif[0x0112] = tag
    img.save(str(img_path), exif=exif)

    loaded = load_conditioning_image(str(img_path), enforce_divisible_16=True)
    if expected_rotated:
        # Dimensions swapped: width should be ~80, height ~160
        assert loaded.size[0] == 80
        assert loaded.size[1] == 160
    else:
        assert loaded.size[0] == 160
        assert loaded.size[1] == 80

    assert loaded.size[0] % 16 == 0
    assert loaded.size[1] % 16 == 0


def test_dimension_normalization_boundary_values():
    """Test boundary values for 16x normalization."""
    # Value smaller than 16 should clamp up to 16
    w, h, mod = validate_and_normalize_dimensions(5, 7, multiple=16)
    assert w == 16
    assert h == 16
    assert mod is True

    # Exact multiple should not be modified
    w, h, mod = validate_and_normalize_dimensions(1280, 720, multiple=16)
    assert (w, h) == (1280, 720)
    assert mod is False

    # Prime numbers
    w, h, mod = validate_and_normalize_dimensions(1013, 719, multiple=16)
    assert w % 16 == 0
    assert h % 16 == 0
    assert mod is True


def test_save_video_robust_tensor_permutations(tmp_path, mock_ffmpeg_cascade):
    """Stress test save_video_robust with different tensor dimensional layouts."""
    mock_ffmpeg_cascade(failing_codecs=[])

    # Shape 1: [F, H, W, C]
    t1 = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    out1 = str(tmp_path / "layout_fhwc.mp4")
    assert os.path.exists(save_video_robust(t1, out1))

    # Shape 2: [F, C, H, W]
    t2 = np.zeros((8, 3, 64, 64), dtype=np.uint8)
    out2 = str(tmp_path / "layout_fchw.mp4")
    assert os.path.exists(save_video_robust(t2, out2))

    # Shape 3: [C, F, H, W]
    t3 = np.zeros((3, 8, 64, 64), dtype=np.uint8)
    out3 = str(tmp_path / "layout_cfhw.mp4")
    assert os.path.exists(save_video_robust(t3, out3))

    # Shape 4: Floating point normalized [0, 1]
    t4 = np.ones((8, 64, 64, 3), dtype=np.float32) * 0.5
    out4 = str(tmp_path / "layout_float.mp4")
    assert os.path.exists(save_video_robust(t4, out4))
