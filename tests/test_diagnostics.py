"""
Unit tests for hardware preflight inspection, attention backend doctor,
OOM trap flagging, and analytical memory estimation.
"""

import pytest
from wan_video_toolkit.diagnostics import (
    diagnose_hardware_preflight,
    detect_attention_backends,
    inspect_hardware_environment,
    estimate_memory,
    MemoryEstimateResult,
    parse_resolution,
)


def test_parse_resolution_presets_and_custom():
    w, h = parse_resolution("480p")
    assert (w, h) == (832, 480)

    w, h = parse_resolution("720p")
    assert (w, h) == (1280, 720)

    w, h = parse_resolution("1080p")
    assert (w, h) == (1920, 1080)

    w, h = parse_resolution("1920x1080")
    assert (w, h) == (1920, 1080)

    w, h = parse_resolution("832*480")
    assert (w, h) == (832, 480)


def test_hardware_inspection_with_mock_cuda(mock_cuda_rtx4090):
    hw = inspect_hardware_environment()
    assert hw["cuda_available"] is True
    assert hw["device_name"] == "NVIDIA GeForce RTX 4090"
    assert hw["total_vram_gb"] == 24.0
    assert hw["free_vram_gb"] == 22.4
    assert hw["compute_capability"] == "8.9"


def test_attention_backends_windows_fallback(mock_windows_no_msvc):
    attn = detect_attention_backends()
    assert attn["windows_msvc_missing"] is True
    assert attn["flash_attention_2"] is False
    assert attn["sage_attention"] is False
    assert attn["active"] in ("sdpa", "xformers", "naive")
    assert "SDPA" in attn["recommendation"] or "SDPA active" in attn["recommendation"]


def test_diagnose_hardware_preflight_rtx4090(mock_cuda_rtx4090, mock_windows_no_msvc):
    report = diagnose_hardware_preflight(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
    )

    assert report["system_status"] in ("READY", "WARNING_LOW_HEADROOM")
    assert report["hardware"]["device_name"] == "NVIDIA GeForce RTX 4090"
    assert report["evaluation"]["can_run"] is True

    # Check that OOM traps are detected
    trap_ids = [t["trap_id"] for t in report["flagged_traps"]]
    assert "OOM_TRAP_MOE_CONCURRENT" in trap_ids
    assert "OOM_TRAP_T5_VRAM" in trap_ids
    assert "OOM_TRAP_DIT_FP32" in trap_ids
    assert "OOM_TRAP_VAE_DEC_SPIKE" in trap_ids

    # Check golden flags for 24GB GPU
    flags = report["recommended_flags"]
    assert "--precision" in flags and "fp8" in flags
    assert "--expert_offload" in flags and "True" in flags
    assert "--t5_cpu" in flags and "True" in flags
    assert "--tiled_vae" in flags and "True" in flags


def test_estimate_memory_wan2_1_1_3b_480p():
    res = estimate_memory(
        model_variant="wan2.1_1.3b",
        resolution="480p",
        num_frames=81,
        precision="bf16",
        t5_cpu=True,
        tiled_vae=True,
    )
    assert isinstance(res, MemoryEstimateResult)
    assert res.latent_tokens == 32760  # (832/16) * (480/16) * (1 + 80/4) = 52 * 30 * 21 = 32760
    assert res.active_dit_params == 1_332_602_880
    assert res.bytes_per_param == 2.0
    assert res.fits_24gb_gpu is True
    assert res.fits_16gb_gpu is True
    assert res.fits_12gb_gpu is True


def test_estimate_memory_wan2_1_14b_720p():
    res = estimate_memory(
        model_variant="wan2.1_14b",
        resolution="720p",
        num_frames=81,
        precision="fp8",
        t5_cpu=True,
        tiled_vae=True,
    )
    assert res.latent_tokens == 75600  # (1280/16) * (720/16) * 21 = 80 * 45 * 21 = 75600
    assert res.active_dit_params == 14_249_779_200
    assert res.bytes_per_param == 1.0
    assert res.active_dit_weights_gb > 13.0  # ~13.27 GB
    assert res.fits_24gb_gpu is True


def test_estimate_memory_wan2_2_moe_offload_contrast():
    # Sequential expert swapping enabled
    res_offload = estimate_memory(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        precision="fp8",
        moe_offload=True,
        t5_cpu=True,
        tiled_vae=True,
    )
    assert res_offload.active_dit_weights_gb > 13.0
    assert res_offload.dormant_expert_weights_gb == 0.0
    assert res_offload.peak_vram_gb < 23.5
    assert res_offload.fits_24gb_gpu is True
    assert res_offload.oom_risk_on_24gb == "Safe on 24GB GPUs (RTX 3090/4090)"

    # Sequential offload disabled (concurrent loading)
    res_concurrent = estimate_memory(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        precision="fp8",
        moe_offload=False,
        t5_cpu=True,
        tiled_vae=True,
    )
    assert res_concurrent.dormant_expert_weights_gb > 10.0
    assert res_concurrent.peak_vram_without_moe_offload_gb > 30.0
    assert res_concurrent.fits_24gb_gpu is False
    assert "OOM" in res_concurrent.oom_risk_on_24gb


def test_tiled_vs_untiled_vae_spike():
    res_tiled = estimate_memory(
        model_variant="wan2.1_14b",
        resolution="720p",
        num_frames=81,
        tiled_vae=True,
    )
    assert res_tiled.vae_peak_gb == 1.8

    res_untiled = estimate_memory(
        model_variant="wan2.1_14b",
        resolution="720p",
        num_frames=81,
        tiled_vae=False,
    )
    # 12.5 * 81 * 1280 * 720 / 1024^3 ~= 8.69 GB
    assert res_untiled.vae_peak_gb > 8.0
    assert res_untiled.peak_vram_gb > res_tiled.peak_vram_gb


def test_precision_scaling_footprint():
    res_fp32 = estimate_memory(model_variant="wan2.1_1.3b", precision="fp32")
    res_bf16 = estimate_memory(model_variant="wan2.1_1.3b", precision="bf16")
    res_fp8 = estimate_memory(model_variant="wan2.1_1.3b", precision="fp8")

    assert res_fp32.bytes_per_param == 4.0
    assert res_bf16.bytes_per_param == 2.0
    assert res_fp8.bytes_per_param == 1.0

    assert res_fp32.active_dit_weights_gb == pytest.approx(2.0 * res_bf16.active_dit_weights_gb, rel=0.05)
    assert res_bf16.active_dit_weights_gb == pytest.approx(2.0 * res_fp8.active_dit_weights_gb, rel=0.05)
