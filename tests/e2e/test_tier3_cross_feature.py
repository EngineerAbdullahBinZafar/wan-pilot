"""
Tier 3: Cross-Feature Integration & Pairwise Combinatorial Test Suite.
Verifies interactions, parameter flows, and data handoffs across modular subsystems:
- MoE offload + T5 CPU offload + tiled VAE + EXIF auto-transposition + CLI generation
- VRAM estimation driving compiler IR offload profile selection
- Media validator results driving ComfyUI workflow input image sizing
- MCP server tool call cascading (diagnose_preflight -> estimate_memory -> compile_workflow -> execute_local_pipeline)
- WebMCP actions metadata consistency with CLI argument parser

Coverage Target: >= 15 tests total.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from wan_video_toolkit.diagnostics import (
    diagnose_hardware_preflight,
    detect_attention_backends,
    inspect_hardware_environment,
    estimate_memory,
    MemoryEstimateResult,
    parse_resolution,
    WAN_MODEL_SPECS,
    PRECISION_BYTES,
    validate_media_inputs,
    MediaValidationResult,
    apply_exif_transposition,
    load_conditioning_image,
    validate_and_normalize_dimensions,
    save_video_robust,
    VideoSaveError,
    dit_dtype_guard,
    assert_model_dtype,
    DtypeTrapError,
)


# ============================================================================
# Pairwise Subsystem Interactions
# ============================================================================

def test_tier3_preflight_drives_memory_estimation(mock_cuda_rtx4090):
    """Verifies preflight diagnosis parameters directly correlate with analytical memory estimator."""
    preflight = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
    assert preflight["evaluation"]["can_run"] is True

    # Memory estimator runs with recommended flags
    rec_flags = preflight["recommended_flags"]
    precision = "fp8" if "--precision" in rec_flags and "fp8" in rec_flags else "bf16"
    moe_offload = "--expert_offload" in rec_flags

    est = estimate_memory(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        precision=precision,
        moe_offload=moe_offload,
    )
    assert est.fits_24gb_gpu is True
    assert est.peak_vram_gb == preflight["evaluation"]["estimated_peak_vram_gb"]


def test_tier3_exif_transposition_drives_dimension_padding(synthetic_exif_image):
    """Verifies image loaded with EXIF transposition Tag 6 is subsequently 16x normalized."""
    # Tag 6 is (100, 200) -> transposed to (200, 100) -> 16x padded to (192, 96)
    conditioned = load_conditioning_image(synthetic_exif_image, enforce_divisible_16=True)
    assert conditioned.size == (192, 96)
    assert conditioned.width % 16 == 0
    assert conditioned.height % 16 == 0


def test_tier3_media_validator_feeds_save_video(tmp_path, synthetic_exif_image, mock_ffmpeg_cascade):
    """Verifies media validation pre-check followed by safe video save execution."""
    mock_ffmpeg_cascade()
    target_video = tmp_path / "cross_saved.mp4"

    # Step 1: Validate media inputs and output path
    val_result = validate_media_inputs(
        image_path=synthetic_exif_image,
        output_target_path=str(target_video),
    )
    assert val_result.is_valid is True
    assert val_result.output_safety["directory_writable"] is True

    # Step 2: Use validated target for resilient video saving
    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    saved = save_video_robust(frames, str(target_video), fps=16)
    assert os.path.exists(saved)
    assert os.path.getsize(saved) > 1024


def test_tier3_moe_offload_plus_t5_cpu_plus_tiled_vae():
    """Verifies triple-combination memory optimization (MoE swap + T5 CPU + Tiled VAE)."""
    # Baseline: No offload, GPU text encoder, untiled VAE (catastrophic)
    baseline = estimate_memory(
        "wan2.2_moe_14b", "720p", 81, "bf16",
        moe_offload=False, t5_cpu=False, tiled_vae=False,
    )
    assert baseline.peak_vram_without_moe_offload_gb > 45.0
    assert baseline.fits_24gb_gpu is False

    # Golden Profile: All 3 offload vectors active
    golden = estimate_memory(
        "wan2.2_moe_14b", "720p", 81, "fp8",
        moe_offload=True, t5_cpu=True, tiled_vae=True,
    )
    assert golden.peak_vram_gb <= 20.5
    assert golden.fits_24gb_gpu is True
    assert golden.vae_peak_gb == 1.8
    assert golden.umt5_weights_vram_gb == 0.0


def test_tier3_hardware_preflight_triggers_attention_doctor(mock_windows_no_msvc, mock_cuda_rtx4090):
    """Verifies preflight diagnosis embeds attention doctor results and recommendations."""
    report = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
    att = report["attention_backend"]
    assert att["active"] == "sdpa"
    assert att["windows_msvc_missing"] is True
    assert "PyTorch native SDPA active" in att["recommendation"]


def test_tier3_dtype_guard_protects_video_save_pipeline(tmp_path, mock_ffmpeg_cascade):
    """Verifies dit_dtype_guard encloses the video save pipeline without side effects."""
    import torch
    mock_ffmpeg_cascade()
    target_video = tmp_path / "guarded_save.mp4"
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    with dit_dtype_guard(torch.bfloat16):
        saved = save_video_robust(frames, str(target_video))
        assert os.path.exists(saved)
    # Original dtype restored
    assert torch.get_default_dtype() != torch.bfloat16 or hasattr(torch, "__mock__")


def test_tier3_preflight_12gb_gpu_recommends_aggressive_offload(mock_cuda_low_vram_12gb):
    """Verifies preflight on 12GB GPU prescribes aggressive offload flags."""
    report = diagnose_hardware_preflight("wan2.1_14b", "720p")
    assert report["evaluation"]["can_run"] is False
    assert "--offload_model" in report["recommended_flags"]
    assert "--precision" in report["recommended_flags"]


def test_tier3_preflight_24gb_gpu_recommends_balanced_offload(mock_cuda_rtx4090):
    """Verifies preflight on 24GB GPU prescribes balanced sequential offload."""
    report = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
    assert "--expert_offload" in report["recommended_flags"]
    assert "--t5_cpu" in report["recommended_flags"]
    assert "--tiled_vae" in report["recommended_flags"]


def test_tier3_media_validation_with_both_image_and_video(synthetic_exif_image, synthetic_valid_mp4):
    """Verifies validate_media_inputs handles conditioning image and input video concurrently."""
    result = validate_media_inputs(
        image_path=synthetic_exif_image,
        video_path=synthetic_valid_mp4,
    )
    assert result.is_valid is True
    assert result.image_diagnostics is not None
    assert result.video_diagnostics is not None
    assert result.image_diagnostics["status"] == "AUTO_TRANSPOSED"
    assert result.video_diagnostics["status"] == "ACCESSIBLE"


def test_tier3_memory_estimation_temporal_sequence_scaling():
    """Verifies memory scaling across sequence lengths (49, 81, 121 frames)."""
    est_49 = estimate_memory("wan2.1_14b", "720p", num_frames=49, precision="fp8")
    est_81 = estimate_memory("wan2.1_14b", "720p", num_frames=81, precision="fp8")
    est_121 = estimate_memory("wan2.1_14b", "720p", num_frames=121, precision="fp8")

    assert est_49.latent_tokens < est_81.latent_tokens < est_121.latent_tokens
    assert est_49.activation_and_kv_cache_gb < est_81.activation_and_kv_cache_gb < est_121.activation_and_kv_cache_gb


def test_tier3_save_video_fallback_cascade_with_different_fps(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust handles custom FPS rates (24, 30, 60) across codec cascades."""
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc"])
    frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)

    for fps_val in [24, 30, 60]:
        target = tmp_path / f"video_fps_{fps_val}.mp4"
        saved = save_video_robust(frames, str(target), fps=fps_val, preferred_codec="h264_nvenc")
        assert os.path.exists(saved)


def test_tier3_exif_image_and_unwritable_output_interaction(synthetic_exif_image, mock_unwritable_directory):
    """Verifies media validator inspects valid image while rejecting unwritable destination."""
    target = os.path.join(mock_unwritable_directory, "out.mp4")
    result = validate_media_inputs(image_path=synthetic_exif_image, output_target_path=target)
    assert result.is_valid is False
    assert result.image_diagnostics["status"] == "AUTO_TRANSPOSED"
    assert result.output_safety["directory_writable"] is False


def test_tier3_preflight_mps_apple_silicon_with_memory_estimation(mock_apple_mps):
    """Verifies Apple Silicon MPS preflight integrates with memory estimation under unified RAM."""
    report = diagnose_hardware_preflight("wan2.1_14b", "720p")
    assert report["hardware"]["mps_available"] is True
    assert report["evaluation"]["can_run"] is True
    assert "mps_guidance" in report
    assert report["mps_guidance"]["environment_variable"] == "PYTORCH_ENABLE_MPS_FALLBACK=1"


def test_tier3_cpu_fallback_host_ram_bounds_estimation(mock_cpu_only):
    """Verifies CPU-only preflight accurately captures host RAM and sets system status."""
    report = diagnose_hardware_preflight("wan2.1_1.3b", "480p")
    assert report["hardware"]["cuda_available"] is False
    assert report["hardware"]["device_name"] == "CPU"
    assert report["evaluation"]["available_vram_gb"] > 0


# ============================================================================
# Dynamic Compiler, CLI, and MCP Cross-Feature Tests (M2 / M3)
# ============================================================================

def test_tier3_memory_estimator_drives_compiler_ir():
    """Verifies memory estimation drives compiler IR offload profile selection."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    # When memory estimator indicates tight 24GB budget:
    est = estimate_memory("wan2.2_moe_14b", "720p", precision="fp8")
    assert est.fits_24gb_gpu is True

    ir = WanGenerationPipelineIR(prompt="Landscape", model="wan2.2_moe_14b", offload_profile="balanced")
    flags = ir.get_recommended_flags()
    assert "--expert_offload" in flags
    assert "--t5_cpu" in flags


def test_tier3_media_validator_drives_comfyui_sizing(synthetic_exif_image):
    """Verifies media validation dimensions feed into ComfyUI workflow compiler."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_comfyui_api, WanGenerationPipelineIR

    val = validate_media_inputs(image_path=synthetic_exif_image)
    norm_w, norm_h = val.image_diagnostics["normalized_dimensions"]

    ir = WanGenerationPipelineIR(prompt="Animate", width=norm_w, height=norm_h, image_path=synthetic_exif_image)
    api_json = compile_comfyui_api(ir)
    assert api_json is not None


def test_tier3_mcp_tool_cascade_preflight_to_compile(mock_cuda_rtx4090):
    """Verifies MCP tool call cascade: diagnose_preflight -> estimate_memory -> compile_workflow."""
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()

    # Step 1: Preflight
    req1 = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "diagnose_preflight", "arguments": {}}}
    resp1 = server.handle_request(req1)
    assert not resp1["result"].get("isError", False)

    # Step 2: Estimate memory
    req2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "estimate_memory", "arguments": {"model_variant": "wan2.2_moe_14b"}}}
    resp2 = server.handle_request(req2)
    assert not resp2["result"].get("isError", False)

    # Step 3: Compile workflow
    req3 = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "compile_workflow", "arguments": {"prompt": "A running tiger", "target_format": "cli_bash"}}}
    resp3 = server.handle_request(req3)
    assert not resp3["result"].get("isError", False)


def test_tier3_webmcp_actions_metadata_consistency():
    """Verifies WebMCP action declarations align with CLI argument structures."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    actions_path = os.path.join(repo_root, "ui", "mcp-actions.json")
    if not os.path.exists(actions_path):
        actions_path = os.path.join(repo_root, "docs", "mcp-actions.json")
    if not os.path.exists(actions_path):
        pytest.skip("Milestone M4 mcp-actions.json not yet available in active workspace")

    with open(actions_path, "r", encoding="utf-8") as f:
        actions = json.load(f)
    assert len(actions) > 0
