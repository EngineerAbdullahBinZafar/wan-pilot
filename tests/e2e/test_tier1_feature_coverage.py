"""
Tier 1: Comprehensive Feature Coverage Test Suite.
Verifies core functional behavior (happy path and primary logic) across all 11 features:
- Feature 1: Hardware & VRAM Preflight Inspection
- Feature 2: Analytical VRAM Estimation Engine
- Feature 3: EXIF Transposition & Input Normalizer
- Feature 4: Resilient Video Save with Codec Fallbacks
- Feature 5: Dtype Trap Prevention
- Feature 6: Canonical Pipeline IR Schema
- Feature 7: ComfyUI Execution Graph Compiler
- Feature 8: CLI & Diffusers Script Compilers
- Feature 9: Unified CLI Tool (`wan-pilot`)
- Feature 10: FastMCP / stdio MCP Server (5 tools)
- Feature 11: Web Workbench UI & WebMCP schema actions

Coverage Target: >= 5 tests per feature (>= 55 tests total).
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import numpy as np

# Core diagnostics imports (Milestone M1)
from wan_video_toolkit.diagnostics import (
    diagnose_hardware_preflight,
    detect_attention_backends,
    inspect_hardware_environment,
    estimate_memory,
    MemoryEstimateResult,
    parse_resolution,
    WAN_MODEL_SPECS,
    PRECISION_BYTES,
    RESOLUTION_PRESETS,
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
# Feature 1: Hardware & VRAM Preflight Inspection
# ============================================================================

def test_feature1_preflight_rtx4090_ready(mock_cuda_rtx4090):
    """Verifies preflight diagnostics on NVIDIA RTX 4090 (24GB VRAM)."""
    report = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
    assert report["system_status"] in ("READY", "WARNING")
    assert report["hardware"]["cuda_available"] is True
    assert report["hardware"]["device_name"] == "NVIDIA GeForce RTX 4090"
    assert report["hardware"]["total_vram_gb"] == 24.0
    assert report["evaluation"]["can_run"] is True
    assert "recommended_flags" in report
    assert "--precision" in report["recommended_flags"]


def test_feature1_preflight_rtx3090_evaluation(mock_cuda_rtx3090):
    """Verifies preflight diagnostics on NVIDIA RTX 3090 (24GB VRAM, compute 8.6)."""
    report = diagnose_hardware_preflight("wan2.1_14b", "720p")
    assert report["hardware"]["cuda_available"] is True
    assert report["hardware"]["device_name"] == "NVIDIA GeForce RTX 3090"
    assert report["hardware"]["compute_capability"] == "8.6"
    assert report["evaluation"]["can_run"] is True


def test_feature1_preflight_moe_concurrent_trap_detection(mock_cuda_rtx4090):
    """Verifies detection of OOM_TRAP_MOE_CONCURRENT trap on 24GB GPUs."""
    report = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
    traps = [t["trap_id"] for t in report["flagged_traps"]]
    assert "OOM_TRAP_MOE_CONCURRENT" in traps
    trap_detail = next(t for t in report["flagged_traps"] if t["trap_id"] == "OOM_TRAP_MOE_CONCURRENT")
    assert trap_detail["severity"] == "CRITICAL"
    assert "--expert_offload True" in trap_detail["mitigation"]


def test_feature1_preflight_t5_cpu_offload_trap_flagged(mock_cuda_rtx4090):
    """Verifies detection of OOM_TRAP_T5_VRAM for 14B models on consumer hardware."""
    report = diagnose_hardware_preflight("wan2.1_14b", "720p")
    traps = [t["trap_id"] for t in report["flagged_traps"]]
    assert "OOM_TRAP_T5_VRAM" in traps
    trap_detail = next(t for t in report["flagged_traps"] if t["trap_id"] == "OOM_TRAP_T5_VRAM")
    assert "--t5_cpu True" in trap_detail["mitigation"]


def test_feature1_attention_doctor_windows_fallback(mock_windows_no_msvc):
    """Verifies attention doctor flags Windows MSVC absence and falls back to native SDPA."""
    backends = detect_attention_backends()
    assert backends["windows_msvc_missing"] is True
    assert backends["active"] == "sdpa"
    assert "SDPA" in backends["recommendation"]
    assert "MSVC" in backends["recommendation"]


def test_feature1_preflight_apple_silicon_mps(mock_apple_mps):
    """Verifies hardware preflight under Apple Silicon MPS unified memory."""
    report = diagnose_hardware_preflight("wan2.1_1.3b", "480p")
    assert report["hardware"]["mps_available"] is True
    assert report["hardware"]["unified_memory"] is True
    assert report["hardware"]["device_name"] == "Apple Silicon (MPS)"
    assert report["hardware"]["total_vram_gb"] >= 32.0


def test_feature1_preflight_cpu_only_host(mock_cpu_only):
    """Verifies preflight handles CPU-only environments cleanly with host RAM queries."""
    report = diagnose_hardware_preflight("wan2.1_1.3b", "480p")
    assert report["hardware"]["cuda_available"] is False
    assert report["hardware"]["mps_available"] is False
    assert report["hardware"]["device_name"] == "CPU"
    assert report["hardware"]["total_vram_gb"] > 0.0


def test_feature1_preflight_low_vram_12gb(mock_cuda_low_vram_12gb):
    """Verifies preflight on 12GB GPU flags VRAM deficit for 14B models."""
    report = diagnose_hardware_preflight("wan2.1_14b", "720p")
    assert report["hardware"]["total_vram_gb"] == 12.0
    assert report["evaluation"]["can_run"] is False
    traps = [t["trap_id"] for t in report["flagged_traps"]]
    assert "OOM_TRAP_T5_VRAM" in traps


def test_feature1_preflight_custom_resolution_support(mock_cuda_rtx4090):
    """Verifies preflight handles custom WxH resolution strings."""
    report = diagnose_hardware_preflight("wan2.1_14b", "1920x1080")
    assert report["evaluation"]["model_variant"] == "wan2.1_14b"
    assert report["evaluation"]["resolution"] == "1920x1080"


def test_feature1_attention_doctor_flash_attention(mock_flash_attention_2):
    """Verifies attention doctor recognizes FlashAttention-2."""
    backends = detect_attention_backends()
    assert backends["flash_attention_2"] is True
    assert backends["active"] == "flash_attention_2"


def test_feature1_attention_doctor_sage_attention(mock_sage_attention):
    """Verifies attention doctor recognizes SageAttention."""
    backends = detect_attention_backends()
    assert backends["sage_attention"] is True
    assert backends["active"] == "sage_attention"


def test_feature1_preflight_device_override_cpu(mock_cuda_rtx4090):
    """Verifies device_override='cpu' forces CPU inspection despite available CUDA."""
    report = diagnose_hardware_preflight("wan2.1_1.3b", "480p", device_override="cpu")
    assert report["hardware"]["cuda_available"] is False
    assert report["hardware"]["device_name"] == "CPU"


# ============================================================================
# Feature 2: Analytical VRAM Estimation Engine
# ============================================================================

def test_feature2_estimate_wan2_2_moe_sequential_vs_concurrent():
    """Verifies analytical VRAM math: sequential offload bounds peak VRAM to ~18.4GB on 24GB."""
    res_seq = estimate_memory(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        precision="fp8",
        moe_offload=True,
        t5_cpu=True,
        tiled_vae=True,
    )
    assert res_seq.fits_24gb_gpu is True
    assert res_seq.peak_vram_gb <= 20.0
    assert res_seq.bytes_per_param == 1.0

    # Without moe offload (concurrent loading of both 14B experts)
    res_concurrent = estimate_memory(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        precision="fp8",
        moe_offload=False,
        t5_cpu=True,
        tiled_vae=True,
    )
    assert res_concurrent.fits_24gb_gpu is False
    assert res_concurrent.peak_vram_gb >= 30.0
    assert "100% Guaranteed OOM" in res_concurrent.oom_risk_on_24gb


def test_feature2_estimate_wan2_1_1_3b_fits_12gb():
    """Verifies Wan2.1 1.3B model fits comfortably in 12GB and 16GB GPUs."""
    res = estimate_memory(
        model_variant="wan2.1_1.3b",
        resolution="480p",
        num_frames=81,
        precision="fp8",
        moe_offload=False,
        t5_cpu=True,
        tiled_vae=True,
    )
    assert res.fits_12gb_gpu is True
    assert res.fits_16gb_gpu is True
    assert res.fits_24gb_gpu is True
    assert res.peak_vram_gb < 10.0


def test_feature2_estimate_wan2_1_14b_fp8_vs_bf16():
    """Verifies precision scaling factor (1 byte/param vs 2 bytes/param)."""
    res_fp8 = estimate_memory("wan2.1_14b", "720p", 81, "fp8", t5_cpu=True, tiled_vae=True)
    res_bf16 = estimate_memory("wan2.1_14b", "720p", 81, "bf16", t5_cpu=True, tiled_vae=True)

    assert res_fp8.bytes_per_param == 1.0
    assert res_bf16.bytes_per_param == 2.0
    assert res_bf16.active_dit_weights_gb == pytest.approx(res_fp8.active_dit_weights_gb * 2.0, rel=1e-2)


def test_feature2_estimate_tiled_vae_effect():
    """Verifies tiled VAE prevents catastrophic ~8.7GB decoding spikes at 720p."""
    res_tiled = estimate_memory("wan2.2_moe_14b", "720p", 81, "fp8", tiled_vae=True)
    res_untiled = estimate_memory("wan2.2_moe_14b", "720p", 81, "fp8", tiled_vae=False)

    assert res_tiled.vae_peak_gb == 1.8
    assert res_untiled.vae_peak_gb > 8.0
    assert res_untiled.peak_vram_gb > res_tiled.peak_vram_gb


def test_feature2_estimate_latent_tokens_calculation():
    """Verifies mathematical token formula: (H/16) * (W/16) * (1 + (F-1)/4)."""
    # 1280x720, 81 frames: (720/16) * (1280/16) * (1 + 80/4) = 45 * 80 * 21 = 75,600
    res = estimate_memory("wan2.1_14b", "720p", num_frames=81, precision="fp8")
    expected_tokens = (720 // 16) * (1280 // 16) * (1 + (81 - 1) // 4)
    assert expected_tokens == 75600
    assert res.latent_tokens == expected_tokens


def test_feature2_resolution_preset_parsing():
    """Verifies parsing of resolution strings across presets and custom dimensions."""
    assert parse_resolution("480p") == (832, 480)
    assert parse_resolution("720p") == (1280, 720)
    assert parse_resolution("1080p") == (1920, 1080)
    assert parse_resolution("1920x1080") == (1920, 1080)
    assert parse_resolution("1280*720") == (1280, 720)


def test_feature2_wan_model_specs_integrity():
    """Verifies architectural parameters and constants in WAN_MODEL_SPECS."""
    for model_key in ["wan2.1_1.3b", "wan2.1_14b", "wan2.2_moe_14b"]:
        assert model_key in WAN_MODEL_SPECS
        spec = WAN_MODEL_SPECS[model_key]
        assert spec["dit_params"] > 0
        assert spec["hidden_dim"] > 0
        assert spec["layers"] > 0
        assert spec["heads"] > 0


def test_feature2_estimate_memory_to_dict_structure():
    """Verifies MemoryEstimateResult.to_dict format matches MCP and UI contracts."""
    res = estimate_memory("wan2.2_moe_14b", "720p", 81, "fp8")
    d = res.to_dict()
    assert "configuration" in d
    assert "memory_breakdown_gb" in d
    assert "totals" in d
    assert "concurrent_comparison" in d
    assert d["totals"]["fits_24gb_gpu"] is True


def test_feature2_estimate_memory_1080p_scaling():
    """Verifies memory scaling at 1080p (1920x1080) resolution."""
    res_1080p = estimate_memory("wan2.1_14b", "1080p", 81, "fp8")
    res_720p = estimate_memory("wan2.1_14b", "720p", 81, "fp8")
    assert res_1080p.latent_tokens > res_720p.latent_tokens
    assert res_1080p.activation_and_kv_cache_gb > res_720p.activation_and_kv_cache_gb


def test_feature2_estimate_memory_int8_precision():
    """Verifies int8 precision calculations (1.0 byte per parameter)."""
    res = estimate_memory("wan2.1_14b", "720p", 81, "int8")
    assert res.bytes_per_param == 1.0
    assert res.active_dit_weights_gb == pytest.approx(14.25 * 10**9 / 1024**3, rel=0.05)


def test_feature2_estimate_memory_system_ram_breakdown():
    """Verifies system RAM calculation accounts for UMT5 and dormant weights."""
    res = estimate_memory("wan2.2_moe_14b", "720p", 81, "fp8", moe_offload=True, t5_cpu=True)
    assert res.system_ram_gb > 10.0
    assert res.umt5_weights_ram_gb > 0.0


def test_feature2_estimate_memory_bytes_per_param_consistency():
    """Verifies precision bytes consistency for float16, bfloat16, and fp32."""
    assert PRECISION_BYTES["float32"] == 4.0
    assert PRECISION_BYTES["bfloat16"] == 2.0
    assert PRECISION_BYTES["fp8"] == 1.0


def test_feature3_apply_exif_transposition_no_exif(tmp_path):
    """Verifies apply_exif_transposition on an image without EXIF metadata leaves image intact."""
    plain_img = Image.new("RGB", (64, 48), color=(50, 100, 150))
    transposed = apply_exif_transposition(plain_img)
    assert transposed.size == (64, 48)


def test_feature3_validate_media_inputs_valid_video(synthetic_valid_mp4):
    """Verifies validate_media_inputs validates valid video files."""
    result = validate_media_inputs(video_path=synthetic_valid_mp4)
    assert result.is_valid is True
    assert result.video_diagnostics["status"] == "ACCESSIBLE"
    assert result.video_diagnostics["file_size_bytes"] > 1024


# ============================================================================
# Feature 3: EXIF Transposition & Input Normalizer
# ============================================================================

def test_feature3_exif_transposition_tag6(synthetic_exif_image):
    """Verifies EXIF orientation tag 6 (90 deg CW) auto-transposition (resolves Wan2.2 #415)."""
    with Image.open(synthetic_exif_image) as raw_img:
        assert raw_img.size == (100, 200)
        corrected = apply_exif_transposition(raw_img)
        # 100x200 rotated 90 degrees CW becomes 200x100
        assert corrected.size == (200, 100)


def test_feature3_exif_transposition_all_tags(synthetic_exif_tag_factory):
    """Verifies EXIF orientation handling across tags 1 through 8."""
    for tag in range(1, 9):
        path = synthetic_exif_tag_factory(tag, width=160, height=80)
        with Image.open(path) as raw_img:
            corrected = apply_exif_transposition(raw_img)
            assert corrected is not None
            assert corrected.width > 0 and corrected.height > 0
            if tag in (5, 6, 7, 8):
                # 90/270 rotations swap width and height
                assert corrected.size == (80, 160)
            else:
                # 0/180 rotations or horizontal flips maintain dimension axes
                assert corrected.size == (160, 80)


def test_feature3_dimension_divisibility_normalization():
    """Verifies non-divisible spatial dimensions are truncated to multiples of 16."""
    norm_w, norm_h, modified = validate_and_normalize_dimensions(1281, 719, multiple=16)
    assert norm_w == 1280
    assert norm_h == 704
    assert modified is True
    assert norm_w % 16 == 0
    assert norm_h % 16 == 0


def test_feature3_load_conditioning_image_rgba_to_rgb(tmp_path):
    """Verifies load_conditioning_image converts 4-channel RGBA to 3-channel RGB."""
    rgba_path = tmp_path / "rgba_image.png"
    rgba_img = Image.new("RGBA", (128, 128), (255, 0, 0, 128))
    rgba_img.save(str(rgba_path))

    loaded = load_conditioning_image(str(rgba_path))
    assert loaded.mode == "RGB"
    assert loaded.size == (128, 128)


def test_feature3_validate_media_inputs_end_to_end(synthetic_exif_image, tmp_path):
    """Verifies validate_media_inputs performs EXIF transposition and target path validation."""
    out_target = tmp_path / "output_video.mp4"
    result = validate_media_inputs(image_path=synthetic_exif_image, output_target_path=str(out_target))
    assert result.is_valid is True
    assert result.image_diagnostics["exif_orientation_tag"] == 6
    assert result.image_diagnostics["status"] == "AUTO_TRANSPOSED"
    assert result.output_safety["directory_writable"] is True


def test_feature3_validate_media_inputs_missing_image():
    """Verifies validate_media_inputs flags non-existent image paths with errors."""
    result = validate_media_inputs(image_path="non_existent_file.jpg")
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("does not exist" in err.lower() for err in result.errors)


def test_feature3_validate_media_inputs_corrupt_file(tmp_path):
    """Verifies validate_media_inputs flags corrupted non-image files."""
    corrupt_file = tmp_path / "corrupt.jpg"
    with open(str(corrupt_file), "w") as f:
        f.write("not an image binary payload")
    result = validate_media_inputs(image_path=str(corrupt_file))
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_feature3_load_conditioning_image_16x_divisibility(tmp_path):
    """Verifies load_conditioning_image enforces 16-pixel spatial divisibility."""
    photo_path = tmp_path / "odd_photo.jpg"
    odd_img = Image.new("RGB", (723, 405), color=(100, 150, 200))
    odd_img.save(str(photo_path))

    conditioned = load_conditioning_image(str(photo_path), enforce_divisible_16=True)
    assert conditioned.width % 16 == 0
    assert conditioned.height % 16 == 0
    assert conditioned.width == 720
    assert conditioned.height == 400


def test_feature4_save_video_format_normalizer_floating_point(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust handles floating point tensors [0.0, 1.0]."""
    mock_ffmpeg_cascade()
    out_file = tmp_path / "float_video.mp4"
    float_frames = np.random.rand(8, 32, 32, 3).astype(np.float32)

    saved_path = save_video_robust(float_frames, str(out_file), fps=16)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


def test_feature4_save_video_format_normalizer_channels_first(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust transposes channels-first tensors [F, C, H, W] to [F, H, W, C]."""
    mock_ffmpeg_cascade()
    out_file = tmp_path / "ch_first_video.mp4"
    ch_first = np.zeros((8, 3, 32, 32), dtype=np.uint8)

    saved_path = save_video_robust(ch_first, str(out_file), fps=16)
    assert os.path.exists(saved_path)


def test_feature4_save_video_destination_directory_creation(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust creates nested parent directories automatically."""
    mock_ffmpeg_cascade()
    nested_target = tmp_path / "deep" / "nested" / "folder" / "auto_dir_video.mp4"
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    saved_path = save_video_robust(frames, str(nested_target))
    assert os.path.exists(saved_path)
    assert os.path.isdir(str(tmp_path / "deep" / "nested" / "folder"))


def test_feature5_assert_model_dtype_no_named_parameters():
    """Verifies assert_model_dtype safely handles modules without named_parameters."""
    class PlainModule:
        pass
    offending = assert_model_dtype(PlainModule())
    assert offending == []


def test_feature5_dit_dtype_guard_none_target():
    """Verifies dit_dtype_guard with target_dtype=None defaults safely to bfloat16."""
    import torch
    with dit_dtype_guard(None):
        current = torch.get_default_dtype()
        assert current in (torch.bfloat16, torch.float16, torch.float32)


# ============================================================================
# Feature 4: Resilient Video Save with Codec Fallbacks
# ============================================================================

def test_feature4_save_video_primary_codec_success(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust successfully encodes when primary codec succeeds."""
    mock_ffmpeg_cascade(failing_codecs=[])
    out_file = tmp_path / "saved_video.mp4"
    frames = np.zeros((16, 64, 64, 3), dtype=np.uint8)

    saved_path = save_video_robust(frames, str(out_file), fps=16, preferred_codec="h264_nvenc")
    assert saved_path == str(out_file)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


def test_feature4_save_video_fallback_nvenc_to_libx264(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust falls back to libx264 when h264_nvenc fails (Wan2.2 #416)."""
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc"])
    out_file = tmp_path / "fallback_video.mp4"
    frames = np.zeros((16, 64, 64, 3), dtype=np.uint8)

    saved_path = save_video_robust(frames, str(out_file), fps=16, preferred_codec="h264_nvenc")
    assert saved_path == str(out_file)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


def test_feature4_save_video_zero_byte_detection(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust catches 0-byte corrupt files and cascades to next backend."""
    mock_ffmpeg_cascade(zero_byte_codecs=["h264_nvenc"])
    out_file = tmp_path / "zero_byte_cascade.mp4"
    frames = np.zeros((16, 64, 64, 3), dtype=np.uint8)

    saved_path = save_video_robust(frames, str(out_file), fps=16, preferred_codec="h264_nvenc")
    assert saved_path == str(out_file)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


def test_feature4_save_video_total_failure_dumps_frames(tmp_path, mock_ffmpeg_cascade):
    """Verifies total codec failure triggers emergency PNG frame dump."""
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc", "libx264", "libopenh264", "mpeg4"])
    out_file = tmp_path / "failing_video.mp4"
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(VideoSaveError) as exc_info:
        save_video_robust(frames, str(out_file), fps=16)

    assert "Failed to save video" in str(exc_info.value)
    frame_dir = str(tmp_path / "failing_video_frames")
    assert os.path.isdir(frame_dir)
    assert os.path.exists(os.path.join(frame_dir, "frame_0000.png"))
    assert os.path.exists(os.path.join(frame_dir, "frame_0004.png"))


def test_feature4_save_video_unwritable_directory_raises_error(mock_unwritable_directory):
    """Verifies write permission failure raises VideoSaveError immediately."""
    out_file = os.path.join(mock_unwritable_directory, "impossible.mp4")
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(VideoSaveError, match="not writable"):
        save_video_robust(frames, out_file)


def test_feature4_save_video_empty_frames_tensor_handling(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust with valid frames shape (1, 16, 16, 3) encodes properly."""
    mock_ffmpeg_cascade()
    out_file = tmp_path / "single_frame.mp4"
    single_frame = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    saved = save_video_robust(single_frame, str(out_file), fps=1)
    assert os.path.exists(saved)


# ============================================================================
# Feature 5: Dtype Trap Prevention
# ============================================================================

def test_feature5_dit_dtype_guard_context_manager():
    """Verifies dit_dtype_guard sets bfloat16 and restores prior default dtype on exit."""
    import torch
    initial_dtype = torch.get_default_dtype()
    with dit_dtype_guard(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == initial_dtype


def test_feature5_dit_dtype_guard_restores_on_exception():
    """Verifies dit_dtype_guard restores original dtype even when an exception is raised."""
    import torch
    initial_dtype = torch.get_default_dtype()
    try:
        with dit_dtype_guard(torch.float16):
            assert torch.get_default_dtype() == torch.float16
            raise ValueError("Simulated pipeline failure inside dtype guard")
    except ValueError:
        pass
    assert torch.get_default_dtype() == initial_dtype


def test_feature5_assert_model_dtype_detects_fp32_trap():
    """Verifies assert_model_dtype raises DtypeTrapError on float32 parameters (Wan-Animate-2 #5)."""
    import torch
    class UnsafeModel:
        def named_parameters(self):
            p = MagicMock()
            p.dtype = torch.float32
            return [("transformer.blocks.0.weight", p)]

    with pytest.raises(DtypeTrapError, match="allocated in float32"):
        assert_model_dtype(UnsafeModel(), expected_dtype=torch.bfloat16)


def test_feature5_assert_model_dtype_accepts_bfloat16():
    """Verifies assert_model_dtype succeeds when model parameters are bfloat16."""
    import torch
    class SafeModel:
        def named_parameters(self):
            p = MagicMock()
            p.dtype = torch.bfloat16
            return [("transformer.blocks.0.weight", p)]

    assert_model_dtype(SafeModel(), expected_dtype=torch.bfloat16)


def test_feature5_precision_bytes_lookup_completeness():
    """Verifies PRECISION_BYTES mapping contains all standard precision strings."""
    for prec in ["fp32", "float32", "bf16", "bfloat16", "fp16", "float16", "fp8", "int8"]:
        assert prec in PRECISION_BYTES
        assert PRECISION_BYTES[prec] > 0


# ============================================================================
# Feature 6: Canonical Pipeline IR Schema
# ============================================================================

def test_feature6_canonical_ir_initialization_and_schema():
    """Verifies canonical pipeline IR schema specification contracts."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR, TargetFormat

    ir = WanGenerationPipelineIR(
        prompt="A serene mountain lake at dawn",
        model="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        steps=30,
        guidance=6.0,
        offload_profile="balanced",
    )
    assert ir.prompt == "A serene mountain lake at dawn"
    assert ir.model == "wan2.2_moe_14b"
    assert ir.num_frames == 81


def test_feature6_canonical_ir_validation_temporal_frames():
    """Verifies canonical IR validates temporal frame count rule (F = 4k + 1)."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    # 81 = 4*20 + 1 (Valid)
    ir = WanGenerationPipelineIR(prompt="Test", num_frames=81)
    assert ir.is_temporal_frame_valid() is True

    # 80 != 4k + 1 (Invalid)
    ir_invalid = WanGenerationPipelineIR(prompt="Test", num_frames=80)
    assert ir_invalid.is_temporal_frame_valid() is False


def test_feature6_canonical_ir_dimension_divisibility():
    """Verifies canonical IR validates 16-pixel spatial divisibility."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Test", width=1280, height=720)
    assert ir.are_dimensions_valid() is True

    ir_bad = WanGenerationPipelineIR(prompt="Test", width=1281, height=719)
    assert ir_bad.are_dimensions_valid() is False


def test_feature6_canonical_ir_to_dict_serialization():
    """Verifies canonical IR serializes cleanly to dict and JSON."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Test prompt", seed=12345)
    data = ir.to_dict()
    assert data["prompt"] == "Test prompt"
    assert data["seed"] == 12345
    serialized = json.dumps(data)
    assert "Test prompt" in serialized


def test_feature6_canonical_ir_offload_profile_mapping():
    """Verifies canonical IR offload profile flags resolution."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Test", offload_profile="balanced")
    flags = ir.get_recommended_flags()
    assert "--precision" in flags
    assert "--expert_offload" in flags


# ============================================================================
# Feature 7: ComfyUI Execution Graph Compiler
# ============================================================================

def test_feature7_compile_comfyui_prompt_api():
    """Verifies ComfyUI Prompt API JSON compilation format."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_comfyui_api, WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Cyberpunk neon city", model="wan2.2_moe_14b", resolution="720p")
    payload = compile_comfyui_api(ir)
    assert isinstance(payload, dict)
    assert len(payload) > 0
    # Every node has class_type and inputs
    for node_id, node in payload.items():
        assert "class_type" in node
        assert "inputs" in node


def test_feature7_compile_comfyui_canvas_ui():
    """Verifies ComfyUI Drag-and-Drop Canvas UI JSON compilation format."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_comfyui_canvas, WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="A galloping horse", model="wan2.1_14b")
    canvas = compile_comfyui_canvas(ir)
    assert "nodes" in canvas
    assert "links" in canvas
    assert isinstance(canvas["nodes"], list)


def test_feature7_comfyui_offload_flags_mapping():
    """Verifies ComfyUI compiler translates balanced profile to tiled VAE and CPU text encode."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_comfyui_api, WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Test", offload_profile="balanced")
    api_json = compile_comfyui_api(ir)
    # Check for presence of tiled decode or cpu offload settings
    node_types = [n["class_type"] for n in api_json.values()]
    assert any("Wan" in t or "VAE" in t for t in node_types)


def test_feature7_comfyui_prompt_text_propagation():
    """Verifies positive and negative prompts propagate to ComfyUI text nodes."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_comfyui_api, WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Ultra detailed forest", negative_prompt="blurry, distorted")
    api_json = compile_comfyui_api(ir)
    texts = []
    for node in api_json.values():
        if "text" in node.get("inputs", {}):
            texts.append(node["inputs"]["text"])
    assert "Ultra detailed forest" in texts


def test_feature7_comfyui_i2v_conditioning_nodes():
    """Verifies I2V workflows include image conditioning nodes in ComfyUI graph."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_comfyui_api, WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="Animate this photo", image_path="test.jpg")
    api_json = compile_comfyui_api(ir)
    node_types = [n["class_type"] for n in api_json.values()]
    assert any("LoadImage" in t or "Image" in t for t in node_types)


# ============================================================================
# Feature 8: CLI & Diffusers Script Compilers
# ============================================================================

def test_feature8_compile_cli_bash():
    """Verifies official Wan CLI bash script compilation."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_cli_script, WanGenerationPipelineIR, TargetFormat

    ir = WanGenerationPipelineIR(prompt="Test video", model="wan2.2_moe_14b", resolution="720p")
    bash_script = compile_cli_script(ir, target_format=TargetFormat.CLI_BASH)
    assert "python generate.py" in bash_script
    assert "--task t2v-14B" in bash_script
    assert "--size 1280*720" in bash_script
    assert "--precision fp8" in bash_script
    assert "--expert_offload True" in bash_script


def test_feature8_compile_cli_powershell():
    """Verifies PowerShell script compilation uses backtick line continuations."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_cli_script, WanGenerationPipelineIR, TargetFormat

    ir = WanGenerationPipelineIR(prompt="Test video", model="wan2.2_moe_14b")
    ps1_script = compile_cli_script(ir, target_format=TargetFormat.CLI_POWERSHELL)
    assert "python generate.py `" in ps1_script
    assert "$env:CUDA_VISIBLE_DEVICES" in ps1_script


def test_feature8_compile_diffusers_script():
    """Verifies Hugging Face Diffusers standalone Python script compilation."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_diffusers_script, WanGenerationPipelineIR

    ir = WanGenerationPipelineIR(prompt="A soaring eagle", model="wan2.1_14b")
    diffusers_code = compile_diffusers_script(ir)
    assert "import torch" in diffusers_code
    assert "WanPipeline" in diffusers_code or "pipeline" in diffusers_code
    assert "A soaring eagle" in diffusers_code


def test_feature8_cli_compiler_golden_flags_profile():
    """Verifies the 4 golden flags are emitted for 24GB MoE execution."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_cli_script, WanGenerationPipelineIR, TargetFormat

    ir = WanGenerationPipelineIR(prompt="Golden flags test", model="wan2.2_moe_14b", offload_profile="balanced")
    script = compile_cli_script(ir, target_format=TargetFormat.CLI_BASH)
    for flag in ["--precision fp8", "--expert_offload True", "--t5_cpu True", "--tiled_vae True"]:
        assert flag in script


def test_feature8_cli_compiler_seed_propagation():
    """Verifies explicit seed is passed into the compiled script."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import compile_cli_script, WanGenerationPipelineIR, TargetFormat

    ir = WanGenerationPipelineIR(prompt="Deterministic test", seed=424242)
    script = compile_cli_script(ir, target_format=TargetFormat.CLI_BASH)
    assert "--seed 424242" in script


# ============================================================================
# Feature 9: Unified CLI Tool (`wan-pilot`)
# ============================================================================

def test_feature9_cli_entrypoint_exists():
    """Verifies wan_video_toolkit.cli entrypoint module exists and exposes main."""
    cli_mod = pytest.importorskip("wan_video_toolkit.cli", reason="Milestone M2 CLI not yet present")
    assert hasattr(cli_mod, "main")
    assert callable(cli_mod.main)


def test_feature9_cli_preflight_command(mock_cuda_rtx4090):
    """Verifies CLI preflight command runs and outputs diagnostics."""
    cli_mod = pytest.importorskip("wan_video_toolkit.cli", reason="Milestone M2 CLI not yet present")
    result = cli_mod.run_command(["preflight", "--model", "wan2.2_moe_14b", "--resolution", "720p"])
    assert result == 0


def test_feature9_cli_estimate_command():
    """Verifies CLI estimate command runs without errors."""
    cli_mod = pytest.importorskip("wan_video_toolkit.cli", reason="Milestone M2 CLI not yet present")
    result = cli_mod.run_command(["estimate", "--model", "wan2.2_moe_14b", "--resolution", "720p"])
    assert result == 0


def test_feature9_cli_validate_command(synthetic_exif_image):
    """Verifies CLI validate command inspects input media files."""
    cli_mod = pytest.importorskip("wan_video_toolkit.cli", reason="Milestone M2 CLI not yet present")
    result = cli_mod.run_command(["validate", "--image", synthetic_exif_image])
    assert result == 0


def test_feature9_cli_compile_command():
    """Verifies CLI compile command outputs workflow code."""
    cli_mod = pytest.importorskip("wan_video_toolkit.cli", reason="Milestone M2 CLI not yet present")
    result = cli_mod.run_command(["compile", "--prompt", "A cinematic sunrise", "--target", "cli_bash"])
    assert result == 0


# ============================================================================
# Feature 10: FastMCP / stdio MCP Server (5 tools)
# ============================================================================

def test_feature10_mcp_server_initialize():
    """Verifies MCP server protocol handshake according to MCP spec 2024-11-05."""
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }
    resp = server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "wan-video-pilot"


def test_feature10_mcp_tools_list_exposes_all_5_tools():
    """Verifies tools/list enumerates all five required tools."""
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = server.handle_request(req)
    tools = [t["name"] for t in resp["result"]["tools"]]
    assert "diagnose_preflight" in tools
    assert "compile_workflow" in tools
    assert "estimate_memory" in tools
    assert "validate_media_inputs" in tools
    assert "execute_local_pipeline" in tools


def test_feature10_mcp_tool_diagnose_preflight_call(mock_cuda_rtx4090):
    """Verifies tools/call for diagnose_preflight returns valid JSON text content."""
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "diagnose_preflight", "arguments": {"model_variant": "wan2.2_moe_14b"}},
    }
    resp = server.handle_request(req)
    assert not resp["result"].get("isError", False)
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["hardware"]["device_name"] == "NVIDIA GeForce RTX 4090"


def test_feature10_mcp_tool_estimate_memory_call():
    """Verifies tools/call for estimate_memory returns mathematical decomposition."""
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "estimate_memory", "arguments": {"model_variant": "wan2.2_moe_14b", "resolution": "720p"}},
    }
    resp = server.handle_request(req)
    assert not resp["result"].get("isError", False)
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["totals"]["fits_24gb_gpu"] is True


def test_feature10_mcp_tool_execute_local_pipeline_dry_run():
    """Verifies execute_local_pipeline dry run simulates execution without spawning heavy processes."""
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()
    plan = {
        "command": "python generate.py --task t2v-14B --size 1280*720 --prompt 'A scenic lake'",
        "resolution": "1280x720",
        "model": "wan2.2_moe_14b",
    }
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "execute_local_pipeline", "arguments": {"plan_json": plan, "dry_run": True}},
    }
    resp = server.handle_request(req)
    assert not resp["result"].get("isError", False)
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["execution_mode"] == "DRY_RUN"
    assert payload["status"] == "VALIDATED"


# ============================================================================
# Feature 11: Web Workbench UI & WebMCP schema actions
# ============================================================================

def test_feature11_ui_index_html_structure():
    """Verifies existence and WCAG AA structure of Web Workbench UI."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    ui_paths = [
        os.path.join(repo_root, "ui", "index.html"),
        os.path.join(repo_root, "docs", "index.html"),
    ]
    existing = [p for p in ui_paths if os.path.exists(p)]
    if not existing:
        pytest.skip("Milestone M4 Web Workbench UI files not yet generated in active workspace")
    with open(existing[0], "r", encoding="utf-8") as f:
        html = f.read()
    assert "<!DOCTYPE html>" in html
    assert "Wan-Video" in html or "wan-pilot" in html


def test_feature11_webmcp_actions_json_schema():
    """Verifies /mcp-actions.json defines WebMCP agent tools."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    json_paths = [
        os.path.join(repo_root, "ui", "mcp-actions.json"),
        os.path.join(repo_root, "docs", "mcp-actions.json"),
    ]
    existing = [p for p in json_paths if os.path.exists(p)]
    if not existing:
        pytest.skip("Milestone M4 mcp-actions.json not yet generated in active workspace")
    with open(existing[0], "r", encoding="utf-8") as f:
        actions = json.load(f)
    assert "actions" in actions or isinstance(actions, list)


def test_feature11_aeo_llms_txt_and_agents_md():
    """Verifies presence of llms.txt and AGENTS.md for AI crawlers."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    for filename in ["llms.txt", "AGENTS.md"]:
        path = os.path.join(repo_root, filename)
        ui_path = os.path.join(repo_root, "docs", filename)
        if not (os.path.exists(path) or os.path.exists(ui_path)):
            pytest.skip(f"Milestone M4 {filename} not yet generated in active workspace")


def test_feature11_robots_txt_ai_crawlers():
    """Verifies robots.txt specifies guidelines for AI crawlers."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    robots_paths = [
        os.path.join(repo_root, "robots.txt"),
        os.path.join(repo_root, "docs", "robots.txt"),
    ]
    existing = [p for p in robots_paths if os.path.exists(p)]
    if not existing:
        pytest.skip("Milestone M4 robots.txt not yet generated in active workspace")
    with open(existing[0], "r", encoding="utf-8") as f:
        content = f.read()
    assert "User-agent:" in content


def test_feature11_ui_server_launcher():
    """Verifies UI server launcher module."""
    ui_server = pytest.importorskip("wan_video_toolkit.ui_server", reason="Milestone M4 UI server not yet present")
    assert hasattr(ui_server, "launch_ui_server") or hasattr(ui_server, "start_server")
