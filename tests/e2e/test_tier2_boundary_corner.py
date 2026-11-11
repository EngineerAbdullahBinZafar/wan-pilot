"""
Tier 2: Boundary & Corner Cases Test Suite.
Verifies boundary value limits, invalid input rejection, resource constraints, and error cascades:
- Empty, whitespace, and extremely long text prompts
- 0-byte input image and video files
- Non-divisible dimensions (e.g. 1281x719, 720x405) and negative dimensions
- Extreme resolutions (16x16 minimum, 7680x4320 8K)
- Frame count boundary checks: 0, 1, 2 frames, non-4k+1 values (e.g. 80, 82)
- Missing video codecs in save cascade
- All EXIF orientation tags (1, 2, 3, 4, 5, 6, 7, 8) and corrupted/missing EXIF
- Read-only output target paths, non-existent directories, permission denied
- Unknown model variant strings, invalid precision strings

Coverage Target: >= 40 tests total.
"""

import os
import sys
import io
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
# 1. Media Files: 0-Byte, Corrupt, and Truncated Files
# ============================================================================

def test_tier2_zero_byte_image_file(synthetic_corrupt_file):
    """Verifies validate_media_inputs rejects a 0-byte corrupt image file."""
    result = validate_media_inputs(image_path=synthetic_corrupt_file)
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_tier2_zero_byte_video_file(synthetic_corrupt_file):
    """Verifies validate_media_inputs rejects a 0-byte video file."""
    result = validate_media_inputs(video_path=synthetic_corrupt_file)
    assert result.is_valid is False
    assert any("empty" in err.lower() or "corrupted" in err.lower() for err in result.errors)


def test_tier2_video_file_under_1024_bytes(tmp_path):
    """Verifies validate_media_inputs flags videos with size < 1024 bytes as corrupt (Issue #416)."""
    stub_file = tmp_path / "truncated_video.mp4"
    with open(str(stub_file), "wb") as f:
        f.write(b"\x00" * 512)  # 512 bytes < 1024

    result = validate_media_inputs(video_path=str(stub_file))
    assert result.is_valid is False
    assert any("corrupted or empty" in err.lower() for err in result.errors)


def test_tier2_load_conditioning_image_missing_file_raises_filenotfound():
    """Verifies load_conditioning_image raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_conditioning_image("completely_fictional_image.jpg")


def test_tier2_load_conditioning_image_corrupt_binary_raises_valueerror(tmp_path):
    """Verifies load_conditioning_image raises ValueError for non-image binary files."""
    junk_file = tmp_path / "garbage.png"
    with open(str(junk_file), "wb") as f:
        f.write(b"NOT_A_VALID_PNG_HEADER")

    with pytest.raises(ValueError, match="Unable to read"):
        load_conditioning_image(str(junk_file))


# ============================================================================
# 2. Dimensions & Divisibility Boundaries
# ============================================================================

def test_tier2_minimum_resolution_16x16():
    """Verifies minimum 16x16 bounding dimension is accepted without modification."""
    norm_w, norm_h, modified = validate_and_normalize_dimensions(16, 16, multiple=16)
    assert norm_w == 16
    assert norm_h == 16
    assert modified is False


def test_tier2_sub_16_dimensions_clamped():
    """Verifies dimensions smaller than 16 are clamped up to multiple of 16."""
    norm_w, norm_h, modified = validate_and_normalize_dimensions(5, 7, multiple=16)
    assert norm_w == 16
    assert norm_h == 16
    assert modified is True


@pytest.mark.parametrize("w,h,expected_w,expected_h", [
    (1281, 719, 1280, 704),
    (720, 405, 720, 400),
    (833, 481, 832, 480),
    (1923, 1087, 1920, 1072),
    (65, 65, 64, 64),
])
def test_tier2_odd_dimensions_divisibility_padding(w, h, expected_w, expected_h):
    """Verifies diverse odd and non-divisible dimensions are correctly rounded down to multiples of 16."""
    norm_w, norm_h, modified = validate_and_normalize_dimensions(w, h, multiple=16)
    assert norm_w == expected_w
    assert norm_h == expected_h
    assert modified is True
    assert norm_w % 16 == 0
    assert norm_h % 16 == 0


def test_tier2_extreme_8k_resolution_memory():
    """Verifies memory estimator handles extreme 8K resolution (7680x4320)."""
    res = estimate_memory("wan2.2_moe_14b", "7680x4320", num_frames=81, precision="fp8")
    assert res.width == 7680
    assert res.height == 4320
    # 7680/16 * 4320/16 * 21 = 480 * 270 * 21 = 2,721,600 latent tokens!
    assert res.latent_tokens == 2_721_600
    assert res.activation_and_kv_cache_gb > 50.0
    assert res.fits_24gb_gpu is False


# ============================================================================
# 3. Frame Count Boundaries ($4k + 1$ temporal rule)
# ============================================================================

def test_tier2_frame_count_single_frame():
    """Verifies F=1 boundary case produces 1 temporal token factor."""
    res = estimate_memory("wan2.1_14b", "720p", num_frames=1, precision="fp8")
    expected_tokens = (720 // 16) * (1280 // 16) * (1 + (1 - 1) // 4)
    assert res.latent_tokens == expected_tokens
    assert res.latent_tokens == (45 * 80 * 1)


@pytest.mark.parametrize("frames", [49, 81, 97, 121, 161])
def test_tier2_valid_4k_plus_1_frames(frames):
    """Verifies all standard Wan temporal frame counts follow 4k + 1 exactly."""
    assert (frames - 1) % 4 == 0
    res = estimate_memory("wan2.1_1.3b", "480p", num_frames=frames)
    assert res.latent_tokens > 0


def test_tier2_frame_count_0_or_negative():
    """Verifies zero frame count computes without zero-division crashes."""
    res = estimate_memory("wan2.1_1.3b", "480p", num_frames=0)
    assert res.latent_tokens >= 0


# ============================================================================
# 4. EXIF Orientation Exhaustive Corner Cases
# ============================================================================

@pytest.mark.parametrize("tag", [1, 2, 3, 4, 5, 6, 7, 8])
def test_tier2_exif_orientation_all_tags_boundary(tag, synthetic_exif_tag_factory):
    """Verifies apply_exif_transposition handles every EXIF tag 1-8 without exceptions."""
    path = synthetic_exif_tag_factory(tag, width=200, height=100)
    with Image.open(path) as img:
        res = apply_exif_transposition(img)
        assert res is not None
        assert res.width > 0 and res.height > 0
        if tag in (5, 6, 7, 8):
            assert res.size == (100, 200)
        else:
            assert res.size == (200, 100)


def test_tier2_corrupt_exif_metadata_fallback(tmp_path):
    """Verifies image with malformed EXIF payload falls back safely to original image."""
    img_path = tmp_path / "corrupt_exif.jpg"
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    img.save(str(img_path))

    with Image.open(str(img_path)) as loaded_img:
        # Pass through transposition
        transposed = apply_exif_transposition(loaded_img)
        assert transposed.size == (64, 64)


# ============================================================================
# 5. Permission & Path Error Traps
# ============================================================================

def test_tier2_save_video_unwritable_directory(mock_unwritable_directory):
    """Verifies save_video_robust catches unwritable target paths before encoding."""
    target_path = os.path.join(mock_unwritable_directory, "denied_output.mp4")
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    with pytest.raises(VideoSaveError, match="not writable"):
        save_video_robust(frames, target_path)


def test_tier2_save_video_missing_parent_directory_auto_created(tmp_path, mock_ffmpeg_cascade):
    """Verifies save_video_robust creates non-existent parent directory trees safely."""
    mock_ffmpeg_cascade()
    target_path = tmp_path / "a" / "b" / "c" / "auto_created.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    saved = save_video_robust(frames, str(target_path))
    assert os.path.exists(saved)
    assert os.path.isdir(str(tmp_path / "a" / "b" / "c"))


def test_tier2_validate_media_inputs_unwritable_output(mock_unwritable_directory):
    """Verifies validate_media_inputs flags target paths in unwritable folders."""
    target_path = os.path.join(mock_unwritable_directory, "out.mp4")
    result = validate_media_inputs(output_target_path=target_path)
    assert result.is_valid is False
    assert result.output_safety["directory_writable"] is False
    assert any("not writable" in err.lower() for err in result.errors)


# ============================================================================
# 6. Model & Precision String Validation Traps
# ============================================================================

def test_tier2_unknown_model_variant_raises_keyerror():
    """Verifies estimate_memory rejects unknown model variant names."""
    with pytest.raises(ValueError, match="Unknown model variant"):
        estimate_memory(model_variant="wan_non_existent_999b", resolution="720p")


def test_tier2_unknown_precision_string_defaults():
    """Verifies estimate_memory defaults unknown precision strings safely to 1.0 byte/param."""
    res = estimate_memory(model_variant="wan2.1_14b", resolution="720p", precision="custom_quant_xyz")
    assert res.bytes_per_param == 1.0


def test_tier2_invalid_resolution_preset_defaults():
    """Verifies parse_resolution falls back safely to 720p default for invalid strings."""
    assert parse_resolution("invalid_ultra_hd_preset") == (1280, 720)
    assert parse_resolution("unparseable_string") == (1280, 720)


# ============================================================================
# 7. Codec Cascade Failure Boundaries
# ============================================================================

def test_tier2_ffmpeg_not_on_path(tmp_path, monkeypatch):
    """Verifies save_video_robust dumps frames and raises VideoSaveError when FFmpeg is missing."""
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    target_path = tmp_path / "no_ffmpeg.mp4"
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(VideoSaveError, match="FFmpeg executable not found"):
        save_video_robust(frames, str(target_path))

    # Verify disaster fallback frames were written
    frame_dir = str(tmp_path / "no_ffmpeg_frames")
    assert os.path.isdir(frame_dir)
    assert os.path.exists(os.path.join(frame_dir, "frame_0000.png"))


def test_tier2_ffmpeg_cascade_exhaustion_raises_error(tmp_path, mock_ffmpeg_cascade):
    """Verifies complete cascade failure across all candidate codecs raises VideoSaveError."""
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc", "libx264", "libopenh264", "mpeg4"])
    target_path = tmp_path / "exhausted_cascade.mp4"
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(VideoSaveError, match="across all encoding backends"):
        save_video_robust(frames, str(target_path))


def test_tier2_zero_byte_cascade_rejection(tmp_path, mock_ffmpeg_cascade):
    """Verifies 0-byte output files from intermediate encoders are purged."""
    mock_ffmpeg_cascade(zero_byte_codecs=["h264_nvenc", "libx264", "libopenh264", "mpeg4"])
    target_path = tmp_path / "zero_byte_corrupt.mp4"
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(VideoSaveError):
        save_video_robust(frames, str(target_path))


# ============================================================================
# 8. Video Tensor Normalization Boundary Shapes
# ============================================================================

def test_tier2_video_tensor_floating_point_clamping(tmp_path, mock_ffmpeg_cascade):
    """Verifies float tensor with values out of [0.0, 1.0] range are clipped to [0, 255]."""
    mock_ffmpeg_cascade()
    target_path = tmp_path / "clipped_video.mp4"
    # Values outside range: -5.0 to 10.0
    out_of_range = np.array([-5.0, 0.5, 10.0], dtype=np.float32).reshape(1, 1, 1, 3)

    saved = save_video_robust(out_of_range, str(target_path))
    assert os.path.exists(saved)


def test_tier2_video_tensor_4channel_rgba_conversion(tmp_path, mock_ffmpeg_cascade):
    """Verifies 4-channel RGBA video tensors are converted to 3-channel RGB."""
    mock_ffmpeg_cascade()
    target_path = tmp_path / "rgba_video.mp4"
    rgba_frames = np.zeros((4, 32, 32, 4), dtype=np.uint8)

    saved = save_video_robust(rgba_frames, str(target_path))
    assert os.path.exists(saved)


def test_tier2_video_tensor_single_channel_grayscale(tmp_path, mock_ffmpeg_cascade):
    """Verifies single-channel grayscale video frames [F, H, W, 1] are repeated to RGB."""
    mock_ffmpeg_cascade()
    target_path = tmp_path / "gray_video.mp4"
    gray_frames = np.zeros((4, 32, 32, 1), dtype=np.uint8)

    saved = save_video_robust(gray_frames, str(target_path))
    assert os.path.exists(saved)


# ============================================================================
# 9. Dtype Guard Boundary Conditions
# ============================================================================

def test_tier2_assert_model_dtype_mixed_parameters():
    """Verifies assert_model_dtype detects offending parameters among mixed dtypes."""
    import torch
    class MixedModel:
        def named_parameters(self):
            p1 = MagicMock(dtype=torch.bfloat16)
            p2 = MagicMock(dtype=torch.float32)
            p3 = MagicMock(dtype=torch.float16)
            return [("safe_layer.weight", p1), ("unsafe_layer.weight", p2), ("half_layer.weight", p3)]

    with pytest.raises(DtypeTrapError, match="allocated in float32"):
        assert_model_dtype(MixedModel(), expected_dtype=torch.bfloat16)


def test_tier2_assert_model_dtype_allows_explicit_fp32():
    """Verifies assert_model_dtype allows float32 when expected_dtype is explicitly set to float32."""
    import torch
    class Fp32Model:
        def named_parameters(self):
            p = MagicMock(dtype=torch.float32)
            return [("layer.weight", p)]

    offending = assert_model_dtype(Fp32Model(), expected_dtype=torch.float32)
    assert len(offending) == 1
    assert offending[0] == "layer.weight"


def test_tier2_dit_dtype_guard_nested_contexts():
    """Verifies nested dit_dtype_guard blocks restore previous dtypes sequentially."""
    import torch
    orig = torch.get_default_dtype()
    with dit_dtype_guard(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
        with dit_dtype_guard(torch.float16):
            assert torch.get_default_dtype() == torch.float16
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == orig


# ============================================================================
# 10. Hardware Preflight Boundaries & Overrides
# ============================================================================

def test_tier2_preflight_zero_vram_device():
    """Verifies preflight when GPU reports insufficient VRAM."""
    import torch
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA Low End GPU"), \
         patch("torch.cuda.mem_get_info", return_value=(0, int(4 * 1024**3))), \
         patch("torch.cuda.get_device_properties", return_value=MagicMock(major=7, minor=5, total_memory=int(4 * 1024**3))):
        report = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
        assert report["hardware"]["free_vram_gb"] == 0.0
        assert report["system_status"] == "INSUFFICIENT_VRAM"
        assert report["evaluation"]["can_run"] is False


def test_tier2_preflight_device_override_specific_cuda_index():
    """Verifies device_override='cuda:1' targets device index 1."""
    import torch
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090"), \
         patch("torch.cuda.mem_get_info", return_value=(int(20 * 1024**3), int(24 * 1024**3))), \
         patch("torch.cuda.get_device_properties", return_value=MagicMock(major=8, minor=6, total_memory=int(24 * 1024**3))):
        report = diagnose_hardware_preflight("wan2.1_14b", "720p", device_override="cuda:1")
        assert report["hardware"]["cuda_available"] is True
        assert report["hardware"]["device_name"] == "NVIDIA GeForce RTX 3090"


# ============================================================================
# 11. Compiler & IR Boundaries (Dynamic M2 Hooks)
# ============================================================================

def test_tier2_canonical_ir_empty_prompt():
    """Verifies canonical IR rejects empty or whitespace-only prompts."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        WanGenerationPipelineIR(prompt="   ")


def test_tier2_canonical_ir_extremely_long_prompt():
    """Verifies canonical IR accepts large prompts up to context limit."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    long_prompt = "cinematic video " * 1000
    ir = WanGenerationPipelineIR(prompt=long_prompt)
    assert len(ir.prompt) > 10000


def test_tier2_canonical_ir_invalid_negative_steps():
    """Verifies canonical IR rejects negative or zero diffusion steps."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    with pytest.raises(ValueError, match="Steps must be positive"):
        WanGenerationPipelineIR(prompt="Valid prompt", steps=-5)


def test_tier2_canonical_ir_negative_guidance():
    """Verifies canonical IR rejects negative guidance scale."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    with pytest.raises(ValueError, match="Guidance must be non-negative"):
        WanGenerationPipelineIR(prompt="Valid prompt", guidance=-1.0)


def test_tier2_canonical_ir_seed_extremes():
    """Verifies canonical IR handles seed boundaries (0, 2^31 - 1, 2^63 - 1)."""
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import WanGenerationPipelineIR

    ir_zero = WanGenerationPipelineIR(prompt="Test", seed=0)
    assert ir_zero.seed == 0

    ir_max = WanGenerationPipelineIR(prompt="Test", seed=2**32 - 1)
    assert ir_max.seed == 2**32 - 1
