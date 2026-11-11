"""
Unit tests for EXIF orientation auto-transposition (Wan2.2 #415),
resilient video save fallbacks & validation (Wan2.2 #416),
and DiT dtype guard (Wan-Animate-2 #5).
"""

import os
import pytest
import numpy as np
from PIL import Image

from wan_video_toolkit.diagnostics import (
    apply_exif_transposition,
    load_conditioning_image,
    validate_and_normalize_dimensions,
    save_video_robust,
    VideoSaveError,
    dit_dtype_guard,
    assert_model_dtype,
    DtypeTrapError,
    validate_media_inputs,
    MediaValidationResult,
)


def test_exif_orientation_transposition_resolves_issue_415(synthetic_exif_image):
    # Raw image has size (100, 200) with EXIF orientation tag 6 (Rotate 90 CW)
    with Image.open(synthetic_exif_image) as raw_img:
        assert raw_img.size == (100, 200)
        exif = raw_img.getexif()
        assert exif.get(0x0112) == 6

    # Test apply_exif_transposition directly
    with Image.open(synthetic_exif_image) as raw_img:
        transposed = apply_exif_transposition(raw_img)
        assert transposed.size == (200, 100)

    # Test load_conditioning_image (with 16x normalization)
    # 200 -> 192 (nearest multiple <= 200), 100 -> 96
    conditioned = load_conditioning_image(synthetic_exif_image, enforce_divisible_16=True)
    assert conditioned.size == (192, 96)
    assert conditioned.mode == "RGB"


def test_validate_and_normalize_dimensions_16x():
    w, h, mod = validate_and_normalize_dimensions(100, 200, multiple=16)
    assert w == 96
    assert h == 192
    assert mod is True

    w, h, mod = validate_and_normalize_dimensions(1280, 720, multiple=16)
    assert w == 1280
    assert h == 720
    assert mod is False


def test_save_video_robust_success(tmp_path, mock_ffmpeg_cascade):
    # Configure mock ffmpeg to succeed on default codec
    mock_ffmpeg_cascade(failing_codecs=[])

    video_tensor = np.zeros((16, 64, 64, 3), dtype=np.uint8)
    out_path = str(tmp_path / "output_test.mp4")

    saved_path = save_video_robust(video_tensor, out_path, fps=16)
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


def test_save_video_robust_fallback_nvenc_to_libx264(tmp_path, mock_ffmpeg_cascade):
    # Simulate h264_nvenc failure, causing fallback to libx264
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc"])

    video_tensor = np.zeros((16, 64, 64, 3), dtype=np.uint8)
    out_path = str(tmp_path / "fallback_test.mp4")

    saved_path = save_video_robust(video_tensor, out_path, preferred_codec="h264_nvenc")
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


def test_save_video_robust_total_failure_dumps_frames(tmp_path, mock_ffmpeg_cascade):
    # Simulate all codecs failing: h264_nvenc, libx264, libopenh264, mpeg4
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc", "libx264", "libopenh264", "mpeg4"])

    video_tensor = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    out_path = str(tmp_path / "failed_video.mp4")

    with pytest.raises(VideoSaveError) as exc_info:
        save_video_robust(video_tensor, out_path)

    assert "Failed to save video" in str(exc_info.value)
    # Check that fallback PNG frames were dumped to directory
    frame_dir = str(tmp_path / "failed_video_frames")
    assert os.path.isdir(frame_dir)
    assert os.path.exists(os.path.join(frame_dir, "frame_0000.png"))
    assert os.path.exists(os.path.join(frame_dir, "frame_0004.png"))


def test_save_video_robust_zero_byte_detection(tmp_path, mock_ffmpeg_cascade):
    # Simulate an encoder producing a 0-byte corrupt stub
    mock_ffmpeg_cascade(make_empty_file=True)

    video_tensor = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    out_path = str(tmp_path / "zero_byte.mp4")

    with pytest.raises(VideoSaveError):
        save_video_robust(video_tensor, out_path)


def test_dit_dtype_guard_resolves_issue_5():
    import sys
    import types

    # Ensure fake or real torch module
    torch_mod = sys.modules.get("torch")
    created_fake = False
    if torch_mod is None:
        torch_mod = types.ModuleType("torch")
        torch_mod.float32 = "float32"
        torch_mod.bfloat16 = "bfloat16"
        torch_mod.float16 = "float16"
        torch_mod._default_dtype = torch_mod.float32
        torch_mod.get_default_dtype = lambda: torch_mod._default_dtype
        def set_def(dt):
            torch_mod._default_dtype = dt
        torch_mod.set_default_dtype = set_def
        sys.modules["torch"] = torch_mod
        created_fake = True

    orig_dtype = torch_mod.get_default_dtype()

    # Guard sets bfloat16 and restores afterwards
    with dit_dtype_guard(torch_mod.bfloat16):
        assert torch_mod.get_default_dtype() == torch_mod.bfloat16

    assert torch_mod.get_default_dtype() == orig_dtype

    if created_fake:
        sys.modules.pop("torch", None)


def test_assert_model_dtype_detects_trap():
    import types

    class FakeParam:
        def __init__(self, dtype):
            self.dtype = dtype

    class FakeModel:
        def __init__(self, params):
            self._params = params

        def named_parameters(self):
            return self._params.items()

    # Ensure fake torch
    import sys
    torch_mod = sys.modules.get("torch")
    created_fake = False
    if torch_mod is None:
        torch_mod = types.ModuleType("torch")
        torch_mod.float32 = "float32"
        torch_mod.bfloat16 = "bfloat16"
        sys.modules["torch"] = torch_mod
        created_fake = True

    # Offending model with float32 parameters
    bad_model = FakeModel({
        "layers.0.weight": FakeParam(torch_mod.float32),
        "layers.0.bias": FakeParam(torch_mod.bfloat16),
    })

    with pytest.raises(DtypeTrapError):
        assert_model_dtype(bad_model, expected_dtype=torch_mod.bfloat16)

    # Compliant model with bfloat16 parameters
    good_model = FakeModel({
        "layers.0.weight": FakeParam(torch_mod.bfloat16),
        "layers.0.bias": FakeParam(torch_mod.bfloat16),
    })
    offending = assert_model_dtype(good_model, expected_dtype=torch_mod.bfloat16)
    assert len(offending) == 0

    if created_fake:
        sys.modules.pop("torch", None)


def test_validate_media_inputs_end_to_end(synthetic_exif_image, tmp_path):
    target_video = str(tmp_path / "valid_output.mp4")

    result = validate_media_inputs(
        image_path=synthetic_exif_image,
        output_target_path=target_video,
    )

    assert isinstance(result, MediaValidationResult)
    assert result.is_valid is True
    assert result.image_diagnostics is not None
    assert result.image_diagnostics["exif_bug_mitigated"] is True
    assert result.image_diagnostics["status"] == "AUTO_TRANSPOSED"
    assert result.output_safety is not None
    assert result.output_safety["directory_writable"] is True
    assert len(result.warnings) > 0
    assert len(result.errors) == 0
