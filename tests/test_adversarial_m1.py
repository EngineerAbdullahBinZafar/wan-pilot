"""
Adversarial Stress-Testing Suite for Milestone M1 (Core Preflight Diagnostics & Media Optimization).
Empirically stress-tests:
- Memory estimator boundary conditions (extreme resolutions, 1 frame, 1000 frames, token precision, MoE limits).
- Media validator boundary conditions (EXIF tags 1-8, corrupt EXIF, prime dimensions, codec cascade, 0-byte stubs).
- DiT dtype guard boundary conditions (nested contexts, exception safety, state restoration).
- Preflight diagnostics under extreme hardware profiles.
"""

import os
import math
import types
import pytest
import numpy as np
from PIL import Image

import torch

from wan_video_toolkit.diagnostics import (
    estimate_memory,
    MemoryEstimateResult,
    parse_resolution,
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
    diagnose_hardware_preflight,
    inspect_hardware_environment,
    detect_attention_backends,
)


# ============================================================================
# 1. Memory Estimator Boundaries & Token Math Precision
# ============================================================================

class TestMemoryEstimatorAdversarial:
    """Stress tests for analytical memory estimation engine."""

    @pytest.mark.parametrize("res_str,expected_w,expected_h", [
        ("64x64", 64, 64),
        ("4096x2160", 4096, 2160),
        ("3840x1080", 3840, 1080),  # Ultra-wide
        ("1080x3840", 1080, 3840),  # Ultra-tall
        ("17x19", 17, 19),          # Small prime dimensions
        ("1921x1081", 1921, 1081),  # Odd dimensions
        ("720P", 1280, 720),        # Uppercase preset
        ("  480p  ", 832, 480),     # Whitespace padded
        ("1280*720", 1280, 720),    # Star separator
    ])
    def test_extreme_resolutions_and_aspect_ratios(self, res_str, expected_w, expected_h):
        w, h = parse_resolution(res_str)
        assert (w, h) == (expected_w, expected_h)

        res = estimate_memory(
            model_variant="wan2.1_1.3b",
            resolution=res_str,
            num_frames=81,
            precision="bf16",
        )
        assert isinstance(res, MemoryEstimateResult)
        assert res.width == expected_w
        assert res.height == expected_h

        # Assert no negative numbers, no NaNs, no Inf
        assert not math.isnan(res.peak_vram_gb)
        assert not math.isinf(res.peak_vram_gb)
        assert res.peak_vram_gb >= 0.0
        assert res.latent_tokens >= 0
        assert res.activation_and_kv_cache_gb >= 0.0
        assert res.system_ram_gb >= 0.0

        for key, val in res.memory_breakdown_gb.items():
            assert not math.isnan(val), f"NaN found in breakdown[{key}]"
            assert not math.isinf(val), f"Inf found in breakdown[{key}]"
            assert val >= 0.0, f"Negative value in breakdown[{key}]: {val}"

    @pytest.mark.parametrize("num_frames,expected_f_lat", [
        (1, 1),      # Minimum 1 frame: 1 + max(0, 0)//4 = 1
        (2, 1),      # 1 + 1//4 = 1
        (4, 1),      # 1 + 3//4 = 1
        (5, 2),      # 1 + 4//4 = 2
        (81, 21),    # Standard 81 frames: 1 + 80//4 = 21
        (1000, 250), # 1000 frames: 1 + 999//4 = 250
        (0, 1),      # 0 frames edge case: 1 + max(0, -1)//4 = 1
        (-5, 1),     # Negative frames edge case: 1 + max(0, -6)//4 = 1
    ])
    def test_extreme_frame_counts_boundary(self, num_frames, expected_f_lat):
        res = estimate_memory(
            model_variant="wan2.1_14b",
            resolution="720p",
            num_frames=num_frames,
            precision="fp8",
        )
        # 1280//16 = 80, 720//16 = 45 -> tokens_spatial = 3600
        expected_tokens = expected_f_lat * 3600
        assert res.latent_tokens == expected_tokens
        assert res.latent_tokens >= 0
        assert not math.isnan(res.peak_vram_gb)
        assert res.peak_vram_gb > 0.0

    def test_moe_sequential_vs_concurrent_strict_thresholds(self):
        """
        Critical constraint:
        Confirm concurrent resident strictly exceeds 32GB on fp8,
        while sequential offload stays strictly under 20GB.
        """
        # Sequential offload on fp8
        seq_res = estimate_memory(
            model_variant="wan2.2_moe_14b",
            resolution="720p",
            num_frames=81,
            precision="fp8",
            moe_offload=True,
            t5_cpu=True,
            tiled_vae=True,
        )
        assert seq_res.peak_vram_gb < 20.0, (
            f"Sequential offload peak VRAM ({seq_res.peak_vram_gb} GB) must be strictly under 20GB!"
        )
        assert seq_res.fits_24gb_gpu is True

        # Concurrent resident on fp8
        concurrent_res = estimate_memory(
            model_variant="wan2.2_moe_14b",
            resolution="720p",
            num_frames=81,
            precision="fp8",
            moe_offload=False,
            t5_cpu=True,
            tiled_vae=True,
        )
        assert concurrent_res.peak_vram_gb > 32.0, (
            f"Concurrent resident peak VRAM ({concurrent_res.peak_vram_gb} GB) must strictly exceed 32GB on fp8!"
        )
        assert concurrent_res.fits_24gb_gpu is False
        assert "OOM" in concurrent_res.oom_risk_on_24gb

        # Without any offload at all (moe_offload=False, t5_cpu=False)
        assert seq_res.peak_vram_without_moe_offload_gb > 32.0, (
            f"peak_vram_without_moe_offload_gb ({seq_res.peak_vram_without_moe_offload_gb} GB) must strictly exceed 32GB!"
        )

        # In bf16, concurrent must exceed 54.0 GB
        bf16_res = estimate_memory(
            model_variant="wan2.2_moe_14b",
            resolution="720p",
            num_frames=81,
            precision="bf16",
            moe_offload=True,
        )
        assert bf16_res.peak_vram_without_moe_offload_gb > 54.0, (
            f"bf16 concurrent ({bf16_res.peak_vram_without_moe_offload_gb} GB) must strictly exceed 54GB!"
        )

    @pytest.mark.parametrize("precision,expected_bp", [
        ("fp32", 4.0),
        ("float32", 4.0),
        ("bf16", 2.0),
        ("bfloat16", 2.0),
        ("fp16", 2.0),
        ("float16", 2.0),
        ("fp8", 1.0),
        ("int8", 1.0),
        ("int4", 0.55),
    ])
    def test_precision_scaling_monotonicity(self, precision, expected_bp):
        res = estimate_memory(
            model_variant="wan2.1_14b",
            resolution="720p",
            precision=precision,
        )
        assert res.bytes_per_param == expected_bp
        assert res.active_dit_weights_gb == round((res.active_dit_params * expected_bp) / (1024 ** 3), 2)

    def test_unknown_model_variant_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            estimate_memory(model_variant="non_existent_wan_99b")
        assert "Unknown model variant" in str(exc_info.value)

    def test_parse_resolution_malformed_inputs_fallback(self):
        # Malformed strings should fall back cleanly to 720p without crashing
        assert parse_resolution("gibberish") == (1280, 720)
        assert parse_resolution("x") == (1280, 720)
        assert parse_resolution("1280x") == (1280, 720)
        assert parse_resolution("x720") == (1280, 720)
        assert parse_resolution("") == (1280, 720)


# ============================================================================
# 2. Media Validator Boundaries & Upstream Hotspot Stress Tests
# ============================================================================

class TestMediaValidatorAdversarial:
    """Stress tests for media input validation, EXIF transposition, and video saving."""

    @pytest.mark.parametrize("tag,expects_swap", [
        (1, False),  # 0 deg normal
        (2, False),  # Horizontal flip
        (3, False),  # 180 deg
        (4, False),  # Vertical flip
        (5, True),   # Transpose (flip + 270 deg) -> (W, H) becomes (H, W)
        (6, True),   # 90 deg CW -> (W, H) becomes (H, W)
        (7, True),   # Transverse (flip + 90 deg) -> (W, H) becomes (H, W)
        (8, True),   # 270 deg CW -> (W, H) becomes (H, W)
    ])
    def test_exif_orientation_all_eight_tags(self, tmp_path, tag, expects_swap):
        """
        Verify all 8 EXIF orientation tags from mobile/mirrorless cameras (Wan2.2 #415).
        Asymmetric dimensions (120x80) strictly test whether dimensions are transposed.
        """
        orig_w, orig_h = 120, 80
        img_path = str(tmp_path / f"test_exif_tag_{tag}.jpg")
        img = Image.new("RGB", (orig_w, orig_h), color=(tag * 20, 100, 150))
        exif = img.getexif()
        exif[0x0112] = tag
        img.save(img_path, exif=exif)

        # Test apply_exif_transposition directly
        with Image.open(img_path) as raw_img:
            transposed = apply_exif_transposition(raw_img)
            expected_dim = (orig_h, orig_w) if expects_swap else (orig_w, orig_h)
            assert transposed.size == expected_dim, (
                f"Tag {tag}: expected dimensions {expected_dim}, got {transposed.size}"
            )

        # Test validate_media_inputs integration
        report = validate_media_inputs(image_path=img_path)
        assert report.is_valid is True
        assert report.image_diagnostics is not None
        assert report.image_diagnostics["exif_orientation_tag"] == tag

        if tag in (2, 3, 4, 5, 6, 7, 8):
            assert report.image_diagnostics["exif_bug_mitigated"] is True
            assert report.image_diagnostics["status"] == "AUTO_TRANSPOSED"
        else:
            assert report.image_diagnostics["exif_bug_mitigated"] is False
            assert report.image_diagnostics["status"] == "CLEAN"

    def test_corrupt_exif_and_missing_exif_graceful_handling(self, tmp_path):
        # 1. Clean image with no EXIF
        clean_path = str(tmp_path / "no_exif.png")
        clean_img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        clean_img.save(clean_path)

        with Image.open(clean_path) as img:
            transposed = apply_exif_transposition(img)
            assert transposed.size == (100, 100)

        report = validate_media_inputs(image_path=clean_path)
        assert report.is_valid is True
        assert report.image_diagnostics["exif_orientation_tag"] == 1
        assert report.image_diagnostics["status"] == "CLEAN"

        # 2. Corrupt EXIF bytes
        corrupt_exif_path = str(tmp_path / "corrupt_exif.jpg")
        # Write minimal JPEG header with junk EXIF segment
        with open(corrupt_exif_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe1\x00\x08Exif\x00\x00\xff\xff\xff\xd9")

        # Must not raise unhandled exception in apply_exif_transposition
        report_corrupt = validate_media_inputs(image_path=corrupt_exif_path)
        # Should record error or handle gracefully
        assert isinstance(report_corrupt, MediaValidationResult)

    @pytest.mark.parametrize("input_w,input_h,expected_w,expected_h,was_modified", [
        (17, 19, 16, 16, True),          # Prime numbers
        (1279, 719, 1264, 704, True),    # Just below 720p
        (1, 1, 16, 16, True),            # Minimum clamp
        (3, 5, 16, 16, True),            # Sub-16 dimensions
        (31, 31, 16, 16, True),          # Just below 32
        (32, 32, 32, 32, False),         # Exactly divisible by 16
        (1280, 720, 1280, 720, False),   # Standard 720p divisible by 16
        (1920, 1080, 1920, 1072, True),  # 1080 is not divisible by 16 (1080 // 16 = 67 -> 1072)
    ])
    def test_dimensions_rounding_to_multiples_of_16(self, input_w, input_h, expected_w, expected_h, was_modified):
        norm_w, norm_h, modified = validate_and_normalize_dimensions(input_w, input_h, multiple=16)
        assert norm_w == expected_w
        assert norm_h == expected_h
        assert modified is was_modified
        assert norm_w % 16 == 0
        assert norm_h % 16 == 0
        assert norm_w >= 16
        assert norm_h >= 16

    def test_load_conditioning_image_prime_dimensions(self, tmp_path):
        img_path = str(tmp_path / "prime_image.png")
        img = Image.new("RGB", (17, 19), color=(50, 100, 150))
        img.save(img_path)

        conditioned = load_conditioning_image(img_path, enforce_divisible_16=True)
        assert conditioned.size == (16, 16)
        assert conditioned.mode == "RGB"

        # Without enforcement
        unmodified = load_conditioning_image(img_path, enforce_divisible_16=False)
        assert unmodified.size == (17, 19)

    def test_codec_cascade_failure_dumps_valid_png_frames(self, tmp_path, mock_ffmpeg_cascade):
        """
        Wan2.2 #416 disaster fallback:
        When all video codecs fail, individual PNG frames must be dumped
        and VideoSaveError strictly raised with frame directory information.
        """
        mock_ffmpeg_cascade(failing_codecs=["h264_nvenc", "libx264", "libopenh264", "mpeg4"])

        num_test_frames = 6
        video_tensor = np.ones((num_test_frames, 32, 32, 3), dtype=np.uint8) * 128
        out_file = str(tmp_path / "cascade_fail_video.mp4")

        with pytest.raises(VideoSaveError) as exc_info:
            save_video_robust(video_tensor, out_file)

        error_msg = str(exc_info.value)
        assert "Failed to save video" in error_msg
        assert "fallback" in error_msg.lower()

        # Check that PNG frames were dumped and are valid
        expected_frame_dir = str(tmp_path / "cascade_fail_video_frames")
        assert os.path.isdir(expected_frame_dir)
        dumped_frames = sorted(os.listdir(expected_frame_dir))
        assert len(dumped_frames) == num_test_frames

        for idx, frame_name in enumerate(dumped_frames):
            frame_path = os.path.join(expected_frame_dir, frame_name)
            assert frame_name == f"frame_{idx:04d}.png"
            with Image.open(frame_path) as frame_img:
                assert frame_img.size == (32, 32)
                assert frame_img.mode == "RGB"

    def test_zero_byte_output_strictly_raises_videosaveerror(self, tmp_path, mock_ffmpeg_cascade):
        """Wan2.2 #416: Zero-byte files or files <= 1024 bytes must never pass silently."""
        mock_ffmpeg_cascade(make_empty_file=True)

        video_tensor = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        out_file = str(tmp_path / "silent_corruption_test.mp4")

        with pytest.raises(VideoSaveError):
            save_video_robust(video_tensor, out_file)

    def test_save_video_input_format_normalization(self, tmp_path, mock_ffmpeg_cascade):
        mock_ffmpeg_cascade(failing_codecs=[])

        out_file = str(tmp_path / "format_norm_test.mp4")

        # 1. Float tensor [0.0, 1.0]
        float_tensor = np.random.rand(4, 32, 32, 3).astype(np.float32)
        saved = save_video_robust(float_tensor, out_file)
        assert os.path.exists(saved)

        # 2. Channel-first format [F, C, H, W]
        ch_first = np.zeros((4, 3, 32, 32), dtype=np.uint8)
        saved = save_video_robust(ch_first, out_file)
        assert os.path.exists(saved)

        # 3. Single-channel grayscale [F, H, W, 1]
        gray_tensor = np.zeros((4, 32, 32, 1), dtype=np.uint8)
        saved = save_video_robust(gray_tensor, out_file)
        assert os.path.exists(saved)

        # 4. Invalid 3D tensor
        with pytest.raises(ValueError) as exc:
            save_video_robust(np.zeros((32, 32, 3), dtype=np.uint8), out_file)
        assert "Expected 4D video" in str(exc.value)


# ============================================================================
# 3. DiT Dtype Guard Boundaries (Wan-Animate-2 #5)
# ============================================================================

class TestDtypeGuardAdversarial:
    """Stress tests for dit_dtype_guard and assert_model_dtype."""

    def test_nested_dtype_guards_restoration(self):
        orig_dtype = torch.get_default_dtype()

        target_bf16 = getattr(torch, "bfloat16", torch.float32)
        target_fp16 = getattr(torch, "float16", torch.float32)

        with dit_dtype_guard(target_bf16):
            assert torch.get_default_dtype() == target_bf16

            with dit_dtype_guard(target_fp16):
                assert torch.get_default_dtype() == target_fp16

                with dit_dtype_guard(target_bf16):
                    assert torch.get_default_dtype() == target_bf16

                assert torch.get_default_dtype() == target_fp16

            assert torch.get_default_dtype() == target_bf16

        assert torch.get_default_dtype() == orig_dtype

    def test_dtype_guard_exception_safety(self):
        orig_dtype = torch.get_default_dtype()
        target_bf16 = getattr(torch, "bfloat16", torch.float32)

        with pytest.raises(RuntimeError) as exc_info:
            with dit_dtype_guard(target_bf16):
                assert torch.get_default_dtype() == target_bf16
                raise RuntimeError("Simulated DiT weight initialization failure!")

        assert "Simulated DiT weight" in str(exc_info.value)
        # Invariant: default dtype must be restored despite the exception
        assert torch.get_default_dtype() == orig_dtype

    def test_assert_model_dtype_identifies_single_trapped_layer(self):
        class MockParam:
            def __init__(self, dt):
                self.dtype = dt

        class MockDiTModel:
            def __init__(self, params_dict):
                self._params = params_dict

            def named_parameters(self):
                return self._params.items()

        # 99 bfloat16 parameters, 1 float32 parameter trap
        params = {f"transformer_blocks.{i}.attn.q_proj.weight": MockParam(torch.bfloat16) for i in range(99)}
        params["transformer_blocks.99.ffn.w3.weight"] = MockParam(torch.float32)

        bad_model = MockDiTModel(params)

        with pytest.raises(DtypeTrapError) as exc_info:
            assert_model_dtype(bad_model, expected_dtype=torch.bfloat16)

        assert "Dtype Trap Detected: 1 parameters allocated in float32" in str(exc_info.value)

        # Clean model with all bfloat16 parameters
        params["transformer_blocks.99.ffn.w3.weight"] = MockParam(torch.bfloat16)
        clean_model = MockDiTModel(params)
        offending = assert_model_dtype(clean_model, expected_dtype=torch.bfloat16)
        assert len(offending) == 0


# ============================================================================
# 4. Preflight Diagnostics Adversarial Hardware Scenarios
# ============================================================================

class TestPreflightDiagnosticsAdversarial:
    """Stress tests for preflight hardware auditing and trap detection."""

    def test_preflight_extreme_low_vram_and_traps(self, monkeypatch):
        # Mock a GPU with only 4GB VRAM
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce GTX 1650")
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
            int(3.2 * 1024**3),
            int(4.0 * 1024**3),
        ))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(0.8 * 1024**3))

        report = diagnose_hardware_preflight(
            model_variant="wan2.2_moe_14b",
            resolution="720p",
        )

        assert report["system_status"] == "INSUFFICIENT_VRAM"
        assert report["evaluation"]["can_run"] is False
        assert report["evaluation"]["headroom_gb"] < 0.0

        # Traps must be flagged
        trap_ids = [t["trap_id"] for t in report["flagged_traps"]]
        assert "OOM_TRAP_MOE_CONCURRENT" in trap_ids
        assert "OOM_TRAP_T5_VRAM" in trap_ids
        assert "OOM_TRAP_DIT_FP32" in trap_ids
        assert "OOM_TRAP_VAE_DEC_SPIKE" in trap_ids

        # Flags must enforce max offloading
        flags = report["recommended_flags"]
        assert "--precision" in flags and "fp8" in flags
        assert "--offload_model" in flags and "True" in flags
        assert "--t5_cpu" in flags and "True" in flags

    def test_preflight_datacenter_gpu_80gb(self, monkeypatch):
        # Mock an NVIDIA A100 (80GB VRAM, 78GB free)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA A100-SXM4-80GB")
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
            int(78.0 * 1024**3),
            int(80.0 * 1024**3),
        ))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(2.0 * 1024**3))

        report = diagnose_hardware_preflight(
            model_variant="wan2.2_moe_14b",
            resolution="1080p",
        )

        assert report["system_status"] == "READY"
        assert report["evaluation"]["can_run"] is True
        # At 1080p bf16 with t5_cpu=False, model+activation+T5 takes ~51.92 GB, leaving ~26.08 GB headroom
        assert report["evaluation"]["headroom_gb"] > 20.0
        # High VRAM recommends bf16
        flags = report["recommended_flags"]
        assert "--precision" in flags and "bf16" in flags
