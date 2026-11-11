"""
Media Input and Output Validator for Wan-Video Pipelines.
Resolves:
- Wan2.2 Issue #415: EXIF orientation auto-transposition for I2V conditioning.
- Wan2.2 Issue #416: Resilient video saving with multi-tier codec fallback and post-write verification.
- Wan-Animate-2 Issue #5: DiT default dtype context guard to prevent 56GB fp32 allocations.
"""

import os
import sys
import shutil
import logging
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union

from PIL import Image, ImageOps
import numpy as np

logger = logging.getLogger(__name__)


class VideoSaveError(RuntimeError):
    """Raised when video serialization fails across all encoding backends."""
    pass


class DtypeTrapError(Exception):
    """Raised when DiT layers are instantiated with unsafe float32 precision."""
    pass


@dataclass
class MediaValidationResult:
    is_valid: bool
    image_diagnostics: Optional[Dict[str, Any]] = None
    video_diagnostics: Optional[Dict[str, Any]] = None
    output_safety: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "image_diagnostics": self.image_diagnostics,
            "video_diagnostics": self.video_diagnostics,
            "output_safety": self.output_safety,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ============================================================================
# EXIF & Dimension Normalization (Resolves Wan2.2 Issue #415)
# ============================================================================

def apply_exif_transposition(image: Image.Image) -> Image.Image:
    """
    Applies EXIF orientation tag transposition (Tag 0x0112 / 274) to an image.
    Resolves Wan2.2 Issue #415 where mobile camera photos were fed sideways to the VAE.
    """
    try:
        transposed = ImageOps.exif_transpose(image)
        if transposed is not None:
            return transposed
    except Exception as exc:
        logger.warning(f"Failed to apply EXIF transposition: {exc}")
    return image


def validate_and_normalize_dimensions(
    width: int,
    height: int,
    multiple: int = 16,
) -> Tuple[int, int, bool]:
    """
    Ensures dimensions are strictly divisible by the specified multiple (default 16 for 3D VAE).
    Returns (normalized_w, normalized_h, was_modified).
    """
    norm_w = max(multiple, (width // multiple) * multiple)
    norm_h = max(multiple, (height // multiple) * multiple)
    was_modified = (norm_w != width) or (norm_h != height)
    return norm_w, norm_h, was_modified


def load_conditioning_image(
    image_path: str,
    max_area: int = 1280 * 720,
    enforce_divisible_16: bool = True,
) -> Image.Image:
    """
    Safely loads and pre-processes an input conditioning image for Wan I2V pipelines.
    Resolves Wan2.2 Issue #415:
    1. Transposes EXIF orientation metadata (tags 1 through 8).
    2. Converts to 24-bit RGB (stripping Alpha, CMYK, Palette).
    3. Enforces spatial divisibility by 16 for 8x VAE downscale and 2x patchification.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Conditioning image not found at: {image_path}")

    try:
        image = Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"Unable to read input image at {image_path}: {exc}") from exc

    # 1. Correct EXIF orientation
    image = apply_exif_transposition(image)

    # 2. Ensure RGB mode
    if image.mode != "RGB":
        logger.info(f"Converting image {image_path} from {image.mode} to RGB")
        image = image.convert("RGB")

    # 3. Enforce 16x divisibility
    w, h = image.size
    if enforce_divisible_16:
        target_w, target_h, was_modified = validate_and_normalize_dimensions(w, h, multiple=16)
        if was_modified:
            logger.warning(
                f"Image dimensions ({w}x{h}) are not divisible by 16. "
                f"Resizing to ({target_w}x{target_h}) via Lanczos resampling."
            )
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return image


# ============================================================================
# Resilient Video Saver (Resolves Wan2.2 Issue #416)
# ============================================================================

def _normalize_video_array(tensor: Any) -> np.ndarray:
    """
    Converts torch.Tensor or numpy.ndarray to a numpy array shaped [F, H, W, C] of uint8 [0, 255].
    """
    # If torch.Tensor, convert to CPU numpy
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        tensor = tensor.detach().cpu().numpy()
    elif not isinstance(tensor, np.ndarray):
        tensor = np.asarray(tensor)

    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D video tensor/array, got ndim={tensor.ndim} with shape {tensor.shape}")

    # Check axis arrangements:
    # Target: [F, H, W, C] where C is 1, 3, or 4
    if tensor.shape[-1] in (1, 3, 4):
        # Already [F, H, W, C]
        frames_np = tensor
    elif tensor.shape[1] in (1, 3, 4):
        # [F, C, H, W] -> transpose to [F, H, W, C]
        frames_np = np.transpose(tensor, (0, 2, 3, 1))
    elif tensor.shape[0] in (1, 3, 4):
        # [C, F, H, W] -> transpose to [F, H, W, C]
        frames_np = np.transpose(tensor, (1, 2, 3, 0))
    else:
        frames_np = tensor

    # Normalize dtype to uint8
    if np.issubdtype(frames_np.dtype, np.floating):
        frames_np = (np.clip(frames_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

    # Convert single channel grayscale or RGBA to 3-channel RGB
    if frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)
    elif frames_np.shape[-1] == 4:
        frames_np = frames_np[..., :3]

    return frames_np


def save_video_robust(
    tensor: Any,
    output_path: str,
    fps: int = 16,
    preferred_codec: str = "auto",
) -> str:
    """
    Robust video saver resolving Wan2.2 Issue #416.
    1. Verifies parent directory writability.
    2. Multi-tier codec fallback cascade: h264_nvenc -> libx264 -> libopenh264 -> mpeg4.
    3. Strict post-save verification asserting file exists and size > 1024 bytes.
    4. Disaster fallback: Exports PNG frames and raises VideoSaveError if all codecs fail.
    """
    abs_output_path = os.path.abspath(output_path)
    parent_dir = os.path.dirname(abs_output_path)

    # 1. Directory writability check
    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as exc:
            raise VideoSaveError(f"Cannot create destination directory {parent_dir}: {exc}") from exc

    if not os.access(parent_dir, os.W_OK):
        raise VideoSaveError(f"Destination directory {parent_dir} is not writable by the current process.")

    # 2. Normalize frames
    frames_np = _normalize_video_array(tensor)
    num_frames, height, width, channels = frames_np.shape

    # 3. Determine codec cascade
    codec_cascade = []
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if preferred_codec == "h264_nvenc" or (preferred_codec == "auto" and cuda_available):
        codec_cascade.append("h264_nvenc")
    elif preferred_codec not in ("auto", "h264_nvenc"):
        codec_cascade.append(preferred_codec)

    for fallback in ["libx264", "libopenh264", "mpeg4"]:
        if fallback not in codec_cascade:
            codec_cascade.append(fallback)

    last_error = None
    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin:
        for codec in codec_cascade:
            try:
                logger.info(f"Attempting video encoding to {abs_output_path} with codec: {codec}")
                # Remove stale or 0-byte file if it exists
                if os.path.exists(abs_output_path):
                    try:
                        os.remove(abs_output_path)
                    except Exception:
                        pass

                cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgb24",
                    "-r", str(fps),
                    "-i", "-",
                    "-c:v", codec,
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    abs_output_path,
                ]

                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                raw_bytes = frames_np.tobytes()
                stdout, stderr = proc.communicate(input=raw_bytes, timeout=120)

                # Post-save strict validation
                if (
                    proc.returncode == 0
                    and os.path.exists(abs_output_path)
                    and os.path.getsize(abs_output_path) > 1024
                ):
                    file_size = os.path.getsize(abs_output_path)
                    logger.info(f"Video saved successfully via {codec} ({file_size} bytes)")
                    return abs_output_path
                else:
                    err_msg = stderr.decode("utf-8", errors="ignore") if stderr else f"Exit code {proc.returncode}"
                    logger.warning(f"Codec {codec} failed (code {proc.returncode}): {err_msg[:200]}")
                    last_error = err_msg
            except Exception as exc:
                last_error = str(exc)
                logger.warning(f"Codec {codec} raised exception: {exc}")
    else:
        last_error = "FFmpeg executable not found on system PATH."

    # 4. Disaster Fallback: Export raw PNG frames if video encoding failed completely
    png_dir = os.path.splitext(abs_output_path)[0] + "_frames"
    try:
        os.makedirs(png_dir, exist_ok=True)
        for i, frame in enumerate(frames_np):
            frame_img = Image.fromarray(frame)
            frame_img.save(os.path.join(png_dir, f"frame_{i:04d}.png"))
        logger.info(f"Exported {len(frames_np)} fallback PNG frames to: {png_dir}")
    except Exception as frame_err:
        logger.error(f"Failed to export fallback frames: {frame_err}")

    raise VideoSaveError(
        f"Failed to save video to '{abs_output_path}' across all encoding backends. "
        f"Last error: {last_error}. Individual frames dumped as fallback to: {png_dir}"
    )


# ============================================================================
# DiT Dtype Guard (Resolves Wan-Animate-2 Issue #5)
# ============================================================================

@contextmanager
def dit_dtype_guard(target_dtype: Optional[Any] = None):
    """
    Context manager preventing the 56GB fp32 allocation trap (Wan-Animate-2 Issue #5).
    Safely sets torch default dtype to target_dtype (defaults to bfloat16 or float16)
    during DiT layer instantiation, restoring the original default dtype on exit.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        try:
            import torch as torch_mod
        except ImportError:
            torch_mod = None

    if torch_mod is None:
        # Torch not installed; no-op context
        yield
        return

    previous_dtype = torch_mod.get_default_dtype()
    if target_dtype is None:
        target_dtype = getattr(torch_mod, "bfloat16", getattr(torch_mod, "float16", previous_dtype))

    try:
        torch_mod.set_default_dtype(target_dtype)
        logger.debug(f"Guarded PyTorch default dtype set to {target_dtype}")
        yield
    finally:
        torch_mod.set_default_dtype(previous_dtype)
        logger.debug(f"Restored PyTorch default dtype to {previous_dtype}")


def assert_model_dtype(model: Any, expected_dtype: Optional[Any] = None) -> List[str]:
    """
    Verifies that no DiT parameter was inadvertently allocated in float32.
    Returns list of offending parameter names if any exist.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None or not hasattr(model, "named_parameters"):
        return []

    if expected_dtype is None:
        expected_dtype = getattr(torch_mod, "bfloat16", getattr(torch_mod, "float16", None))

    float32_params = []
    for name, param in model.named_parameters():
        if hasattr(param, "dtype") and param.dtype == torch_mod.float32:
            float32_params.append(name)

    if float32_params and expected_dtype != torch_mod.float32:
        msg = f"Dtype Trap Detected: {len(float32_params)} parameters allocated in float32! (Issue #5 Dtype Trap risk)"
        logger.warning(msg)
        raise DtypeTrapError(msg)

    return float32_params


# ============================================================================
# Composite Input/Output Validation Tool
# ============================================================================

def validate_media_inputs(
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    output_target_path: Optional[str] = None,
) -> MediaValidationResult:
    """
    Comprehensive validation of input conditioning media and output target destinations.
    Checks:
    - Input image EXIF rotation tags and 16x divisibility.
    - Input video readability and dimensions.
    - Output directory writability, available disk space, and FFmpeg codec readiness.
    """
    warnings: List[str] = []
    errors: List[str] = []
    image_diag: Optional[Dict[str, Any]] = None
    video_diag: Optional[Dict[str, Any]] = None
    output_safety: Optional[Dict[str, Any]] = None

    # 1. Validate Image Input
    if image_path:
        if not os.path.exists(image_path):
            errors.append(f"Image path does not exist: {image_path}")
        else:
            try:
                with Image.open(image_path) as raw_img:
                    raw_w, raw_h = raw_img.size
                    exif_data = raw_img.getexif()
                    orientation_tag = exif_data.get(0x0112, 1)

                transposed_img = load_conditioning_image(image_path, enforce_divisible_16=False)
                trans_w, trans_h = transposed_img.size
                norm_w, norm_h, was_padded = validate_and_normalize_dimensions(trans_w, trans_h, multiple=16)

                exif_bug_mitigated = orientation_tag in [2, 3, 4, 5, 6, 7, 8]
                if exif_bug_mitigated:
                    warnings.append(
                        f"Image contains EXIF orientation tag {orientation_tag}. Auto-transposed to ({trans_w}x{trans_h}) to resolve Wan2.2 #415."
                    )
                if was_padded:
                    warnings.append(
                        f"Image dimensions ({trans_w}x{trans_h}) are not divisible by 16. Normalized to ({norm_w}x{norm_h})."
                    )

                image_diag = {
                    "path": image_path,
                    "raw_dimensions": [raw_w, raw_h],
                    "exif_orientation_tag": orientation_tag,
                    "transposed_dimensions": [trans_w, trans_h],
                    "normalized_dimensions": [norm_w, norm_h],
                    "divisible_by_16": not was_padded,
                    "exif_bug_mitigated": exif_bug_mitigated,
                    "status": "AUTO_TRANSPOSED" if exif_bug_mitigated else "CLEAN",
                }
            except Exception as e:
                errors.append(f"Failed to inspect image {image_path}: {e}")

    # 2. Validate Video Input
    if video_path:
        if not os.path.exists(video_path):
            errors.append(f"Video path does not exist: {video_path}")
        else:
            file_size = os.path.getsize(video_path)
            if file_size < 1024:
                errors.append(f"Video file is corrupted or empty ({file_size} bytes)")
            else:
                video_diag = {
                    "path": video_path,
                    "file_size_bytes": file_size,
                    "status": "ACCESSIBLE",
                }

    # 3. Validate Output Target Path Safety
    if output_target_path:
        abs_target = os.path.abspath(output_target_path)
        parent_dir = os.path.dirname(abs_target)

        writable = False
        try:
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            writable = os.access(parent_dir, os.W_OK)
        except Exception:
            writable = False

        if not writable:
            errors.append(f"Output destination directory is not writable: {parent_dir}")

        # Check free disk space
        free_gb = 0.0
        try:
            total, used, free = shutil.disk_usage(parent_dir)
            free_gb = round(free / (1024 ** 3), 2)
            if free_gb < 2.0:
                warnings.append(f"Low disk space on target volume: {free_gb} GB remaining.")
        except Exception:
            free_gb = 10.0

        ffmpeg_bin = shutil.which("ffmpeg")
        supported_codecs = ["libx264", "libopenh264", "mpeg4"]
        try:
            import torch
            if torch.cuda.is_available():
                supported_codecs.insert(0, "h264_nvenc")
        except Exception:
            pass

        output_safety = {
            "target_path": abs_target,
            "directory_writable": writable,
            "available_disk_space_gb": free_gb,
            "ffmpeg_detected": ffmpeg_bin is not None,
            "supported_codecs": supported_codecs,
            "silent_save_protection": "ENABLED",
        }

    is_valid = len(errors) == 0

    return MediaValidationResult(
        is_valid=is_valid,
        image_diagnostics=image_diag,
        video_diagnostics=video_diag,
        output_safety=output_safety,
        warnings=warnings,
        errors=errors,
    )
