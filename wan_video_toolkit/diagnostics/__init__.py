"""
Diagnostics module for Wan-Video Toolkit.
Provides hardware preflight inspection, memory estimation, and media validation.
"""

from .preflight import (
    diagnose_hardware_preflight,
    detect_attention_backends,
    inspect_hardware_environment,
)
from .memory_estimator import (
    estimate_memory,
    MemoryEstimateResult,
    parse_resolution,
    WAN_MODEL_SPECS,
    PRECISION_BYTES,
    RESOLUTION_PRESETS,
)
from .media_validator import (
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

__all__ = [
    "diagnose_hardware_preflight",
    "detect_attention_backends",
    "inspect_hardware_environment",
    "estimate_memory",
    "MemoryEstimateResult",
    "parse_resolution",
    "WAN_MODEL_SPECS",
    "PRECISION_BYTES",
    "RESOLUTION_PRESETS",
    "validate_media_inputs",
    "MediaValidationResult",
    "apply_exif_transposition",
    "load_conditioning_image",
    "validate_and_normalize_dimensions",
    "save_video_robust",
    "VideoSaveError",
    "dit_dtype_guard",
    "assert_model_dtype",
    "DtypeTrapError",
]
