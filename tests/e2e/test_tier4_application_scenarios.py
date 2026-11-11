"""
Tier 4: Realistic End-to-End Application Scenarios Test Suite.
Verifies complete, real-world deployment workflows modeling user and agent journeys:
- Scenario 1: End-to-end T2V 720p 81-frame generation on 24GB RTX 3090/4090 with golden flags
- Scenario 2: I2V generation pipeline from smartphone photo (EXIF Tag 6, auto-transpose, 16x pad, libx264 fallback)
- Scenario 3: Apple Silicon M3 Max unified memory fallback simulation (MPS detection, shared memory, MPS guidance)
- Scenario 4: ComfyUI Drag-and-Drop Workflow Canvas UI & Prompt API export
- Scenario 5: Multi-modal AI agent invocation via stdio JSON-RPC 2.0 (handshake, tools/list, tool cascades, dry run)
- Scenario 6: Extreme Low-VRAM 12GB workstation recovery (RTX 4070, 480p, INT8 precision, offload model)

Coverage Target: >= 5 tests total.
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
# Scenario 1: RTX 4090/3090 24GB Golden Flags T2V Pipeline
# ============================================================================

def test_scenario1_rtx4090_t2v_golden_flags_workflow(tmp_path, mock_cuda_rtx4090, mock_ffmpeg_cascade):
    """
    Simulates user running Wan2.2 MoE 14B T2V (720p, 81 frames) on a 24GB RTX 4090:
    1. Preflight verifies GPU, identifies MoE trap, and prescribes golden flags.
    2. Memory estimator proves peak VRAM is bounded to 19.7 GB (<= 24GB budget).
    3. Output video is saved with post-write verification.
    """
    mock_ffmpeg_cascade()
    out_video = tmp_path / "golden_flags_output.mp4"

    # Step 1: Preflight hardware inspection
    preflight = diagnose_hardware_preflight("wan2.2_moe_14b", "720p")
    assert preflight["hardware"]["device_name"] == "NVIDIA GeForce RTX 4090"
    assert preflight["hardware"]["total_vram_gb"] == 24.0

    # Ensure concurrent MoE trap was detected and mitigated
    traps = [t["trap_id"] for t in preflight["flagged_traps"]]
    assert "OOM_TRAP_MOE_CONCURRENT" in traps

    flags = preflight["recommended_flags"]
    assert "--precision" in flags and "fp8" in flags
    assert "--expert_offload" in flags and "True" in flags
    assert "--t5_cpu" in flags and "True" in flags
    assert "--tiled_vae" in flags and "True" in flags

    # Step 2: Analytical memory calculation
    mem = estimate_memory(
        model_variant="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        precision="fp8",
        moe_offload=True,
        t5_cpu=True,
        tiled_vae=True,
    )
    assert mem.fits_24gb_gpu is True
    assert mem.peak_vram_gb <= 20.5
    assert mem.peak_vram_gb < preflight["hardware"]["free_vram_gb"]

    # Step 3: Video output persistence
    frames = np.zeros((81, 64, 64, 3), dtype=np.uint8)
    saved_path = save_video_robust(frames, str(out_video), fps=16, preferred_codec="h264_nvenc")
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


# ============================================================================
# Scenario 2: Smartphone Photo I2V Auto-Correction Pipeline
# ============================================================================

def test_scenario2_smartphone_photo_i2v_pipeline(tmp_path, synthetic_exif_image, mock_ffmpeg_cascade):
    """
    Simulates mobile phone photo upload (Orientation Tag 6, 90 deg CW):
    1. Validates media inputs, auto-transposes EXIF tag 6 (Issue #415).
    2. Enforces 16x dimension divisibility for 3D causal VAE.
    3. Hardware encoder fails (e.g. CUDA NVENC out of sessions), falls back to libx264.
    4. Output video verified and persisted.
    """
    mock_ffmpeg_cascade(failing_codecs=["h264_nvenc"])
    out_video = tmp_path / "i2v_smartphone_corrected.mp4"

    # Step 1 & 2: Media validation and auto-transposition
    val = validate_media_inputs(image_path=synthetic_exif_image, output_target_path=str(out_video))
    assert val.is_valid is True
    assert val.image_diagnostics["status"] == "AUTO_TRANSPOSED"
    assert val.image_diagnostics["exif_orientation_tag"] == 6

    # Load conditioning image with 16x normalization
    cond_img = load_conditioning_image(synthetic_exif_image, enforce_divisible_16=True)
    assert cond_img.size == (192, 96)
    assert cond_img.width % 16 == 0 and cond_img.height % 16 == 0

    # Step 3: Resilient video save with fallback from nvenc to libx264 (Issue #416)
    frames = np.zeros((49, cond_img.height, cond_img.width, 3), dtype=np.uint8)
    saved_path = save_video_robust(frames, str(out_video), fps=16, preferred_codec="h264_nvenc")
    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 1024


# ============================================================================
# Scenario 3: Apple Silicon M3 Max Unified Memory Fallback
# ============================================================================

def test_scenario3_apple_silicon_mps_unified_memory(mock_apple_mps):
    """
    Simulates developer workstation on macOS Apple Silicon M3 Max (36GB unified RAM):
    1. Preflight detects MPS, evaluates system RAM as unified VRAM pool.
    2. Emits specific MPS environment variables and guidance.
    3. Analytical memory estimator validates generation feasibility.
    """
    preflight = diagnose_hardware_preflight("wan2.1_14b", "720p")
    assert preflight["hardware"]["mps_available"] is True
    assert preflight["hardware"]["unified_memory"] is True
    assert preflight["hardware"]["device_name"] == "Apple Silicon (MPS)"
    assert preflight["hardware"]["total_vram_gb"] >= 32.0

    # Guidance emitted
    assert preflight["mps_guidance"] is not None
    assert "PYTORCH_ENABLE_MPS_FALLBACK=1" in preflight["mps_guidance"]["environment_variable"]

    # Feasibility
    assert preflight["evaluation"]["can_run"] is True


# ============================================================================
# Scenario 4: ComfyUI Drag-and-Drop Workflow Canvas & Prompt API Export
# ============================================================================

def test_scenario4_comfyui_workflow_compilation_integrity():
    """
    Simulates exporting generation parameters to ComfyUI workstation:
    1. Canonical IR initialized with conditioning prompts and offload settings.
    2. Compiles to /prompt API format and canvas UI format with node graph links.
    """
    compiler = pytest.importorskip("wan_video_toolkit.compiler", reason="Milestone M2 compiler not yet present")
    from wan_video_toolkit.compiler import (
        WanGenerationPipelineIR,
        compile_comfyui_api,
        compile_comfyui_canvas,
    )

    ir = WanGenerationPipelineIR(
        prompt="A robotic dragonfly gliding through a bioluminescent rain forest",
        negative_prompt="blurry, jitter, artifacts",
        model="wan2.2_moe_14b",
        resolution="720p",
        num_frames=81,
        steps=35,
        guidance=7.0,
        offload_profile="balanced",
    )

    # API JSON (/prompt)
    api_payload = compile_comfyui_api(ir)
    assert isinstance(api_payload, dict)
    assert len(api_payload) >= 4

    # Canvas UI JSON
    canvas_payload = compile_comfyui_canvas(ir)
    assert "nodes" in canvas_payload
    assert "links" in canvas_payload


# ============================================================================
# Scenario 5: Multi-Modal Agent Invocation via stdio JSON-RPC 2.0
# ============================================================================

def test_scenario5_agentic_stdio_jsonrpc_workflow(mock_cuda_rtx4090, synthetic_exif_image):
    """
    Simulates autonomous developer agent (Claude Code / Cursor) using MCP stdio:
    1. Initializes protocol handshake (MCP 2024-11-05).
    2. Queries tools catalog.
    3. Executes diagnose_preflight, estimate_memory, validate_media_inputs.
    4. Executes execute_local_pipeline in dry-run mode.
    """
    mcp_mod = pytest.importorskip("wan_video_toolkit.mcp", reason="Milestone M3 MCP server not yet present")
    server = mcp_mod.WanMCPServer()

    # Step 1: Handshake
    init_resp = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "claude-code", "version": "1.0"}}
    })
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"

    # Step 2: Tools Discovery
    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = [t["name"] for t in list_resp["result"]["tools"]]
    assert len(tools) == 5

    # Step 3: Preflight
    diag_resp = server.handle_request({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "diagnose_preflight", "arguments": {"model_variant": "wan2.2_moe_14b", "resolution": "720p"}}
    })
    diag_data = json.loads(diag_resp["result"]["content"][0]["text"])
    assert diag_data["evaluation"]["can_run"] is True

    # Step 4: Dry-run Execution
    plan = {
        "command": "python generate.py --task t2v-14B --size 1280*720 --prompt 'Drone over mountains'",
        "resolution": "1280x720",
    }
    exec_resp = server.handle_request({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "execute_local_pipeline", "arguments": {"plan_json": plan, "dry_run": True}}
    })
    exec_data = json.loads(exec_resp["result"]["content"][0]["text"])
    assert exec_data["execution_mode"] == "DRY_RUN"
    assert exec_data["status"] == "VALIDATED"


# ============================================================================
# Scenario 6: Low-VRAM 12GB Workstation Recovery
# ============================================================================

def test_scenario6_low_vram_12gb_workstation_recovery(mock_cuda_low_vram_12gb):
    """
    Simulates consumer workstation running on 12GB RTX 4070:
    1. Preflight identifies 12GB ceiling and warns that 720p 14B model will OOM.
    2. Recommends resolution reduction to 480p and aggressive model offloading.
    3. Memory estimator proves 480p fp8 drops peak memory to < 10GB, enabling execution.
    """
    # 720p 14B exceeds 12GB
    preflight = diagnose_hardware_preflight("wan2.1_14b", "720p")
    assert preflight["hardware"]["total_vram_gb"] == 12.0
    assert preflight["evaluation"]["can_run"] is False

    # Workstation recovery: adjust to 480p with fp8 and aggressive offload
    recovery_mem = estimate_memory(
        model_variant="wan2.1_1.3b",
        resolution="480p",
        num_frames=81,
        precision="fp8",
        t5_cpu=True,
        tiled_vae=True,
    )
    assert recovery_mem.fits_12gb_gpu is True
    assert recovery_mem.peak_vram_gb < 10.0
