---
name: wan-video-pilot
description: Autonomous preflight diagnostics, analytical VRAM budgeting, ComfyUI/CLI compilation, and execution bridge for Wan-Video (Wan2.1 & Wan2.2 MoE).
version: 0.1.0
author: EngineerAbdullahBinZafar
license: Apache-2.0
---

# Wan-Video Pilot (`wan-pilot`)

A production-grade developer toolkit and agent bridge for Alibaba Cloud's **Wan-Video** ecosystem (`Wan2.1` and `Wan2.2 MoE`). Solves the community hardware gap (VRAM consumption & MoE dual-expert swapping crashes), the ComfyUI integration gap, and the autonomous AI agent tooling gap.

## 1. Core Capabilities

1. **Hardware & VRAM Diagnostics**:
   - Evaluates compute capability, available VRAM, and memory headroom.
   - Attention Doctor: Detects PyTorch native SDPA, FlashAttention-2, and provides seamless Windows fallback for missing MSVC compilers.
   - Detects 4 critical OOM traps: `OOM_TRAP_MOE_CONCURRENT`, `OOM_TRAP_T5_VRAM`, `OOM_TRAP_DIT_FP32`, and `OOM_TRAP_VAE_DEC_SPIKE`.

2. **Analytical VRAM Decomposition**:
   - Exact mathematical token accounting: $(H/16) \times (W/16) \times (1 + (F-1)/4)$.
   - MoE dual-expert sequential offloading calculation at boundary timestep $t_{\text{moe}}$ preventing the 54GB concurrent allocation crash.
   - Tiled 3D causal VAE decode memory bounding (~1.8GB peak).

3. **Multi-Target Compiler**:
   - Compiles canonical IR (`WanGenerationPipelineIR`) to:
     - ComfyUI Prompt API (`/prompt`) JSON graphs.
     - ComfyUI drag-and-drop Workflow Canvas UI format.
     - Production Bash & PowerShell CLI scripts with the 4 golden offload flags:
       `--precision fp8 --expert_offload True --t5_cpu True --tiled_vae True`.
     - Standalone Hugging Face Diffusers scripts.

4. **Media Validation & Resilient Encoders**:
   - Resolves `Wan2.2 #415`: Auto-transposes conditioning images with EXIF orientation metadata (tags 1–8) and ensures 16-pixel divisibility.
   - Resolves `Wan2.2 #416`: Resilient multi-tier video encoding cascade (`h264_nvenc` $\to$ `libx264` $\to$ `libopenh264` $\to$ disaster frame dump) with post-write byte size verification.
   - Resolves `Wan-Animate-2 #5`: Enforces `dit_dtype_guard` to prevent DiT layers from quietly reverting to float32 on meta-device instantiation.

5. **Model Context Protocol (MCP)**:
   - FastMCP and stdio JSON-RPC 2.0 server conforming to MCP specification `2024-11-05`.
   - 5 callable tools: `diagnose_preflight`, `estimate_memory`, `validate_media_inputs`, `compile_workflow`, `execute_local_pipeline`.

## 2. FastMCP CLI & Stdio Usage

```bash
# Start MCP server over stdio
python -m wan_video_toolkit.mcp.server

# Unified CLI commands
wan-pilot preflight --model wan2.2_moe_14b --resolution 720p
wan-pilot estimate --model wan2.2_moe_14b --resolution 720p --precision fp8
wan-pilot validate --image path/to/input.jpg
wan-pilot compile --prompt "A cybernetic falcon soaring over futuristic dunes" --target cli_bash
wan-pilot ui --port 8765
```

## 3. Agent Tool Call Examples

### Diagnose Preflight
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "diagnose_preflight",
    "arguments": {
      "model_variant": "wan2.2_moe_14b",
      "resolution": "720p"
    }
  }
}
```

### Compile Workflow
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "compile_workflow",
    "arguments": {
      "prompt": "Bioluminescent marine organisms pulsing in deep ocean trench",
      "target_format": "comfyui_api",
      "resolution": "720p"
    }
  }
}
```
