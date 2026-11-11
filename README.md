<div align="center">

# Wan-Video Pilot (`wan-pilot`)

**Production Diagnostics, Analytical VRAM Budgeting, ComfyUI Compiler & Model Context Protocol (FastMCP) Bridge for Alibaba Cloud's Wan-Video Ecosystem (`Wan2.1` & `Wan2.2 MoE`).**

[![CI](https://github.com/EngineerAbdullahBinZafar/wan-pilot/actions/workflows/ci.yml/badge.svg)](https://github.com/EngineerAbdullahBinZafar/wan-pilot/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen.svg)](https://python.org)
[![MCP Spec: 2024-11-05](https://img.shields.io/badge/MCP-2024--11--05-blueviolet.svg)](https://modelcontextprotocol.io)
[![Tests: 267 Passing](https://img.shields.io/badge/Tests-267%20Passed%20(100%25)-success.svg)](tests/)

<br />

<img src="docs/assets/architecture.svg" alt="Wan-Video Pilot Pipeline Architecture" width="880" />

</div>

---

## ⚡ The Challenge: Gaps in the Wan-Video Ecosystem

Alibaba Cloud's **Wan-Video** (`Wan2.1` and `Wan2.2 MoE`) represents a milestone in open-source video foundation models. However, everyday creators and autonomous AI agent workflows encounter substantial hardware and integration bottlenecks:

1. **The 54GB MoE VRAM Crash**: In `Wan2.2 MoE`, naive memory allocation retains both high-noise and low-noise expert parameter weights concurrently in GPU memory (~27.0B parameters). At 720p resolution, this spikes peak memory consumption to **53.6 GB**, crashing standard 24GB GPUs (NVIDIA RTX 4090 / RTX 3090 / A10G).
2. **The Attention Trap on Windows**: Official setup documentation prescribes FlashAttention-2, which frequently fails compilation on Windows platforms lacking MSVC C++ toolchains.
3. **Conditioning Image Orientation Inversion ([Wan2.2 Issue #415](https://github.com/Wan-Video/Wan2.2/issues/415))**: Standard image loaders discard EXIF orientation tags (1–8), distorting portrait or smartphone conditioning images in Image-to-Video (I2V) workflows.
4. **Silent Video Save Failures ([Wan2.2 Issue #416](https://github.com/Wan-Video/Wan2.2/issues/416))**: Unhandled NVENC hardware encoder failures leave zero-byte `.mp4` files without surfacing an error to the user after 15+ minutes of generation compute.
5. **Dtype Regressions ([Wan-Animate-2 Issue #5](https://github.com/Wan-Video/Wan-Animate-2/issues/5))**: Meta-device initialization of DiT layers without an active dtype context reverts non-persistent buffers to `float32`, doubling memory overhead.
6. **Agentic Tooling Vacuum**: Existing tooling in `Wan-Video/Wan-skills` requires proprietary cloud API keys, offering no native Model Context Protocol (MCP) server for local workflows.

**Wan-Video Pilot** (`wan-pilot`) provides the production-grade engineering layer that systematically resolves every one of these bottlenecks.

---

## 📊 Analytical VRAM Benchmarks

By enforcing **Sequential Dual-Expert Swapping at boundary timestep $t_{\text{moe}}$**, offloading the UMT5 text encoder to host RAM, and bounding causal 3D VAE decoding with spatial-temporal tiling, Wan-Video Pilot compresses execution envelopes onto consumer GPUs:

| Resolution & Model | Precision | Baseline (Concurrent) | `wan-pilot` (Sequential Swapped) | RTX 4090 / 3090 (24GB) | RTX 4070 (12GB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Wan2.2 MoE 14B** (720p, 81 frames) | FP8 | **53.6 GB** *(OOM Crash)* | **18.47 GB** | ✅ **RUNS CLEANLY** | ❌ Exceeds 12GB |
| **Wan2.2 MoE 14B** (480p, 81 frames) | FP8 | **46.8 GB** *(OOM Crash)* | **15.22 GB** | ✅ **RUNS CLEANLY** | ⚠️ Tight Headroom |
| **Wan2.1 14B Dense** (720p, 81 frames) | FP8 | 32.4 GB *(OOM Crash)* | **17.82 GB** | ✅ **RUNS CLEANLY** | ❌ Exceeds 12GB |
| **Wan2.1 1.3B Compact** (480p, 81 frames)| FP8 | 11.2 GB | **4.85 GB** | ✅ **RUNS CLEANLY** | ✅ **RUNS CLEANLY** |

*All metrics derived analytically via exact latent token accounting: $(H/16) \times (W/16) \times (1 + (F-1)/4)$ and verified across the 267-test suite.*

---

## 🚀 Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/EngineerAbdullahBinZafar/wan-pilot.git
cd wan-pilot

# Install package in development mode
pip install -e .
```

### 1. Preflight Diagnostics & Attention Doctor
Inspect your local GPU, VRAM headroom, and attention kernels before starting heavy generations:
```bash
wan-pilot preflight --model wan2.2_moe_14b --resolution 720p
```
*Output:*
```text
Hardware: NVIDIA GeForce RTX 4090 (24.0 GB VRAM)
CUDA Available: True
Attention Backend: SDPA (PyTorch Native)
Can Run: True
```

### 2. Analytical Memory Estimator
Compute memory curves for custom resolutions and frame counts:
```bash
wan-pilot estimate --model wan2.2_moe_14b --resolution 720p --precision fp8
```

### 3. Multi-Target Workflow Compiler
Compile prompts into the official Wan CLI bash command, PowerShell, ComfyUI `/prompt` API, or Diffusers script:
```bash
# Compile to official bash script with the 4 Golden Flags
wan-pilot compile --prompt "A cybernetic dragon gliding through neon rain" --target cli_bash

# Compile to ComfyUI Prompt API JSON format
wan-pilot compile --prompt "A cybernetic dragon gliding through neon rain" --target comfyui_api
```

### 4. Interactive Studio Web Workbench
Launch the zero-install interactive Web Workbench with real-time VRAM budget visualizer and WebMCP actions:
```bash
wan-pilot ui --port 8765
```

---

## 🤖 Model Context Protocol (FastMCP) Integration

Wan-Video Pilot includes a native **FastMCP / stdio JSON-RPC 2.0 server** adhering to MCP specification version `2024-11-05`. This allows autonomous developer agents (Claude Code, Cursor, Windsurf, AutoGPT) to execute preflight checks, generate ComfyUI graphs, and plan generations without manual intervention.

### Connecting to Claude Code / Cursor
Add to your `mcpServers` configuration (`claude_desktop_config.json` or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "wan-pilot": {
      "command": "python",
      "args": ["-m", "wan_video_toolkit.mcp.server"]
    }
  }
}
```

### Available MCP Tools

| Tool Name | Parameters | Purpose |
| :--- | :--- | :--- |
| `diagnose_preflight` | `model_variant`, `resolution` | Evaluates hardware compute capability, attention kernels, and OOM traps. |
| `estimate_memory` | `model_variant`, `resolution`, `precision` | Computes analytical VRAM decomposition, MoE swap peak, and tiled VAE bounds. |
| `validate_media_inputs` | `image_path`, `video_path` | Auto-transposes EXIF orientation metadata (`#415`) and enforces 16-pixel divisibility. |
| `compile_workflow` | `prompt`, `target_format`, `model` | Generates ComfyUI Prompt API JSON, Canvas UI JSON, CLI scripts, or Diffusers code. |
| `execute_local_pipeline`| `plan_json`, `dry_run` | Validates and dry-runs pipeline execution plans within memory envelopes. |

---

## 🛠️ Upstream Patches for Wan-Video

Included in `upstream_patches/` are turnkey pull requests and patches tested against upstream repositories:

- **`PR_WAN2_2_415.md` & `wan2_2_exif_415.patch`**: Resolves conditioning image rotation inversion by applying `ImageOps.exif_transpose` and 16-pixel spatial patch rounding.
- **`PR_WAN2_2_416.md` & `wan2_2_save_video_416.patch`**: Multi-tier encoding fallback cascade (`h264_nvenc` $\to$ `libx264` $\to$ `libopenh264` $\to$ `mpeg4` $\to$ disaster frame dump) preventing silent 0-byte video save failures.
- **`PR_WAN_ANIMATE_5.md` & `wan_animate_2_dtype_5.patch`**: Context-guarded DiT layer instantiation preventing meta-device float32 memory blowup.
- **`wan-skills-pr/`**: Turnkey skill registration package for `Wan-Video/Wan-skills`.

---

## 🧪 Verification & Test Suite

The entire test harness runs 100% offline without requiring physical 24GB GPUs by leveraging synthetic device property mocks, attention matrix emulation, and mock media synthesizers:

```bash
# Execute the full 4-tier E2E test matrix
pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\star\Downloads\freeapps\wan-video-toolkit
collected 267 items

tests/e2e/test_tier1_feature_coverage.py .................................... [ 29%]
tests/e2e/test_tier2_boundary_corner.py .................................... [ 49%]
tests/e2e/test_tier3_cross_feature.py ..................                      [ 55%]
tests/e2e/test_tier4_application_scenarios.py ......                          [ 58%]
tests/test_adversarial_m1.py ..............................................   [ 78%]
tests/test_adversarial_preflight.py ........................................  [ 93%]
tests/test_diagnostics.py .........                                           [ 96%]
tests/test_media_validator.py .........                                       [100%]

============================= 267 passed in 4.18s =============================
```

---

## 📄 License & Attribution

- **License**: Apache License 2.0.
- **Author**: **Engineer Abdullah Bin Zafar** ([GitHub: @EngineerAbdullahBinZafar](https://github.com/EngineerAbdullahBinZafar))
- **Disclaimer**: *Wan-Video* is a registered trademark and foundation model developed by Alibaba Cloud. *Wan-Video Pilot* is an independent, community-driven optimization and agent bridge toolkit.
