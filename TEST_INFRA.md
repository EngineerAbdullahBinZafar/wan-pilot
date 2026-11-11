# E2E Test Infra: Wan-Video Toolkit (`wan-pilot`)

## Test Philosophy
- Opaque-box, requirement-driven, deterministic video generation and compiler verification.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.
- Hard invariant: Memory safety for consumer GPUs ($< 24\ \text{GB}$ VRAM ceiling enforced via sequential MoE expert swapping, UMT5 CPU offload, and tiled VAE).
- Media and execution safety: Zero silent failures (post-write size assertion $> 1024$ bytes, multi-tier codec fallback cascade), and orientation preservation via EXIF Tag `0x0112` transposition.
- Offline synthetic testability: 100% executable without physical GPUs via high-fidelity synthetic mock layers (CUDA properties, attention matrix doctor, meta-device memory budgeting, and media containers).

## Feature Inventory & Test Matrix
| # | Feature | Source | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross/Pairwise) | Tier 4 (Scenario) |
|---|---------|--------|:-----------------:|:-----------------:|:-----------------------:|:-----------------:|
| 1 | Hardware & VRAM Preflight Inspection | M1, Survey 1/3 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 2 | Analytical VRAM Estimation Engine | M1, Survey 1 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 3 | EXIF Transposition & Input Normalizer | M1, Survey 1/3 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 4 | Resilient Video Save with Codec Fallbacks | M1, Survey 1/3 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 5 | Dtype Trap Prevention | M1, Survey 1/3 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 6 | Canonical Pipeline IR Schema | M2, Survey 2 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 7 | ComfyUI Execution Graph Compiler | M2, Survey 2 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 8 | CLI & Diffusers Script Compilers | M2, Survey 2 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 9 | Unified CLI Tool (`wan-pilot`) | M2, Survey 2/3 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 10 | FastMCP / stdio MCP Server (5 tools) | M3, Survey 3 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |
| 11 | Web Workbench UI & WebMCP Actions | M4, Survey 2 | $\ge 5$ tests | $\ge 5$ tests | ✓ | ✓ |

## Test Architecture
- Test Runner: Standard `pytest` invocation (`pytest tests/ -v`).
- Directory Structure:
  ```
  wan-video-toolkit/
  ├── tests/
  │   ├── conftest.py                             # 4-layer synthetic mocks & test fixtures
  │   └── e2e/
  │       ├── test_tier1_feature_coverage.py       # Tier 1: Functional feature coverage (>=50 tests)
  │       ├── test_tier2_boundary_corner.py        # Tier 2: Boundary value & error traps (>=40 tests)
  │       ├── test_tier3_cross_feature.py          # Tier 3: Cross-feature & pairwise pipelines (>=15 tests)
  │       └── test_tier4_application_scenarios.py  # Tier 4: Real-world workloads & agent flows (>=5 tests)
  ```
- Synthetic Mock Layers (`tests/conftest.py`):
  1. `MockTorchCUDA`: Emulates RTX 4090, RTX 3090, Apple MPS, and CPU-only devices (`torch.cuda.is_available`, `mem_get_info`, `device_properties`).
  2. `MockAttentionMatrix`: Simulates presence/absence of SDPA, FlashAttention-2, SageAttention, and Windows MSVC compilation hurdles.
  3. `MockMetaTensors`: Simulates meta-device zero-memory parameter allocations and patchification for Wan2.1 and Wan2.2 MoE.
  4. `MockMediaSynthesizer`: Synthesizes EXIF-rotated JPEGs (Tags 1-8), valid MP4 containers, 0-byte corrupt files, and read-only targets.
  5. `MockMCPIO`: Stdio JSON-RPC 2.0 protocol testing harness simulating client handshakes and tool requests.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target Environment | Target Model |
|---|----------|--------------------|-------------------|--------------|
| 1 | RTX 3090/4090 24GB Golden Flags T2V | Preflight, Memory Estimator, Compiler, CLI | Consumer 24GB GPU | Wan2.2 MoE 14B (720p, 81 frames) |
| 2 | Smartphone Photo I2V Auto-Correction | EXIF Transposition, 16x Padding, Safe Save | Mobile Upload Pipeline | Wan2.1 14B I2V |
| 3 | Apple Silicon Unified Memory Fallback | MPS Preflight, Shared RAM Budgeting, CPU Offload | macOS M3 Max (36GB) | Wan2.2 MoE 14B |
| 4 | ComfyUI Drag-and-Drop Canvas & API Export | Canonical IR, ComfyUI Compiler, Node Layout | ComfyUI Workstation | Wan2.1 1.3B & 14B |
| 5 | Stdio JSON-RPC 2.0 AI Agent Workflow | FastMCP Server, Preflight, Memory, Media, Dry-Run | Claude Code / Cursor Agent | Wan2.2 MoE Dual-Expert |

## Coverage Thresholds
- Tier 1: $\ge 5$ tests per feature ($\ge 50$ tests total, target: 55+)
- Tier 2: $\ge 5$ boundary/corner cases per feature ($\ge 40$ tests total, target: 45+)
- Tier 3: Pairwise feature interactions & cascading pipelines ($\ge 15$ tests, target: 16+)
- Tier 4: Realistic end-to-end multi-modal application scenarios ($\ge 5$ tests, target: 6+)
- Total suite minimum: $\ge 110$ tests (target: 120+ tests)
- 100% pass rate requirement across all platforms in offline CI.
