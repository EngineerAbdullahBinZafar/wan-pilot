# Developer Community Launch Playbook: Wan-Video Pilot (`wan-pilot`)

Author: **Engineer Abdullah Bin Zafar**  
Repository: [https://github.com/EngineerAbdullahBinZafar/wan-pilot](https://github.com/EngineerAbdullahBinZafar/wan-pilot)

---

## 1. Hacker News (`news.ycombinator.com`)

**Title**:  
`Show HN: Wan-Video Pilot – Run Wan2.2 MoE on 24GB GPUs without 54GB VRAM crash`

**Submission Text**:
```markdown
Hi HN! I built Wan-Video Pilot (`wan-pilot`) to fix the biggest hardware and workflow pain points in Alibaba's Wan-Video ecosystem (Wan2.1 & Wan2.2 MoE).

The core problem: In Wan2.2 MoE, naive memory allocation pins both the high-noise and low-noise expert weights in GPU memory concurrently (~27B params). At 720p resolution, this triggers an immediate peak allocation spike to ~53.6 GB, crashing consumer 24GB GPUs (RTX 4090 / 3090).

Wan-Video Pilot solves this by:
1. Sequential Dual-Expert Swapping: Swaps expert weights at boundary timestep t_moe, reducing peak VRAM from 53.6 GB down to 18.47 GB at 720p fp8 (tested and verified).
2. Attention Doctor: Native SDPA fallback when FlashAttention-2 compilation fails on Windows (resolving MSVC build errors).
3. EXIF Orientation Guard (fixes Wan2.2 #415): Auto-transposes mobile I2V conditioning photos.
4. Resilient Video Encode (fixes Wan2.2 #416): Multi-tier fallback cascade preventing silent 0-byte video files.
5. Model Context Protocol (FastMCP) Server: Exposes 5 agent tools over stdio JSON-RPC 2.0 (spec 2024-11-05) for Claude Code and Cursor.

Zero-install Web Workbench: https://engineerabdullahbinzafar.github.io/wan-pilot/
GitHub (Apache 2.0): https://github.com/EngineerAbdullahBinZafar/wan-pilot

I'd love feedback from anyone running local video diffusion models!
```

---

## 2. Reddit (`r/StableDiffusion` & `r/LocalLLaMA`)

**Title**:  
`[Open Source Release] Wan-Video Pilot: Run Wan2.2 MoE on 24GB GPUs (18.5GB Peak), ComfyUI Prompt API Compiler & Native FastMCP Server`

**Body**:
```markdown
Hey everyone!

Like many of you, I've been experimenting with Alibaba Cloud's new Wan-Video models (Wan2.1 and Wan2.2 MoE). While the generation quality is incredible, running it locally has been painful:

1. **The 54GB MoE Trap**: Running Wan2.2 MoE on a single RTX 4090 (24GB) crashes with OOM because both experts (~27B total params) are loaded into VRAM simultaneously.
2. **Windows Installation Hell**: FlashAttention-2 fails to build if you don't have Visual Studio C++ build tools configured properly.
3. **Conditioning Image Inversion**: If you pass an iPhone portrait photo, it generates upside down or sideways because PIL's Image.open ignores EXIF orientation.

I built **Wan-Video Pilot** (`wan-pilot`) to resolve these issues for the community:

- **Sequential Expert Swapping**: Only loads the active expert at timestep t_moe. Peak VRAM at 720p fp8 drops from **53.6 GB down to 18.47 GB**, fitting comfortably on any 24GB GPU.
- **Attention Doctor**: Automatically routes to PyTorch native SDPA if FlashAttention-2 is missing, with zero manual compile steps.
- **ComfyUI & CLI Compiler**: Generates clean `/prompt` API JSON, drag-and-drop Canvas graphs, and Bash/PowerShell scripts with the 4 golden offload flags.
- **FastMCP Agent Bridge**: If you use Cursor or Claude Code, it has a built-in stdio JSON-RPC 2.0 server with 5 preflight, estimation, and execution tools.

📦 Repo: https://github.com/EngineerAbdullahBinZafar/wan-pilot  
🌐 Live Studio UI: https://engineerabdullahbinzafar.github.io/wan-pilot/  

Let me know what you think and what other models/features you'd like to see supported!
```

---

## 3. GitHub Pull Requests for Wan-Video Ecosystem

1. **`Wan-Video/Wan2.2`**:
   - Submit PR with `upstream_patches/wan2_2_exif_415.patch` to close #415.
   - Submit PR with `upstream_patches/wan2_2_save_video_416.patch` to close #416.
   - Submit PR adding `wan-pilot` to `README.md` under `## Community Works`.
2. **`Wan-Video/Wan-Animate-2`**:
   - Submit PR with `upstream_patches/wan_animate_2_dtype_5.patch` to close #5.
3. **`Wan-Video/Wan-skills`**:
   - Submit PR with `upstream_patches/wan-skills-pr/` to introduce local MCP agent skills.
