# Wan-Video Pilot Skill Contribution for Wan-skills

This directory provides a ready-to-merge contribution package for [`Wan-Video/Wan-skills`](https://github.com/Wan-Video/Wan-skills).

## Background & Need
Currently, `Wan-Video/Wan-skills` contains only cloud-based DashScope API wrappers. This PR introduces the first **Local Hardware & Agentic Bridge Skill** that enables autonomous developer agents (Cursor, Claude Code, AutoGPT, LangChain) to interface directly with local or remote Wan2.1 and Wan2.2 MoE instances via the standard Model Context Protocol (MCP spec `2024-11-05`).

## Included Features
1. **Model Context Protocol (MCP) Stdio JSON-RPC 2.0 Server**:
   - `diagnose_preflight`: Hardware and attention backend verification.
   - `estimate_memory`: Exact analytical VRAM breakdown with MoE swapping calculations.
   - `validate_media_inputs`: Automated EXIF transposition (`#415`) and 16-pixel rounding.
   - `compile_workflow`: Multi-target compiler (ComfyUI /prompt JSON, ComfyUI canvas, Bash, PowerShell, Diffusers).
   - `execute_local_pipeline`: Safe dry-run and local pipeline runner.
2. **Zero-install Web Workbench**:
   - WebMCP schema declarations and dark studio visualizer.

## Installation into Wan-skills
Copy this directory to `Wan-skills/wan-pilot` and register the skill in `Wan-skills` catalog.
