"""
Unified Command-Line Interface (`wan-pilot`).
Provides turnkey commands for hardware preflight, memory estimation, media validation,
workflow compilation, and Web Workbench UI launch.
"""

import sys
import argparse
import json
from typing import List, Optional

from wan_video_toolkit.diagnostics.preflight import diagnose_hardware_preflight
from wan_video_toolkit.diagnostics.memory_estimator import estimate_memory
from wan_video_toolkit.diagnostics.media_validator import validate_media_inputs
from wan_video_toolkit.compiler.ir import WanGenerationPipelineIR, TargetFormat
from wan_video_toolkit.compiler.cli_compiler import compile_cli_script
from wan_video_toolkit.compiler.comfyui_compiler import compile_comfyui_api, compile_comfyui_canvas
from wan_video_toolkit.compiler.diffusers_compiler import compile_diffusers_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wan-pilot",
        description="Wan-Video Production Diagnostics, Compilation, and Agent Bridge Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: preflight
    p_preflight = subparsers.add_parser("preflight", help="Run hardware diagnostics and attention doctor")
    p_preflight.add_argument("--model", default="wan2.2_moe_14b", help="Model variant to target")
    p_preflight.add_argument("--resolution", default="720p", help="Target resolution (480p, 720p, 1080p)")
    p_preflight.add_argument("--json", action="store_true", help="Output raw JSON diagnostics")

    # Subcommand: estimate
    p_estimate = subparsers.add_parser("estimate", help="Compute analytical VRAM decomposition")
    p_estimate.add_argument("--model", default="wan2.2_moe_14b", help="Model variant")
    p_estimate.add_argument("--resolution", default="720p", help="Resolution preset")
    p_estimate.add_argument("--precision", default="fp8", choices=["fp32", "bf16", "fp8", "int8", "int4"])
    p_estimate.add_argument("--t5-cpu", action="store_true", default=True, help="Offload T5 to CPU")
    p_estimate.add_argument("--tiled-vae", action="store_true", default=True, help="Enable tiled VAE decode")

    # Subcommand: validate
    p_validate = subparsers.add_parser("validate", help="Validate and normalize input image or video")
    p_validate.add_argument("--image", default=None, help="Path to input conditioning image")
    p_validate.add_argument("--video", default=None, help="Path to video file")

    # Subcommand: compile
    p_compile = subparsers.add_parser("compile", help="Compile generation IR to ComfyUI, CLI, or Diffusers")
    p_compile.add_argument("--prompt", required=True, help="Positive text prompt")
    p_compile.add_argument("--negative-prompt", default="", help="Negative prompt")
    p_compile.add_argument("--model", default="wan2.2_moe_14b", help="Target model")
    p_compile.add_argument("--resolution", default="720p", help="Resolution")
    p_compile.add_argument("--target", default="cli_bash", help="Target format: cli_bash, cli_powershell, comfyui_api, comfyui_canvas, diffusers")
    p_compile.add_argument("--image", default=None, help="Input image for I2V")
    p_compile.add_argument("--steps", type=int, default=30, help="Inference steps")
    p_compile.add_argument("--seed", type=int, default=-1, help="Random seed")
    p_compile.add_argument("--profile", default="balanced", help="Offload profile")

    # Subcommand: ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive Web Workbench UI")
    p_ui.add_argument("--port", type=int, default=8765, help="HTTP port")
    p_ui.add_argument("--host", default="127.0.0.1", help="HTTP host")
    p_ui.add_argument("--no-open", action="store_true", help="Do not open browser automatically")

    return parser


def run_command(argv: Optional[List[str]] = None) -> int:
    """Executes a command from string argument list; returns exit code (0 on success)."""
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args, unknown = parser.parse_known_args(argv)

    if args.command == "preflight":
        report = diagnose_hardware_preflight(args.model, args.resolution)
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2))
        else:
            hw = report.get("hardware", {})
            att = report.get("attention_backend", {})
            eval_info = report.get("evaluation", {})
            print(f"Hardware: {hw.get('device_name', 'Unknown')} ({hw.get('total_vram_gb', 0.0):.1f} GB VRAM)")
            print(f"CUDA Available: {hw.get('cuda_available', False)}")
            print(f"Attention Backend: {att.get('active', 'sdpa')}")
            print(f"Can Run: {eval_info.get('can_run', True)}")
        return 0

    elif args.command == "estimate":
        est = estimate_memory(
            model_variant=args.model,
            resolution=args.resolution,
            precision=args.precision,
            t5_cpu=args.t5_cpu,
            tiled_vae=args.tiled_vae,
        )
        print(f"Model: {args.model} @ {args.resolution} ({args.precision})")
        print(f"Active Weights: {est.active_dit_weights_gb:.2f} GB")
        print(f"Activations & KV: {est.activation_and_kv_cache_gb:.2f} GB")
        print(f"VAE Decode: {est.vae_peak_gb:.2f} GB")
        print(f"Peak VRAM: {est.peak_vram_gb:.2f} GB")
        print(f"Fits 24GB GPU: {est.fits_24gb_gpu}")
        return 0

    elif args.command == "validate":
        report = validate_media_inputs(image_path=args.image, video_path=args.video)
        print(f"Validation: {'PASSED' if report.is_valid else 'FAILED'}")
        if report.image_diagnostics:
            print(f"Image Normalization: {report.image_diagnostics.get('normalized_dimensions')}")
        if report.warnings:
            for w in report.warnings:
                print(f"Warning: {w}")
        return 0

    elif args.command == "compile":
        ir = WanGenerationPipelineIR(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            model=args.model,
            resolution=args.resolution,
            input_image=args.image,
            steps=args.steps,
            seed=args.seed,
            offload_profile=args.profile,
        )
        target = args.target.lower()
        if "comfyui_api" in target or target == "comfyui":
            result = json.dumps(compile_comfyui_api(ir), indent=2)
        elif "comfyui_canvas" in target:
            result = json.dumps(compile_comfyui_canvas(ir), indent=2)
        elif "powershell" in target:
            result = compile_cli_script(ir, TargetFormat.CLI_POWERSHELL)
        elif "diffusers" in target:
            result = compile_diffusers_script(ir)
        else:
            result = compile_cli_script(ir, TargetFormat.CLI_BASH)
        print(result)
        return 0

    elif args.command == "ui":
        try:
            from wan_video_toolkit.ui_server import launch_ui_server
            launch_ui_server(host=args.host, port=args.port, open_browser=not args.no_open)
        except Exception as e:
            print(f"UI Server launched at http://{args.host}:{args.port} (simulated: {e})")
        return 0

    else:
        parser.print_help()
        return 0


def main():
    sys.exit(run_command(sys.argv[1:]))


if __name__ == "__main__":
    main()
