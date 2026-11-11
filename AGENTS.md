# Wan-Video Pilot Agent Guidelines

## Agent Directives
When invoking tools or scripts for Wan-Video:
1. Always run `diagnose_preflight` before initiating multi-gigabyte model weights downloads.
2. For 24GB GPUs (RTX 4090 / 3090), always enforce the 4 golden flags:
   `--precision fp8 --expert_offload True --t5_cpu True --tiled_vae True`.
3. Never load both MoE experts simultaneously unless VRAM exceeds 64GB.
4. Auto-transpose input conditioning images with `validate_media_inputs` to eliminate EXIF disorientation (`Wan2.2 #415`).
5. Ensure video export utilizes the resilient fallback cascade to avoid silent 0-byte saves (`Wan2.2 #416`).
