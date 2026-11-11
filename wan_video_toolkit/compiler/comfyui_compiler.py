"""
ComfyUI Execution Graph Compiler for Wan-Video.
Compiles WanGenerationPipelineIR into:
1. ComfyUI /prompt API execution graph (JSON dict with class_type & inputs)
2. ComfyUI drag-and-drop Workflow Canvas UI format (nodes & links)
"""

from typing import Dict, Any, List
from .ir import WanGenerationPipelineIR


def compile_comfyui_api(ir: WanGenerationPipelineIR) -> Dict[str, Any]:
    """
    Compiles IR into ComfyUI Prompt API JSON format accepted by POST /prompt.
    """
    nodes: Dict[str, Any] = {}
    is_tiled = ir.offload_profile in ("balanced", "low_vram", "aggressive_cpu_offload")
    precision = "fp8" if is_tiled else "bf16"

    # Node 1: Model Loader
    nodes["1"] = {
        "class_type": "WanVideoModelLoader",
        "inputs": {
            "model_name": ir.model,
            "precision": precision,
            "expert_offload": is_tiled,
            "t5_cpu": is_tiled,
        },
    }

    # Node 2: Positive Text Conditioning
    nodes["2"] = {
        "class_type": "WanTextEncode",
        "inputs": {
            "text": ir.prompt,
            "model": ["1", 0],
            "cpu_offload": is_tiled,
        },
    }

    # Node 3: Negative Text Conditioning
    nodes["3"] = {
        "class_type": "WanTextEncode",
        "inputs": {
            "text": ir.negative_prompt or "low quality, bad motion, blurry, artifacts",
            "model": ["1", 0],
            "cpu_offload": is_tiled,
        },
    }

    # Node 4: VAE Loader
    nodes["4"] = {
        "class_type": "WanVAELoader",
        "inputs": {
            "model_name": "Wan2.1_VAE.pth",
            "tiled_decode": is_tiled,
        },
    }

    # Node 5: Sampler
    sampler_inputs = {
        "model": ["1", 0],
        "positive": ["2", 0],
        "negative": ["3", 0],
        "steps": ir.steps,
        "cfg": ir.guidance,
        "seed": ir.seed if ir.seed >= 0 else 42,
        "width": ir.width,
        "height": ir.height,
        "num_frames": ir.num_frames,
        "fps": ir.fps,
    }

    # Node 6: Optional Image conditioning for I2V
    if ir.input_image:
        nodes["6"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": ir.input_image,
            },
        }
        sampler_inputs["image_conditioning"] = ["6", 0]

    nodes["5"] = {
        "class_type": "WanSampler",
        "inputs": sampler_inputs,
    }

    # Node 7: VAE Decode
    nodes["7"] = {
        "class_type": "WanVAEDecodeTiled" if is_tiled else "WanVAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["4", 0],
            "tiled": is_tiled,
        },
    }

    # Node 8: Video Save
    nodes["8"] = {
        "class_type": "WanVideoSave",
        "inputs": {
            "images": ["7", 0],
            "filename_prefix": "wan_generation",
            "fps": ir.fps,
            "format": "video/h264-mp4",
        },
    }

    return nodes


def compile_comfyui_canvas(ir: WanGenerationPipelineIR) -> Dict[str, Any]:
    """
    Compiles IR into ComfyUI Drag-and-Drop Workflow Canvas UI format.
    """
    is_tiled = ir.offload_profile in ("balanced", "low_vram", "aggressive_cpu_offload")
    nodes: List[Dict[str, Any]] = [
        {
            "id": 1,
            "type": "WanVideoModelLoader",
            "pos": [40, 100],
            "size": [300, 150],
            "widgets_values": [ir.model, "fp8" if is_tiled else "bf16", is_tiled],
        },
        {
            "id": 2,
            "type": "WanTextEncode",
            "pos": [380, 100],
            "size": [400, 180],
            "widgets_values": [ir.prompt],
        },
        {
            "id": 3,
            "type": "WanTextEncode",
            "pos": [380, 320],
            "size": [400, 180],
            "widgets_values": [ir.negative_prompt],
        },
        {
            "id": 4,
            "type": "WanSampler",
            "pos": [820, 120],
            "size": [320, 360],
            "widgets_values": [ir.steps, ir.guidance, ir.seed, ir.width, ir.height, ir.num_frames],
        },
        {
            "id": 5,
            "type": "WanVAEDecodeTiled" if is_tiled else "WanVAEDecode",
            "pos": [1180, 150],
            "size": [260, 200],
            "widgets_values": [is_tiled],
        },
        {
            "id": 6,
            "type": "WanVideoSave",
            "pos": [1480, 150],
            "size": [300, 220],
            "widgets_values": ["wan_video", ir.fps],
        },
    ]

    links: List[List[Any]] = [
        [1, 1, 0, 4, 0, "MODEL"],
        [2, 2, 0, 4, 1, "CONDITIONING"],
        [3, 3, 0, 4, 2, "CONDITIONING"],
        [4, 4, 0, 5, 0, "LATENT"],
        [5, 5, 0, 6, 0, "IMAGE"],
    ]

    return {
        "last_node_id": 6,
        "last_link_id": 5,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"generator": "wan-pilot-compiler", "version": "0.1.0"},
        "version": 0.4,
    }
