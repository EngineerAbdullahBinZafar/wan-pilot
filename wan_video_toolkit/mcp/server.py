"""
FastMCP and Stdio JSON-RPC 2.0 Server for Wan-Video.
Fully conforms to Model Context Protocol (MCP) specification 2024-11-05.
"""

import sys
import json
import logging
from typing import Dict, Any, List

from .tools import (
    tool_diagnose_preflight,
    tool_estimate_memory,
    tool_validate_media_inputs,
    tool_compile_workflow,
    tool_execute_local_pipeline,
)

logger = logging.getLogger("wan_pilot.mcp")


TOOLS_METADATA = [
    {
        "name": "diagnose_preflight",
        "description": "Inspects local GPU hardware environment, compute capability, VRAM headroom, and attention backends.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_variant": {"type": "string", "default": "wan2.2_moe_14b"},
                "resolution": {"type": "string", "default": "720p"},
            },
        },
    },
    {
        "name": "compile_workflow",
        "description": "Compiles canonical pipeline IR into ComfyUI (/prompt or canvas UI JSON), CLI script, or Diffusers code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "target_format": {"type": "string", "default": "cli_bash"},
                "model": {"type": "string", "default": "wan2.2_moe_14b"},
                "resolution": {"type": "string", "default": "720p"},
                "negative_prompt": {"type": "string", "default": ""},
                "image_path": {"type": "string"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "estimate_memory",
        "description": "Computes exact analytical VRAM breakdown, MoE dual-expert swapping, and tiled VAE bounds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_variant": {"type": "string", "default": "wan2.2_moe_14b"},
                "resolution": {"type": "string", "default": "720p"},
                "precision": {"type": "string", "default": "fp8"},
                "t5_cpu": {"type": "boolean", "default": True},
                "tiled_vae": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "validate_media_inputs",
        "description": "Validates and normalizes conditioning image (EXIF orientation auto-transposition) or input video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "video_path": {"type": "string"},
            },
        },
    },
    {
        "name": "execute_local_pipeline",
        "description": "Validates and simulates local execution of a generated Wan pipeline plan in dry-run mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_json": {"type": "object"},
                "dry_run": {"type": "boolean", "default": True},
            },
        },
    },
]


class WanMCPServer:
    """
    Model Context Protocol (MCP) stdio JSON-RPC 2.0 server.
    """

    def __init__(self):
        self.protocol_version = "2024-11-05"
        self.server_info = {"name": "wan-video-pilot", "version": "0.1.0"}

    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single JSON-RPC 2.0 request dictionary and returns a response dictionary."""
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": self.server_info,
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": TOOLS_METADATA,
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            try:
                if tool_name == "diagnose_preflight":
                    res = tool_diagnose_preflight(
                        model_variant=args.get("model_variant", "wan2.2_moe_14b"),
                        resolution=args.get("resolution", "720p"),
                    )
                elif tool_name == "estimate_memory":
                    res = tool_estimate_memory(
                        model_variant=args.get("model_variant", "wan2.2_moe_14b"),
                        resolution=args.get("resolution", "720p"),
                        precision=args.get("precision", "fp8"),
                        t5_cpu=args.get("t5_cpu", True),
                        tiled_vae=args.get("tiled_vae", True),
                    )
                elif tool_name == "validate_media_inputs":
                    res = tool_validate_media_inputs(
                        image_path=args.get("image_path"),
                        video_path=args.get("video_path"),
                    )
                elif tool_name == "compile_workflow":
                    res = tool_compile_workflow(
                        prompt=args.get("prompt", ""),
                        target_format=args.get("target_format", "cli_bash"),
                        model=args.get("model", "wan2.2_moe_14b"),
                        resolution=args.get("resolution", "720p"),
                        negative_prompt=args.get("negative_prompt", ""),
                        image_path=args.get("image_path"),
                    )
                elif tool_name == "execute_local_pipeline":
                    res = tool_execute_local_pipeline(
                        plan_json=args.get("plan_json"),
                        dry_run=args.get("dry_run", True),
                    )
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method not found: {tool_name}"},
                    }

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                        "isError": False,
                    },
                }

            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error executing {tool_name}: {str(e)}"}],
                        "isError": True,
                    },
                }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    def run_stdio(self):
        """Standard stdio JSON-RPC 2.0 loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
