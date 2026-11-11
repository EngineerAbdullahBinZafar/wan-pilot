"""
Model Context Protocol (MCP) server module for Wan-Video.
Exports the stdio JSON-RPC 2.0 server and tool definitions.
"""

from .server import WanMCPServer
from .tools import (
    tool_diagnose_preflight,
    tool_estimate_memory,
    tool_validate_media_inputs,
    tool_compile_workflow,
    tool_execute_local_pipeline,
)

__all__ = [
    "WanMCPServer",
    "tool_diagnose_preflight",
    "tool_estimate_memory",
    "tool_validate_media_inputs",
    "tool_compile_workflow",
    "tool_execute_local_pipeline",
]
