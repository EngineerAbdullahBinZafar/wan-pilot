"""
Global Pytest Fixtures & 4-Tier Synthetic Mocking Harness for Wan-Video Toolkit.
Enables 100% offline CI execution without physical GPUs or external hardware.

Mock Layers:
1. MockTorchCUDA: Simulates RTX 4090, RTX 3090, Apple Silicon MPS, and CPU-only hosts.
2. MockAttentionMatrix: Simulates SDPA, FlashAttention-2, SageAttention, xFormers, and Windows MSVC absence.
3. MockMetaTensors: Simulates zero-memory meta-device tensor operations and patchification.
4. MockMediaSynthesizer: Programmatic generation of EXIF rotated JPEGs (Tags 1-8), valid MP4 containers, corrupt files, read-only dirs.
5. MockMCPIO & CLIRunner: Stdio JSON-RPC 2.0 protocol testing harness.
"""

import os
import sys
import io
import struct
import json
import types
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ============================================================================
# Synthetic Fallback Torch Module (Enables 100% offline runs when torch is absent)
# ============================================================================

def _ensure_torch_available():
    """Provides a synthetic torch mock module if torch is not installed in CI."""
    if "torch" in sys.modules and sys.modules["torch"] is not None:
        return sys.modules["torch"]
    try:
        import torch
        return torch
    except ImportError:
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.4.0+cpu.mock"

        # Dtypes
        class Dtype:
            def __init__(self, name):
                self.name = name
            def __repr__(self):
                return f"torch.{self.name}"

        fake_torch.float32 = Dtype("float32")
        fake_torch.float16 = Dtype("float16")
        fake_torch.bfloat16 = Dtype("bfloat16")
        fake_torch.int8 = Dtype("int8")
        fake_torch.uint8 = Dtype("uint8")

        current_default_dtype = fake_torch.float32

        def get_default_dtype():
            return current_default_dtype

        def set_default_dtype(d):
            nonlocal current_default_dtype
            current_default_dtype = d

        fake_torch.get_default_dtype = get_default_dtype
        fake_torch.set_default_dtype = set_default_dtype

        # Cuda submodule
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: False
        fake_cuda.device_count = lambda: 0
        fake_cuda.get_device_name = lambda idx=0: "CPU"
        fake_cuda.mem_get_info = lambda device=None: (0, 0)
        fake_cuda.memory_allocated = lambda device=None: 0
        fake_cuda.empty_cache = lambda: None

        class DeviceProperties:
            name = "CPU"
            total_memory = 0
            major = 0
            minor = 0
            multi_processor_count = 0

        fake_cuda.get_device_properties = lambda idx=0: DeviceProperties()
        fake_torch.cuda = fake_cuda

        # Backends submodule
        fake_backends = types.ModuleType("torch.backends")
        fake_mps = types.ModuleType("torch.backends.mps")
        fake_mps.is_available = lambda: False
        fake_backends.mps = fake_mps
        fake_torch.backends = fake_backends

        # NN submodule
        fake_nn = types.ModuleType("torch.nn")
        fake_functional = types.ModuleType("torch.nn.functional")
        fake_functional.scaled_dot_product_attention = lambda q, k, v, **kwargs: q
        fake_nn.functional = fake_functional

        class FakeModule:
            def __init__(self):
                self._parameters = {}
                self._modules = {}
            def parameters(self):
                return []
            def to(self, *args, **kwargs):
                return self

        fake_nn.Module = FakeModule
        fake_torch.nn = fake_nn

        # Device
        class FakeDevice:
            def __init__(self, name):
                self.type = str(name)
            def __repr__(self):
                return f"device(type='{self.type}')"

        fake_torch.device = FakeDevice
        fake_torch.zeros = lambda *shape, **kwargs: MagicMock(shape=shape, dtype=kwargs.get("dtype", fake_torch.float32))
        fake_torch.randn = lambda *shape, **kwargs: MagicMock(shape=shape, dtype=kwargs.get("dtype", fake_torch.float32))

        sys.modules["torch"] = fake_torch
        sys.modules["torch.cuda"] = fake_cuda
        sys.modules["torch.backends"] = fake_backends
        sys.modules["torch.backends.mps"] = fake_mps
        sys.modules["torch.nn"] = fake_nn
        sys.modules["torch.nn.functional"] = fake_functional
        return fake_torch

_ensure_torch_available()


# ============================================================================
# Layer 1: MockTorchCUDA Fixtures
# ============================================================================

class FakeCUDADeviceProperties:
    def __init__(self, name="NVIDIA GeForce RTX 4090", total_memory=24 * 1024**3, major=8, minor=9, sm=128):
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor
        self.multi_processor_count = sm


@pytest.fixture
def mock_cuda_rtx4090(monkeypatch):
    """Mocks an available NVIDIA RTX 4090 (24GB VRAM, 22.4GB free, compute 8.9)."""
    import torch
    props = FakeCUDADeviceProperties("NVIDIA GeForce RTX 4090", 24 * 1024**3, 8, 9, 128)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
        int(22.4 * 1024**3),  # Free: 22.4 GB
        int(24.0 * 1024**3),  # Total: 24.0 GB
    ))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(1.6 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return props


@pytest.fixture
def mock_cuda_rtx3090(monkeypatch):
    """Mocks an available NVIDIA RTX 3090 (24GB VRAM, 21.0GB free, compute 8.6)."""
    import torch
    props = FakeCUDADeviceProperties("NVIDIA GeForce RTX 3090", 24 * 1024**3, 8, 6, 82)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
        int(21.0 * 1024**3),
        int(24.0 * 1024**3),
    ))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(2.0 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return props


@pytest.fixture
def mock_cuda_low_vram_12gb(monkeypatch):
    """Mocks an NVIDIA RTX 4070 (12GB VRAM, 11.0GB free)."""
    import torch
    props = FakeCUDADeviceProperties("NVIDIA GeForce RTX 4070", 12 * 1024**3, 8, 9, 46)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 4070")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (
        int(11.0 * 1024**3),
        int(12.0 * 1024**3),
    ))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(1.0 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return props


@pytest.fixture
def mock_apple_mps(monkeypatch):
    """Mocks Apple Silicon MPS runtime with 36GB unified memory."""
    import torch
    import psutil
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    fake_vm = MagicMock(total=36 * 1024**3, available=28 * 1024**3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake_vm)
    return fake_vm


@pytest.fixture
def mock_cpu_only(monkeypatch):
    """Mocks a CPU-only host with 16GB RAM and no GPU accelerator."""
    import torch
    import psutil
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    fake_vm = MagicMock(total=16 * 1024**3, available=12 * 1024**3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake_vm)
    return fake_vm


# ============================================================================
# Layer 2: MockAttentionMatrix Fixtures
# ============================================================================

@pytest.fixture
def mock_attention_sdpa(monkeypatch):
    """Simulates native PyTorch SDPA active without third-party attention libraries."""
    import torch
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", lambda q, k, v, **kwargs: q, raising=False)
    # Block flash_attn and sageattention
    orig_import = __import__
    def guarded_import(name, *args, **kwargs):
        if name in ("flash_attn", "sageattention", "xformers", "xformers.ops"):
            raise ImportError(f"Simulated absence of {name}")
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded_import)


@pytest.fixture
def mock_windows_no_msvc(monkeypatch):
    """Simulates Windows host missing MSVC/CUDA compilation toolchain for FlashAttention."""
    monkeypatch.setattr(sys, "platform", "win32")
    orig_import = __import__
    def guarded_import(name, *args, **kwargs):
        if name in ("flash_attn", "sageattention"):
            raise ImportError(f"No module named '{name}' (simulated missing MSVC toolchain on win32)")
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded_import)


@pytest.fixture
def mock_flash_attention_2(monkeypatch):
    """Simulates presence of FlashAttention-2."""
    fake_flash = types.ModuleType("flash_attn")
    fake_flash.__version__ = "2.5.8"
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash)
    return fake_flash


@pytest.fixture
def mock_sage_attention(monkeypatch):
    """Simulates presence of SageAttention."""
    fake_sage = types.ModuleType("sageattention")
    fake_sage.__version__ = "1.0.6"
    monkeypatch.setitem(sys.modules, "sageattention", fake_sage)
    return fake_sage


# ============================================================================
# Layer 3: MockMetaTensors Fixtures
# ============================================================================

@pytest.fixture
def mock_meta_device():
    """Provides meta-device tensor factories for zero-memory parameter modeling."""
    import torch
    class MetaTensorFactory:
        @staticmethod
        def allocate_linear(in_features: int, out_features: int, dtype=None):
            dtype = dtype or torch.get_default_dtype()
            weight = MagicMock()
            weight.shape = (out_features, in_features)
            weight.numel = lambda: in_features * out_features
            weight.dtype = dtype
            weight.device = "meta"
            return weight

        @staticmethod
        def allocate_dit_params(num_params: int, dtype=None):
            dtype = dtype or torch.get_default_dtype()
            param = MagicMock()
            param.numel = lambda: num_params
            param.dtype = dtype
            param.device = "meta"
            return param

    return MetaTensorFactory()


# ============================================================================
# Layer 4: MockMediaSynthesizer Fixtures
# ============================================================================

@pytest.fixture
def synthetic_exif_image(tmp_path):
    """Generates a test JPEG image with EXIF Tag 0x0112 = 6 (90 degrees CW rotation)."""
    img_path = tmp_path / "synthetic_rotated_tag6.jpg"
    img = Image.new("RGB", (100, 200), color=(255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # 90 degrees CW
    img.save(str(img_path), exif=exif)
    return str(img_path)


@pytest.fixture
def synthetic_exif_tag_factory(tmp_path):
    """Factory fixture generating images with any specified EXIF orientation tag (1-8)."""
    def _create(tag: int, width: int = 120, height: int = 80) -> str:
        filename = f"synthetic_exif_tag_{tag}.jpg"
        file_path = tmp_path / filename
        img = Image.new("RGB", (width, height), color=(10 * tag, 20 * tag, 30 * tag))
        exif = img.getexif()
        exif[0x0112] = tag
        img.save(str(file_path), exif=exif)
        return str(file_path)
    return _create


@pytest.fixture
def synthetic_valid_mp4(tmp_path):
    """Creates a synthetic binary MP4 container file exceeding 1024 bytes."""
    mp4_path = tmp_path / "synthetic_valid_video.mp4"
    # Write valid ISO Base Media File Format (isom) header box
    ftyp_data = b"ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    ftyp_box = struct.pack(">I", len(ftyp_data) + 4) + ftyp_data
    payload = b"\x00" * 2048  # Exceed 1024 bytes
    with open(str(mp4_path), "wb") as f:
        f.write(ftyp_box)
        f.write(payload)
    return str(mp4_path)


@pytest.fixture
def synthetic_corrupt_file(tmp_path):
    """Creates an empty 0-byte corrupt file."""
    corrupt_path = tmp_path / "corrupt_0byte.mp4"
    corrupt_path.touch()
    return str(corrupt_path)


@pytest.fixture
def mock_unwritable_directory(tmp_path, monkeypatch):
    """Creates a directory and mocks os.access to simulate read-only / permission-denied."""
    target_dir = tmp_path / "read_only_dir"
    target_dir.mkdir(exist_ok=True)
    orig_access = os.access
    def fake_access(path, mode, *args, **kwargs):
        if str(target_dir) in str(path) and (mode & os.W_OK):
            return False
        return orig_access(path, mode, *args, **kwargs)
    monkeypatch.setattr(os, "access", fake_access)
    return str(target_dir)


@pytest.fixture
def mock_ffmpeg_cascade(monkeypatch):
    """
    Simulates FFmpeg subprocess execution with controllable codec failure/success behavior.
    """
    def _configure(failing_codecs=None, zero_byte_codecs=None, make_empty_file=False):
        failing = set(failing_codecs or [])
        zero_byte = set(zero_byte_codecs or [])

        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None)

        def fake_popen(cmd, **kwargs):
            codec = None
            if "-c:v" in cmd:
                idx = cmd.index("-c:v")
                codec = cmd[idx + 1]
            out_file = cmd[-1]

            mock_proc = MagicMock()
            if codec in failing:
                mock_proc.returncode = 1
                mock_proc.communicate.return_value = (b"", f"Error: Codec {codec} failed to initialize".encode())
            elif codec in zero_byte or make_empty_file:
                mock_proc.returncode = 0
                with open(out_file, "wb") as f:
                    pass
                mock_proc.communicate.return_value = (b"", b"")
            else:
                mock_proc.returncode = 0
                with open(out_file, "wb") as f:
                    f.write(b"\x00" * 4096)
                mock_proc.communicate.return_value = (b"", b"")

            return mock_proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)

    return _configure


# ============================================================================
# Layer 5: Helper Fixtures for MCP and CLI
# ============================================================================

class MockMCPClient:
    """Helper client to invoke MCP server handlers via JSON-RPC 2.0 payloads."""
    def __init__(self, server_instance):
        self.server = server_instance

    def call_tool(self, name: str, arguments: dict, request_id: int = 1) -> dict:
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }
        return self.server.handle_request(req)

    def list_tools(self, request_id: int = 2) -> dict:
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {},
        }
        return self.server.handle_request(req)

    def initialize(self, request_id: int = 3) -> dict:
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }
        return self.server.handle_request(req)


@pytest.fixture
def mcp_client():
    """Provides a connected MockMCPClient instance."""
    def _create(server):
        return MockMCPClient(server)
    return _create
