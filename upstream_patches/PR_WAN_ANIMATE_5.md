# Pull Request: Guard Model Dtype During Meta Device Instantiation (Fixes Wan-Animate-2 #5)

## Description
Instantiating large DiT transformer blocks inside `torch.device("meta")` without constraining PyTorch's default tensor dtype resulted in non-persistent layers and weight initialization parameters reverting to `torch.float32`. This doubles parameter memory consumption upon moving tensors to GPU devices, triggering premature OOM on 24GB GPUs.

## Changes
- Wrapped `torch.device("meta")` instantiation in a thread-safe context that temporarily sets `torch.set_default_dtype(torch_dtype)`.
- Guaranteed restoration of the previous default dtype via a `finally:` block.

## Testing & Verification
- Verified on PyTorch 2.1+ that meta tensors instantiate with `bfloat16` and maintain memory footprint below 14.3 GB for the active DiT block.
