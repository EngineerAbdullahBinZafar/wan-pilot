// Wan-Video Pilot Studio Workbench Engine
// Author: Engineer Abdullah Bin Zafar

document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.getElementById("modelSelect");
  const resolutionSelect = document.getElementById("resolutionSelect");
  const precisionSelect = document.getElementById("precisionSelect");
  const moeOffload = document.getElementById("moeOffload");
  const t5Cpu = document.getElementById("t5Cpu");
  const tiledVae = document.getElementById("tiledVae");
  const promptInput = document.getElementById("promptInput");
  const outputCode = document.getElementById("outputCode");
  const codeLangTag = document.getElementById("codeLangTag");
  const copyBtn = document.getElementById("copyBtn");
  const peakVramText = document.getElementById("peakVramText");
  const gpuFitBadge = document.getElementById("gpuFitBadge");
  const vramBar = document.getElementById("vramBar");
  const concurrentVramText = document.getElementById("concurrentVramText");
  const tabBtns = document.querySelectorAll(".tab-btn");

  let activeFormat = "cli_bash";

  // Mathematical VRAM calculator
  function recalculateVRAM() {
    const model = modelSelect.value;
    const res = resolutionSelect.value;
    const prec = precisionSelect.value;
    const isMoe = model.includes("moe");
    const is14b = model.includes("14b");

    let bp = 1.0;
    if (prec === "bf16") bp = 2.0;
    else if (prec === "int4") bp = 0.55;

    let activeParams = is14b ? 14.3e9 : 1.3e9;
    let totalParams = isMoe ? 27.0e9 : activeParams;

    let activeWeightsGb = (activeParams * bp) / (1024 ** 3);
    let totalWeightsGb = (totalParams * bp) / (1024 ** 3);

    // Latents calculation
    let w = 1280, h = 720, f = 81;
    if (res === "480p") { w = 832; h = 480; }
    else if (res === "1080p") { w = 1920; h = 1080; }

    let latentTokens = Math.floor(h / 16) * Math.floor(w / 16) * (1 + Math.floor((f - 1) / 4));
    let actKvGb = (latentTokens * 40 * 12 * 2 * 2) / (1024 ** 3); // Approx

    let vaeGb = tiledVae.checked ? 1.8 : ((125 * f * w * h) / (1024 ** 3));
    let t5Vram = t5Cpu.checked ? 0.0 : ((4.8e9 * 2) / (1024 ** 3));

    let peakVram = activeWeightsGb + actKvGb + vaeGb + t5Vram + 1.2;
    let concurrentPeak = totalWeightsGb + actKvGb + vaeGb + t5Vram + 2.5;

    peakVramText.textContent = `${peakVram.toFixed(2)} GB`;
    concurrentVramText.textContent = `${concurrentPeak.toFixed(1)} GB (Crash / OOM)`;

    let pct = Math.min(100, (peakVram / 24.0) * 100);
    vramBar.style.width = `${pct}%`;

    if (peakVram <= 24.0) {
      gpuFitBadge.textContent = "Fits 24GB GPU";
      gpuFitBadge.className = "badge badge-success";
      vramBar.style.background = "linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))";
    } else {
      gpuFitBadge.textContent = "Exceeds 24GB!";
      gpuFitBadge.className = "badge badge-danger";
      vramBar.style.background = "var(--accent-red)";
    }
  }

  function updateCodeOutput() {
    const prompt = promptInput.value;
    const res = resolutionSelect.value;
    let sizeStr = "1280*720";
    if (res === "480p") sizeStr = "832*480";
    if (res === "1080p") sizeStr = "1920*1080";

    if (activeFormat === "cli_bash") {
      codeLangTag.textContent = "BASH";
      outputCode.textContent = `#!/usr/bin/env bash
# Official Wan2.2 Execution Script (Bash)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python generate.py \\
  --task t2v-14B \\
  --size ${sizeStr} \\
  --prompt "${prompt}" \\
  --precision fp8 \\
  --expert_offload True \\
  --t5_cpu True \\
  --tiled_vae True`;
    } else if (activeFormat === "cli_powershell") {
      codeLangTag.textContent = "POWERSHELL";
      outputCode.textContent = `# Official Wan2.2 Execution Script (PowerShell)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python generate.py \`
  --task t2v-14B \`
  --size ${sizeStr} \`
  --prompt "${prompt}" \`
  --precision fp8 \`
  --expert_offload True \`
  --t5_cpu True \`
  --tiled_vae True`;
    } else if (activeFormat === "comfyui_api") {
      codeLangTag.textContent = "JSON";
      const payload = {
        "1": {"class_type": "WanVideoModelLoader", "inputs": {"model_name": "wan2.2_moe_14b", "precision": "fp8", "expert_offload": true}},
        "2": {"class_type": "WanTextEncode", "inputs": {"text": prompt, "model": ["1", 0]}},
        "3": {"class_type": "WanSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "steps": 30, "size": sizeStr}},
        "4": {"class_type": "WanVAEDecodeTiled", "inputs": {"samples": ["3", 0], "tiled": true}},
        "5": {"class_type": "WanVideoSave", "inputs": {"images": ["4", 0]}}
      };
      outputCode.textContent = JSON.stringify(payload, null, 2);
    } else if (activeFormat === "comfyui_canvas") {
      codeLangTag.textContent = "JSON";
      const canvas = {
        "version": 0.4,
        "nodes": [
          {"id": 1, "type": "WanVideoModelLoader", "pos": [40, 100]},
          {"id": 2, "type": "WanTextEncode", "pos": [380, 100], "widgets_values": [prompt]},
          {"id": 3, "type": "WanSampler", "pos": [820, 120]},
          {"id": 4, "type": "WanVAEDecodeTiled", "pos": [1180, 150]}
        ],
        "links": [[1, 1, 0, 3, 0], [2, 2, 0, 3, 1], [3, 3, 0, 4, 0]]
      };
      outputCode.textContent = JSON.stringify(canvas, null, 2);
    } else if (activeFormat === "diffusers") {
      codeLangTag.textContent = "PYTHON";
      outputCode.textContent = `import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16, device_map="auto")
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

output = pipe(
    prompt="${prompt}",
    height=${sizeStr.split('*')[1]},
    width=${sizeStr.split('*')[0]},
    num_frames=81,
    num_inference_steps=30,
)
export_to_video(output.frames[0], "output.mp4")`;
    }
  }

  // Tab switching
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeFormat = btn.dataset.target;
      updateCodeOutput();
    });
  });

  // Listeners
  modelSelect.addEventListener("change", recalculateVRAM);
  resolutionSelect.addEventListener("change", () => { recalculateVRAM(); updateCodeOutput(); });
  precisionSelect.addEventListener("change", recalculateVRAM);
  moeOffload.addEventListener("change", recalculateVRAM);
  t5Cpu.addEventListener("change", recalculateVRAM);
  tiledVae.addEventListener("change", recalculateVRAM);
  promptInput.addEventListener("input", updateCodeOutput);

  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(outputCode.textContent).then(() => {
      const orig = copyBtn.textContent;
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = orig; }, 1500);
    });
  });

  // Initial render
  recalculateVRAM();
  updateCodeOutput();
});
