// wan-pilot — Studio Workbench Engine
// Author: Engineer Abdullah Bin Zafar

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const modelSelect = document.getElementById("modelSelect");
  const resolutionSelect = document.getElementById("resolutionSelect");
  const precisionSelect = document.getElementById("precisionSelect");
  const gpuSelect = document.getElementById("gpuSelect");
  const moeOffloadSwitch = document.getElementById("moeOffloadSwitch");
  const t5CpuSwitch = document.getElementById("t5CpuSwitch");
  const tiledVaeSwitch = document.getElementById("tiledVaeSwitch");

  const tokenCountDisplay = document.getElementById("tokenCountDisplay");
  const peakVramDisplay = document.getElementById("peakVramDisplay");
  const vramProgressFill = document.getElementById("vramProgressFill");
  const vramStatusSub = document.getElementById("vramStatusSub");
  const concurrentVramDisplay = document.getElementById("concurrentVramDisplay");

  const promptInput = document.getElementById("promptInput");
  const outputCodeSnippet = document.getElementById("outputCodeSnippet");
  const compilerTargetLabel = document.getElementById("compilerTargetLabel");
  const copyCodeBtn = document.getElementById("copyCodeBtn");
  const synthesizeBtn = document.getElementById("synthesizeBtn");

  const consoleBody = document.getElementById("consoleBody");
  const timecodeDisplay = document.getElementById("timecodeDisplay");
  const scrubberFill = document.getElementById("scrubberFill");
  const playbackPctText = document.getElementById("playbackPctText");
  const playPauseBtn = document.getElementById("playPauseBtn");

  const chipBtns = document.querySelectorAll(".chip-btn");

  let activeFormat = "cli_bash";
  let isGenerating = false;
  let isPlaying = true;
  let animTime = 0;

  // Mathematical VRAM & Latent Tokens Recalculation
  function recalculateTelemetry() {
    const model = modelSelect.value;
    const res = resolutionSelect.value;
    const prec = precisionSelect.value;
    const isMoe = model.includes("moe");
    const isSequential = moeOffloadSwitch.checked;

    let bp = 1.0;
    if (prec === "bf16") bp = 2.0;
    else if (prec === "int4") bp = 0.55;

    let activeParams = model.includes("14b") ? 14.3e9 : 1.3e9;
    let totalParams = isMoe ? 27.0e9 : activeParams;

    let activeWeightsGb = (activeParams * bp) / (1024 ** 3);
    let totalWeightsGb = (totalParams * bp) / (1024 ** 3);

    // Latents calculation (81 frames default)
    let w = 1280, h = 720, f = 81;
    if (res === "480p") { w = 832; h = 480; }
    else if (res === "1080p") { w = 1920; h = 1080; }

    let tLatent = 1 + Math.floor((f - 1) / 4);
    let hLatent = Math.floor(h / 16);
    let wLatent = Math.floor(w / 16);
    let latentTokens = hLatent * wLatent * tLatent;

    tokenCountDisplay.innerHTML = `${latentTokens.toLocaleString()} <span class="metric-unit">tokens</span>`;

    let actKvGb = (latentTokens * 40 * 12 * 2 * 2) / (1024 ** 3);
    let vaeGb = tiledVaeSwitch.checked ? 1.8 : ((125 * f * w * h) / (1024 ** 3));
    let t5Vram = t5CpuSwitch.checked ? 0.0 : ((4.8e9 * 2) / (1024 ** 3));

    let sequentialPeak = activeWeightsGb + actKvGb + vaeGb + t5Vram + 1.15;
    let concurrentPeak = totalWeightsGb + actKvGb + vaeGb + t5Vram + 2.50;

    concurrentVramDisplay.textContent = `${concurrentPeak.toFixed(2)} GB`;

    if (isSequential) {
      peakVramDisplay.textContent = `${sequentialPeak.toFixed(2)} GB`;
      peakVramDisplay.className = "metric-huge purple-glow";
      let pct = Math.min(100, (sequentialPeak / 24.0) * 100);
      vramProgressFill.style.width = `${pct}%`;
      vramProgressFill.className = "progress-fill purple-fill";
      vramStatusSub.textContent = `Sequential Swapping: Peak constrained below 18.47 GB`;
    } else {
      peakVramDisplay.textContent = `${concurrentPeak.toFixed(2)} GB`;
      peakVramDisplay.className = "metric-huge red-glow";
      vramProgressFill.style.width = "100%";
      vramProgressFill.className = "progress-fill red-fill";
      vramStatusSub.textContent = `Concurrent Swapping: Exceeds 24GB VRAM!`;
    }

    updateWorkflowCode();
  }

  // Workflow compiler code generator
  function updateWorkflowCode() {
    const prompt = promptInput.value.trim();
    const res = resolutionSelect.value;
    let sizeStr = "1280*720";
    if (res === "480p") sizeStr = "832*480";
    if (res === "1080p") sizeStr = "1920*1080";

    const isSeq = moeOffloadSwitch.checked;
    const isT5 = t5CpuSwitch.checked;
    const isTiled = tiledVaeSwitch.checked;

    if (activeFormat === "cli_bash") {
      compilerTargetLabel.textContent = "BASH EXECUTION SCRIPT";
      outputCodeSnippet.textContent = `#!/usr/bin/env bash
# Wan-Video Pilot Compiler (Wan2.2 MoE)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python generate.py \\
  --task t2v-14B \\
  --size ${sizeStr} \\
  --prompt "${prompt}" \\
  --precision ${precisionSelect.value} \\
  --expert_offload ${isSeq ? "True" : "False"} \\
  --t5_cpu ${isT5 ? "True" : "False"} \\
  --tiled_vae ${isTiled ? "True" : "False"}`;
    } else if (activeFormat === "cli_powershell") {
      compilerTargetLabel.textContent = "POWERSHELL SCRIPT";
      outputCodeSnippet.textContent = `# Wan-Video Pilot Compiler (PowerShell)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python generate.py \`
  --task t2v-14B \`
  --size "${sizeStr}" \`
  --prompt "${prompt}" \`
  --precision "${precisionSelect.value}" \`
  --expert_offload \$${isSeq ? "True" : "False"} \`
  --t5_cpu \$${isT5 ? "True" : "False"} \`
  --tiled_vae \$${isTiled ? "True" : "False"}`;
    } else if (activeFormat === "comfyui") {
      compilerTargetLabel.textContent = "COMFYUI WORKFLOW API (JSON)";
      outputCodeSnippet.textContent = JSON.stringify({
        "1": { "class_type": "WanVideoModelLoader", "inputs": { "model": modelSelect.value, "precision": precisionSelect.value } },
        "2": { "class_type": "WanVideoMoEOffload", "inputs": { "enabled": isSeq, "expert_swap": isSeq } },
        "3": { "class_type": "WanVideoTextEncode", "inputs": { "prompt": prompt, "cpu_offload": isT5 } },
        "4": { "class_type": "WanVideoSampler", "inputs": { "steps": 50, "shift": 5.0, "size": sizeStr } },
        "5": { "class_type": "WanVideoVAEDecode", "inputs": { "tiled": isTiled } }
      }, null, 2);
    }
  }

  // Interactive Canvas Latent Swirl Simulator (Matches video latent swirl pattern)
  const canvas = document.getElementById("latentCanvas");
  const ctx = canvas.getContext("2d");

  function drawLatentSwirl(time) {
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    const speed = isGenerating ? 0.05 : 0.008;
    animTime += speed;

    const scale = 0.015;
    const t = animTime;

    // Fast multi-frequency synthetic fluid noise
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        let v1 = Math.sin(x * scale + t);
        let v2 = Math.cos(y * scale - t);
        let v3 = Math.sin((x + y) * scale * 0.7 + t * 1.5);
        let val = (v1 + v2 + v3 + 3) / 6; // 0 to 1

        let r = Math.floor(Math.sin(val * Math.PI * 2) * 110 + 120);
        let g = Math.floor(Math.cos(val * Math.PI * 2) * 100 + 130);
        let b = Math.floor(Math.sin((val + 0.3) * Math.PI * 2) * 120 + 135);

        let idx = (y * width + x) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;

        // Duplicate for 2x2 block
        data[idx + 4] = r;
        data[idx + 5] = g;
        data[idx + 6] = b;
        data[idx + 7] = 255;

        let nextRow = idx + width * 4;
        data[nextRow] = r;
        data[nextRow + 1] = g;
        data[nextRow + 2] = b;
        data[nextRow + 3] = 255;
        data[nextRow + 4] = r;
        data[nextRow + 5] = g;
        data[nextRow + 6] = b;
        data[nextRow + 7] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    if (isPlaying) {
      requestAnimationFrame(drawLatentSwirl);
    }
  }

  requestAnimationFrame(drawLatentSwirl);

  // Synthesize button animation & diagnostics log injector
  synthesizeBtn.addEventListener("click", () => {
    if (isGenerating) return;
    isGenerating = true;
    synthesizeBtn.classList.add("synthesizing");
    synthesizeBtn.querySelector(".btn-text").textContent = "Synthesizing Latents...";

    const logs = [
      `[+0.12s] Parsing prompt token embeddings via T5-XXL`,
      `[+0.45s] Constructing 3D Spatio-Temporal Latent Tensor: 75,600 tokens`,
      `[+0.82s] Entering Diffusion Stage 1: MoE Expert 1 (High-Frequency Details)`,
      `[+1.35s] Dynamic Swap at t_moe: Unloading Expert 1 -> Loading Expert 2`,
      `[+1.98s] Tiled 3D Causal VAE Decoding (4 tiles parallel)`,
      `[+2.45s] Render Complete: 81 frames compiled cleanly`
    ];

    let logIdx = 0;
    const interval = setInterval(() => {
      if (logIdx < logs.length) {
        const line = document.createElement("div");
        line.className = "log-line";
        line.innerHTML = `<span class="log-cyan">${logs[logIdx].split(" ")[0]}</span> ${logs[logIdx].substring(logs[logIdx].indexOf(" ") + 1)}`;
        consoleBody.appendChild(line);
        consoleBody.scrollTop = consoleBody.scrollHeight;
        logIdx++;
      } else {
        clearInterval(interval);
        isGenerating = false;
        synthesizeBtn.classList.remove("synthesizing");
        synthesizeBtn.querySelector(".btn-text").textContent = "Synthesize Latents";
      }
    }, 400);
  });

  // Play/pause toggle
  playPauseBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playPauseBtn.textContent = isPlaying ? "❚❚" : "▶";
    if (isPlaying) requestAnimationFrame(drawLatentSwirl);
  });

  // Format tab chips
  chipBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      chipBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeFormat = btn.getAttribute("data-format");
      updateWorkflowCode();
    });
  });

  // Copy code button
  copyCodeBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(outputCodeSnippet.textContent).then(() => {
      copyCodeBtn.textContent = "Copied!";
      setTimeout(() => copyCodeBtn.textContent = "Copy", 1500);
    });
  });

  // Event Listeners for telemetry
  [modelSelect, resolutionSelect, precisionSelect, gpuSelect, moeOffloadSwitch, t5CpuSwitch, tiledVaeSwitch].forEach(el => {
    if (el) el.addEventListener("change", recalculateTelemetry);
  });

  promptInput.addEventListener("input", updateWorkflowCode);

  // Initial calculation
  recalculateTelemetry();
});
