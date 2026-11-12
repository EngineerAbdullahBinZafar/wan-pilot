// wan-pilot — Studio Workbench Engine (Flagship Edition)
// Author: Engineer Abdullah Bin Zafar

document.addEventListener("DOMContentLoaded", () => {
  // Mode Switcher Elements
  const modeStudioBtn = document.getElementById("modeStudioBtn");
  const modeDevBtn = document.getElementById("modeDevBtn");

  // Prompt & Preset Elements
  const promptInput = document.getElementById("promptInput");
  const presetChips = document.querySelectorAll(".preset-chip");
  const synthesizeBtn = document.getElementById("synthesizeBtn");

  // Viewport & Scrubber Elements
  const latentCanvas = document.getElementById("latentCanvas");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const scrubberTrack = document.getElementById("scrubberTrack");
  const scrubberFill = document.getElementById("scrubberFill");
  const playbackPctText = document.getElementById("playbackPctText");
  const timecodeDisplay = document.getElementById("timecodeDisplay");
  const downloadMp4Btn = document.getElementById("downloadMp4Btn");
  const previewResBadge = document.getElementById("previewResBadge");

  // Settings & Ratio Elements
  const aspectBtns = document.querySelectorAll(".aspect-btn");
  const qualityPresetSelect = document.getElementById("qualityPresetSelect");

  // GPU & VRAM Elements
  const unifiedVramReadout = document.getElementById("unifiedVramReadout");
  const vramSafeSegment = document.getElementById("vramSafeSegment");
  const vramHeadroomLabel = document.getElementById("vramHeadroomLabel");

  // Dev Optimization Elements
  const moeOffloadSwitch = document.getElementById("moeOffloadSwitch");
  const t5CpuSwitch = document.getElementById("t5CpuSwitch");
  const tiledVaeSwitch = document.getElementById("tiledVaeSwitch");
  const modelSelect = document.getElementById("modelSelect");
  const resolutionSelect = document.getElementById("resolutionSelect");
  const precisionSelect = document.getElementById("precisionSelect");
  const tokenCountDisplay = document.getElementById("tokenCountDisplay");

  // Compiler & Logs Elements
  const compTabs = document.querySelectorAll(".comp-tab");
  const outputCodeSnippet = document.getElementById("outputCodeSnippet");
  const copyCodeBtn = document.getElementById("copyCodeBtn");
  const copyBtnText = document.getElementById("copyBtnText");
  const consoleBody = document.getElementById("consoleBody");

  // State
  let activeFormat = "cli_bash";
  let activeRatio = "16:9";
  let isGenerating = false;
  let isPlaying = true;
  let animTime = 0;
  let scrubPct = 75;

  // 1. Dual Mode Switching Logic
  modeStudioBtn.addEventListener("click", () => {
    modeStudioBtn.classList.add("active");
    modeDevBtn.classList.remove("active");
    document.body.classList.remove("mode-dev");
    document.body.classList.add("mode-studio");
  });

  modeDevBtn.addEventListener("click", () => {
    modeDevBtn.classList.add("active");
    modeStudioBtn.classList.remove("active");
    document.body.classList.remove("mode-studio");
    document.body.classList.add("mode-dev");
  });

  // 2. Preset Prompt Chips
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const promptText = chip.getAttribute("data-prompt");
      if (promptText) {
        promptInput.value = promptText;
        updateCompilerOutput();
      }
    });
  });

  // 3. Aspect Ratio Selection
  aspectBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      aspectBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeRatio = btn.getAttribute("data-ratio");
      if (activeRatio === "16:9") {
        resolutionSelect.value = "720p";
      } else if (activeRatio === "9:16") {
        resolutionSelect.value = "720p";
      }
      previewResBadge.textContent = `${activeRatio} • ${resolutionSelect.value}`;
      recalculateTelemetry();
    });
  });

  // Quality Preset synchronization
  qualityPresetSelect.addEventListener("change", () => {
    const val = qualityPresetSelect.value;
    if (val === "cinematic") {
      resolutionSelect.value = "720p";
      precisionSelect.value = "fp8";
    } else if (val === "balanced") {
      resolutionSelect.value = "720p";
      precisionSelect.value = "fp8";
    } else if (val === "fast") {
      resolutionSelect.value = "480p";
      precisionSelect.value = "int4";
    }
    recalculateTelemetry();
  });

  // 4. Mathematical VRAM & Telemetry Calculation
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

    tokenCountDisplay.textContent = `${latentTokens.toLocaleString()} tokens`;

    let actKvGb = (latentTokens * 40 * 12 * 2 * 2) / (1024 ** 3);
    let vaeGb = tiledVaeSwitch.checked ? 1.8 : ((125 * f * w * h) / (1024 ** 3));
    let t5Vram = t5CpuSwitch.checked ? 0.0 : ((4.8e9 * 2) / (1024 ** 3));

    let sequentialPeak = activeWeightsGb + actKvGb + vaeGb + t5Vram + 1.15;
    let concurrentPeak = totalWeightsGb + actKvGb + vaeGb + t5Vram + 2.50;

    let currentPeak = isSequential ? sequentialPeak : concurrentPeak;
    let maxVram = 24.0;
    let pct = Math.min(100, (currentPeak / maxVram) * 100);

    unifiedVramReadout.textContent = `${currentPeak.toFixed(2)} GB / ${maxVram.toFixed(2)} GB`;
    vramSafeSegment.style.width = `${pct}%`;

    let headroom = Math.max(0, maxVram - currentPeak);

    if (currentPeak <= maxVram) {
      unifiedVramReadout.className = "gpu-readout";
      vramSafeSegment.className = "vram-segment safe-segment";
      vramHeadroomLabel.textContent = `${headroom.toFixed(2)} GB Free Headroom (Safe)`;
    } else {
      unifiedVramReadout.className = "gpu-readout danger";
      vramSafeSegment.className = "vram-segment warning-segment";
      vramHeadroomLabel.textContent = `Exceeds 24GB! Requires 80GB H100 GPU`;
    }

    updateCompilerOutput();
  }

  // 5. Syntax-Highlighted Compiler Output
  function updateCompilerOutput() {
    const prompt = promptInput.value.trim();
    const res = resolutionSelect.value;
    let sizeStr = "1280*720";
    if (res === "480p") sizeStr = "832*480";
    if (res === "1080p") sizeStr = "1920*1080";

    const isSeq = moeOffloadSwitch.checked;
    const isT5 = t5CpuSwitch.checked;
    const isTiled = tiledVaeSwitch.checked;
    const prec = precisionSelect.value;

    if (activeFormat === "cli_bash") {
      outputCodeSnippet.innerHTML = 
`<span class="code-comment">#!/usr/bin/env bash</span>
<span class="code-comment"># Wan-Video Pilot Flagship Execution Pipeline (Wan2.2 MoE)</span>
<span class="code-command">export</span> CUDA_VISIBLE_DEVICES=<span class="code-val">0</span>
<span class="code-command">export</span> PYTORCH_CUDA_ALLOC_CONF=<span class="code-string">"expandable_segments:True"</span>

<span class="code-command">python</span> generate.py \\
  <span class="code-flag">--task</span> t2v-14B \\
  <span class="code-flag">--size</span> ${sizeStr} \\
  <span class="code-flag">--prompt</span> <span class="code-string">"${escapeHtml(prompt)}"</span> \\
  <span class="code-flag">--precision</span> ${prec} \\
  <span class="code-flag">--expert_offload</span> <span class="code-val">${isSeq ? "True" : "False"}</span> \\
  <span class="code-flag">--t5_cpu</span> <span class="code-val">${isT5 ? "True" : "False"}</span> \\
  <span class="code-flag">--tiled_vae</span> <span class="code-val">${isTiled ? "True" : "False"}</span>`;
    } else if (activeFormat === "cli_powershell") {
      outputCodeSnippet.innerHTML = 
`<span class="code-comment"># Wan-Video Pilot Flagship Execution Pipeline (PowerShell)</span>
$env:CUDA_VISIBLE_DEVICES = <span class="code-string">"0"</span>
$env:PYTORCH_CUDA_ALLOC_CONF = <span class="code-string">"expandable_segments:True"</span>

<span class="code-command">python</span> generate.py \`
  <span class="code-flag">--task</span> t2v-14B \`
  <span class="code-flag">--size</span> <span class="code-string">"${sizeStr}"</span> \`
  <span class="code-flag">--prompt</span> <span class="code-string">"${escapeHtml(prompt)}"</span> \`
  <span class="code-flag">--precision</span> <span class="code-string">"${prec}"</span> \`
  <span class="code-flag">--expert_offload</span> $<span class="code-val">${isSeq ? "True" : "False"}</span> \`
  <span class="code-flag">--t5_cpu</span> $<span class="code-val">${isT5 ? "True" : "False"}</span> \`
  <span class="code-flag">--tiled_vae</span> $<span class="code-val">${isTiled ? "True" : "False"}</span>`;
    } else if (activeFormat === "comfyui") {
      const comfyJson = {
        "1": { "class_type": "WanVideoModelLoader", "inputs": { "model": modelSelect.value, "precision": prec } },
        "2": { "class_type": "WanVideoMoEOffload", "inputs": { "enabled": isSeq, "expert_swap": isSeq } },
        "3": { "class_type": "WanVideoTextEncode", "inputs": { "prompt": prompt, "cpu_offload": isT5 } },
        "4": { "class_type": "WanVideoSampler", "inputs": { "steps": 50, "shift": 5.0, "size": sizeStr } },
        "5": { "class_type": "WanVideoVAEDecode", "inputs": { "tiled": isTiled } }
      };
      outputCodeSnippet.textContent = JSON.stringify(comfyJson, null, 2);
    }
  }

  function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // 6. Interactive Canvas Fluid Latent Swirl Simulator
  const ctx = latentCanvas.getContext("2d");

  function drawLatentSwirl() {
    if (!ctx) return;
    const width = latentCanvas.width;
    const height = latentCanvas.height;

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    const speed = isGenerating ? 0.06 : 0.012;
    animTime += speed;

    const scale = 0.012;
    const t = animTime;

    // Fast multi-frequency fluid noise
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        let v1 = Math.sin(x * scale + t);
        let v2 = Math.cos(y * scale - t);
        let v3 = Math.sin((x + y) * scale * 0.7 + t * 1.5);
        let val = (v1 + v2 + v3 + 3) / 6;

        let r = Math.floor(Math.sin(val * Math.PI * 2) * 110 + 120);
        let g = Math.floor(Math.cos(val * Math.PI * 2) * 100 + 130);
        let b = Math.floor(Math.sin((val + 0.3) * Math.PI * 2) * 120 + 135);

        let idx = (y * width + x) * 4;
        data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = 255;
        data[idx + 4] = r; data[idx + 5] = g; data[idx + 6] = b; data[idx + 7] = 255;

        let nextRow = idx + width * 4;
        data[nextRow] = r; data[nextRow + 1] = g; data[nextRow + 2] = b; data[nextRow + 3] = 255;
        data[nextRow + 4] = r; data[nextRow + 5] = g; data[nextRow + 6] = b; data[nextRow + 7] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    if (isPlaying) {
      requestAnimationFrame(drawLatentSwirl);
    }
  }

  requestAnimationFrame(drawLatentSwirl);

  // 7. Synthesize Latents Execution Cycle
  synthesizeBtn.addEventListener("click", () => {
    if (isGenerating) return;
    isGenerating = true;
    synthesizeBtn.classList.add("synthesizing");
    synthesizeBtn.innerHTML = `<span class="btn-icon">⚡</span><span class="btn-text">Synthesizing Latents...</span>`;

    const logs = [
      `[+0.12s] Parsing prompt embeddings via T5-XXL host memory`,
      `[+0.45s] Constructing 3D Spatio-Temporal Latent Tensor: 75,600 tokens`,
      `[+0.82s] Diffusion Pass 1: MoE Expert 1 (High-Frequency Details)`,
      `[+1.35s] Dynamic Swap at t_moe: Unloading Expert 1 -> Loading Expert 2`,
      `[+1.98s] Tiled 3D Causal VAE Decoding (4 tiles parallel)`,
      `[+2.45s] Synthesis Complete: 81 frames compiled cleanly`
    ];

    let logIdx = 0;
    const interval = setInterval(() => {
      if (logIdx < logs.length) {
        const line = document.createElement("div");
        line.className = "term-line";
        const parts = logs[logIdx].split(" ");
        line.innerHTML = `<span class="t-cyan">${parts[0]}</span> ${logs[logIdx].substring(parts[0].length + 1)}`;
        consoleBody.appendChild(line);
        consoleBody.scrollTop = consoleBody.scrollHeight;
        logIdx++;
      } else {
        clearInterval(interval);
        isGenerating = false;
        synthesizeBtn.classList.remove("synthesizing");
        synthesizeBtn.innerHTML = `<span class="btn-icon">▶</span><span class="btn-text">Synthesize Latents</span>`;
      }
    }, 380);
  });

  // Play/Pause transport toggle
  playPauseBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playPauseBtn.textContent = isPlaying ? "❚❚" : "▶";
    if (isPlaying) requestAnimationFrame(drawLatentSwirl);
  });

  // Scrubber click handler
  scrubberTrack.addEventListener("click", (e) => {
    const rect = scrubberTrack.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    scrubPct = Math.round((clickX / rect.width) * 100);
    scrubberFill.style.width = `${scrubPct}%`;
    playbackPctText.textContent = `${scrubPct}%`;
  });

  // Download MP4 stub
  downloadMp4Btn.addEventListener("click", () => {
    downloadMp4Btn.innerHTML = `<span>Preparing MP4...</span>`;
    setTimeout(() => {
      downloadMp4Btn.innerHTML = `<span>Downloaded!</span>`;
      setTimeout(() => {
        downloadMp4Btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
          <span>Download MP4</span>`;
      }, 1500);
    }, 800);
  });

  // Compiler tabs
  compTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      compTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeFormat = tab.getAttribute("data-format");
      updateCompilerOutput();
    });
  });

  // Copy code with checkmark confirmation
  copyCodeBtn.addEventListener("click", () => {
    const textToCopy = outputCodeSnippet.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
      copyBtnText.textContent = "Copied!";
      copyCodeBtn.classList.add("copied");
      setTimeout(() => {
        copyBtnText.textContent = "Copy";
        copyCodeBtn.classList.remove("copied");
      }, 1800);
    });
  });

  // Reactive inputs
  [modelSelect, resolutionSelect, precisionSelect, moeOffloadSwitch, t5CpuSwitch, tiledVaeSwitch].forEach(el => {
    if (el) el.addEventListener("change", recalculateTelemetry);
  });

  promptInput.addEventListener("input", updateCompilerOutput);

  // Initial render
  recalculateTelemetry();
});
