// RB-UNet Research Workstation Orchestrator
// Supports both Static Showcase Deployment (Vercel / GitHub Pages) and Live ML Backend (Local / Render)

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // PATH & API RESOLUTION HELPERS
  // =========================================================================
  function getAssetPath(path) {
    if (!path) return "";
    const clean = path.replace(/^\/+/, "");
    const isSubDir = window.location.pathname.includes("/ui/") || window.location.pathname.endsWith("/ui");
    return (isSubDir ? "../" : "./") + clean;
  }

  function getApiBase() {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const paramApi = urlParams.get("api");
      if (paramApi) {
        localStorage.setItem("rb_unet_api_url", paramApi.replace(/\/+$/, ""));
        return paramApi.replace(/\/+$/, "");
      }
    } catch (e) {}

    try {
      const stored = localStorage.getItem("rb_unet_api_url");
      if (stored) return stored.replace(/\/+$/, "");
    } catch (e) {}

    if (window.VITE_API_URL) return window.VITE_API_URL.replace(/\/+$/, "");
    if (window.BACKEND_API_URL) return window.BACKEND_API_URL.replace(/\/+$/, "");
    return "";
  }

  function apiUrl(endpoint) {
    const base = getApiBase();
    const cleanEp = endpoint.startsWith("/") ? endpoint : "/" + endpoint;
    return base ? `${base}${cleanEp}` : cleanEp;
  }

  // =========================================================================
  // NAVIGATION TABS
  // =========================================================================
  const navTabs = document.querySelectorAll(".nav-tab");
  const sections = document.querySelectorAll(".page-section");

  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-target");

      navTabs.forEach(t => t.classList.remove("active"));
      sections.forEach(s => s.classList.remove("active"));

      tab.classList.add("active");
      const targetSection = document.getElementById(targetId);
      if (targetSection) targetSection.classList.add("active");

      if (targetId === "section-robustness") {
        fetchResearchResults();
        if (!robustnessHasRun) {
          runDefaultLiveRobustness();
        }
      } else if (targetId === "section-results") {
        fetchResearchResults();
      }
    });
  });

  // =========================================================================
  // SYSTEM STATUS & BACKEND SWITCHER
  // =========================================================================
  let isLiveBackend = false;
  const statusIndicatorDot = document.getElementById("status-indicator-dot");
  const systemStatusText = document.getElementById("system-status-text");
  const navStatus = document.querySelector(".nav-status");

  if (navStatus) {
    navStatus.style.cursor = "pointer";
    navStatus.title = "Click to configure Live Python Backend URL (e.g. Render / Localhost)";
    navStatus.addEventListener("click", () => {
      const current = getApiBase() || "";
      const userUrl = prompt(
        "RB-UNet Backend Connection:\n\n" +
        "• Enter Python Backend API URL (e.g., https://your-backend.onrender.com or http://localhost:8000)\n" +
        "• Leave empty to use Showcase Mode / same-origin.",
        current
      );
      if (userUrl !== null) {
        if (userUrl.trim()) {
          localStorage.setItem("rb_unet_api_url", userUrl.trim().replace(/\/+$/, ""));
        } else {
          localStorage.removeItem("rb_unet_api_url");
        }
        if (systemStatusText) systemStatusText.textContent = "Connecting...";
        checkSystemStatus();
        researchDataLoaded = false;
        cachedResults = null;
        fetchResearchResults();
      }
    });
  }

  async function checkSystemStatus() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const resp = await fetch(apiUrl("/api/status"), { signal: controller.signal });
      clearTimeout(timeoutId);
      if (resp.ok) {
        const data = await resp.json();
        if (data.best_checkpoint_exists && data.baseline_checkpoint_exists) {
          isLiveBackend = true;
          if (systemStatusText) systemStatusText.textContent = "Live ML Ready";
          if (statusIndicatorDot) {
            statusIndicatorDot.style.backgroundColor = "var(--accent-green)";
            statusIndicatorDot.style.boxShadow = "0 0 6px rgba(16, 185, 129, 0.5)";
          }
          return;
        } else if (data.best_checkpoint_exists) {
          isLiveBackend = true;
          if (systemStatusText) systemStatusText.textContent = "RB-UNet Ready";
          if (statusIndicatorDot) {
            statusIndicatorDot.style.backgroundColor = "var(--accent-green)";
          }
          return;
        }
      }
    } catch (e) {
      // Backend not running / static deployment
    }

    // Showcase Demo Mode fallback
    isLiveBackend = false;
    if (systemStatusText) systemStatusText.textContent = "Showcase Demo Mode";
    if (statusIndicatorDot) {
      statusIndicatorDot.style.backgroundColor = "var(--accent-cyan)";
      statusIndicatorDot.style.boxShadow = "0 0 6px rgba(14, 165, 233, 0.5)";
    }
  }
  checkSystemStatus();

  // =========================================================================
  // STATE MANAGEMENT
  // =========================================================================
  let currentImageBase64 = null;
  let currentPresetName = null;
  let precomputedPresets = null;
  let activeCondition = "Clean";
  let robustnessHasRun = false;

  // Live Segmentation Elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnSegment = document.getElementById("btn-segment");
  const segmentSpinner = document.getElementById("segment-spinner");
  const btnText = btnSegment ? btnSegment.querySelector(".btn-text") : null;

  const resultsCard = document.getElementById("results-card");
  const emptyState = document.getElementById("empty-state");
  const visualContainer = document.getElementById("results-visual-container");
  const stateTag = document.getElementById("inference-state-tag");

  const dimOriginal = document.getElementById("dim-original");
  const imgOriginal = document.getElementById("img-original");
  const imgMask = document.getElementById("img-mask");
  const imgOverlay = document.getElementById("img-overlay");
  const imgContour = document.getElementById("img-contour");
  const imgBoundary = document.getElementById("img-boundary");

  const mainPanelsGrid = document.getElementById("main-panels-grid");
  const panelOriginal = document.getElementById("panel-original");
  const panelMask = document.getElementById("panel-mask");
  const panelOverlay = document.getElementById("panel-overlay");
  const panelContour = document.getElementById("panel-contour");

  const valLesionArea = document.getElementById("val-lesion-area");
  const valInferTime = document.getElementById("val-infer-time");
  const valMeanFgProb = document.getElementById("val-mean-fg-prob");
  const valOperatingThTag = document.getElementById("val-operating-th-tag");
  const valOperatingThDisplay = document.getElementById("val-operating-th-display");

  // Segmented Viewport Mode Switching
  const viewModeBtns = document.querySelectorAll("#view-mode-selector .seg-btn");
  viewModeBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      viewModeBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.getAttribute("data-mode");
      applyViewMode(mode);
    });
  });

  function applyViewMode(mode) {
    if (!mainPanelsGrid) return;
    if (mode === "tri-panel") {
      mainPanelsGrid.style.gridTemplateColumns = "repeat(3, 1fr)";
      if (panelOriginal) panelOriginal.style.display = "flex";
      if (panelMask) panelMask.style.display = "flex";
      if (panelOverlay) panelOverlay.style.display = "flex";
      if (panelContour) panelContour.style.display = "none";
    } else if (mode === "contour") {
      mainPanelsGrid.style.gridTemplateColumns = "repeat(2, 1fr)";
      if (panelOriginal) panelOriginal.style.display = "flex";
      if (panelMask) panelMask.style.display = "none";
      if (panelOverlay) panelOverlay.style.display = "none";
      if (panelContour) panelContour.style.display = "flex";
    } else if (mode === "overlay") {
      mainPanelsGrid.style.gridTemplateColumns = "repeat(2, 1fr)";
      if (panelOriginal) panelOriginal.style.display = "flex";
      if (panelMask) panelMask.style.display = "none";
      if (panelOverlay) panelOverlay.style.display = "flex";
      if (panelContour) panelContour.style.display = "none";
    } else if (mode === "mask") {
      mainPanelsGrid.style.gridTemplateColumns = "repeat(2, 1fr)";
      if (panelOriginal) panelOriginal.style.display = "flex";
      if (panelMask) panelMask.style.display = "flex";
      if (panelOverlay) panelOverlay.style.display = "none";
      if (panelContour) panelContour.style.display = "none";
    }
  }

  // Auxiliary Boundary Drawer Toggle
  const boundaryToggle = document.getElementById("boundary-toggle");
  const boundaryDrawerContent = document.getElementById("boundary-drawer-content");
  if (boundaryToggle && boundaryDrawerContent) {
    boundaryToggle.addEventListener("click", () => {
      boundaryDrawerContent.classList.toggle("open");
      const arrow = boundaryToggle.querySelector(".drawer-arrow");
      if (arrow) arrow.textContent = boundaryDrawerContent.classList.contains("open") ? "▲" : "▼";
    });
  }

  // Safe image display helper
  function displayImage(imgElement, srcUrl) {
    if (!imgElement) return;
    imgElement.style.display = "block";
    imgElement.src = srcUrl;
    const parent = imgElement.parentElement;
    if (parent) {
      const fallback = parent.querySelector(".img-fallback-msg");
      if (fallback) fallback.remove();
    }
  }

  // Drop zone events
  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("hover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("hover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("hover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });
  }

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid JPG or PNG dermoscopy image file.");
      return;
    }
    currentPresetName = null; // Custom user file
    const reader = new FileReader();
    reader.onload = (event) => {
      currentImageBase64 = event.target.result;
      if (btnSegment) btnSegment.disabled = false;
      if (stateTag) stateTag.textContent = "Image Loaded";

      if (emptyState) emptyState.style.display = "none";
      if (visualContainer) visualContainer.style.display = "block";
      displayImage(imgOriginal, currentImageBase64);
      if (imgMask) imgMask.src = "";
      if (imgOverlay) imgOverlay.src = "";
      if (imgContour) imgContour.src = "";
      if (imgBoundary) imgBoundary.src = "";
      if (valLesionArea) valLesionArea.textContent = "--%";
      if (valInferTime) valInferTime.textContent = "-- ms";
      if (valMeanFgProb) valMeanFgProb.textContent = "--";
      if (dimOriginal) dimOriginal.textContent = "PENDING INFERENCE";
    };
    reader.readAsDataURL(file);
  }

  // Quick Preset Handlers (Real ISIC Dataset Images)
  const presetButtons = document.querySelectorAll(".btn-preset");
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const presetType = btn.getAttribute("data-preset");
      loadPresetImage(presetType);
    });
  });

  async function loadPresetImage(presetName) {
    try {
      currentPresetName = presetName;
      if (stateTag) stateTag.textContent = "Loading Preset...";
      const imgPath = getAssetPath(`outputs/sample_images/${presetName}.jpg`);
      const resp = await fetch(imgPath);
      if (!resp.ok) {
        throw new Error(`Preset file not found: ${presetName}.jpg`);
      }
      const blob = await resp.blob();
      const reader = new FileReader();
      reader.onload = async (e) => {
        currentImageBase64 = e.target.result;
        if (btnSegment) btnSegment.disabled = false;

        if (emptyState) emptyState.style.display = "none";
        if (visualContainer) visualContainer.style.display = "block";
        displayImage(imgOriginal, currentImageBase64);

        // Automatically trigger segmentation
        await executeLiveInference();
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error("Error loading preset:", err);
      if (stateTag) stateTag.textContent = "Preset Error";
    }
  }

  // Trigger Live Segmentation Inference
  if (btnSegment) {
    btnSegment.addEventListener("click", executeLiveInference);
  }

  function renderInferenceSuccess(data) {
    displayImage(imgOriginal, data.original || currentImageBase64);
    displayImage(imgMask, data.mask);
    displayImage(imgOverlay, data.overlay);
    if (data.contour && imgContour) {
      displayImage(imgContour, data.contour);
    } else if (imgContour) {
      displayImage(imgContour, data.overlay);
    }
    if (data.boundary && imgBoundary) {
      displayImage(imgBoundary, data.boundary);
    }

    if (dimOriginal) dimOriginal.textContent = data.original_dimensions || "ISIC Dermoscopy";
    if (valLesionArea) valLesionArea.textContent = `${data.lesion_area_pct}%`;
    if (valInferTime) valInferTime.textContent = `${data.inference_time_ms} ms`;
    if (valMeanFgProb && data.mean_foreground_prob !== undefined) {
      valMeanFgProb.textContent = typeof data.mean_foreground_prob === "number" ? data.mean_foreground_prob.toFixed(3) : data.mean_foreground_prob;
    }
    if (valOperatingThTag && data.threshold !== undefined) {
      valOperatingThTag.textContent = typeof data.threshold === "number" ? data.threshold.toFixed(2) : data.threshold;
    }
    if (valOperatingThDisplay && data.threshold !== undefined) {
      valOperatingThDisplay.textContent = `${typeof data.threshold === "number" ? data.threshold.toFixed(2) : data.threshold} (Val-Sweep)`;
    }
    if (stateTag) stateTag.textContent = isLiveBackend ? "Live Segmentation Complete" : "Showcase Segmentation Complete";
  }

  async function executeLiveInference() {
    if (!currentImageBase64) return;

    if (btnSegment) btnSegment.disabled = true;
    if (segmentSpinner) segmentSpinner.style.display = "inline-block";
    if (btnText) btnText.textContent = "Analyzing...";
    if (stateTag) stateTag.textContent = "Running RB-UNet...";

    if (valLesionArea) valLesionArea.textContent = "...";
    if (valInferTime) valInferTime.textContent = "...";
    if (valMeanFgProb) valMeanFgProb.textContent = "...";

    // 1. Try Live Backend API if active
    if (isLiveBackend || getApiBase()) {
      try {
        const response = await fetch(apiUrl("/api/segment"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: currentImageBase64 })
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            renderInferenceSuccess(data);
            if (btnSegment) btnSegment.disabled = false;
            if (segmentSpinner) segmentSpinner.style.display = "none";
            if (btnText) btnText.textContent = "Segment Lesion";
            return;
          }
        }
      } catch (err) {
        console.warn("Live API call failed, attempting showcase preset fallback:", err);
      }
    }

    // 2. Showcase Preset Fallback (Instant, 100% genuine masks)
    if (currentPresetName) {
      if (!precomputedPresets) {
        try {
          const pResp = await fetch(getAssetPath("outputs/presets_data.json"));
          if (pResp.ok) {
            precomputedPresets = await pResp.json();
          }
        } catch (e) {
          console.warn("Could not load presets_data.json:", e);
        }
      }

      if (precomputedPresets && precomputedPresets[currentPresetName]) {
        const pData = precomputedPresets[currentPresetName];
        renderInferenceSuccess({
          original: currentImageBase64,
          mask: pData.mask,
          overlay: pData.overlay,
          contour: pData.overlay,
          boundary: pData.boundary,
          lesion_area_pct: pData.lesion_area_pct,
          inference_time_ms: pData.inference_time_ms,
          mean_foreground_prob: pData.mean_fg_confidence,
          threshold: pData.threshold_used,
          original_dimensions: "ISIC Benchmark Cohort"
        });
        if (btnSegment) btnSegment.disabled = false;
        if (segmentSpinner) segmentSpinner.style.display = "none";
        if (btnText) btnText.textContent = "Segment Lesion";
        return;
      }
    }

    // 3. User uploaded custom image without a live backend running
    if (stateTag) stateTag.textContent = "Showcase Mode Active";
    alert(
      "⚡ Showcase Demo Mode:\n\n" +
      "The 5 ISIC benchmark presets (Small Lesion, Large Lesion, Irregular Margin, Low Contrast, Challenging) are fully interactive with pre-calculated segmentation masks.\n\n" +
      "To run live inference on your own custom uploaded images, connect your live Python backend by clicking the status pill in the top bar or running `python app.py`."
    );

    if (btnSegment) btnSegment.disabled = false;
    if (segmentSpinner) segmentSpinner.style.display = "none";
    if (btnText) btnText.textContent = "Segment Lesion";
  }

  // =========================================================================
  // LIVE ROBUSTNESS LAB (INTERACTIVE STRESS-TEST)
  // =========================================================================
  const robCurrentCondition = document.getElementById("robustness-current-condition");
  const robStatusTag = document.getElementById("rob-status-tag");
  const robActiveCondLabel = document.getElementById("rob-active-cond-label");
  const robCleanInput = document.getElementById("rob-clean-input");
  const robDegradedInput = document.getElementById("rob-degraded-input");
  const robBaseMask = document.getElementById("rob-base-mask");
  const robRbMask = document.getElementById("rob-rb-mask");

  const robConsistencyVal = document.getElementById("rob-consistency-val");
  const robDeltaAreaVal = document.getElementById("rob-delta-area-val");
  const robBaseAreaVal = document.getElementById("rob-base-area-val");
  const robBaseMsVal = document.getElementById("rob-base-ms-val");
  const robRbAreaVal = document.getElementById("rob-rb-area-val");
  const robRbMsVal = document.getElementById("rob-rb-ms-val");

  const conditionPills = document.querySelectorAll(".condition-pill");
  conditionPills.forEach(pill => {
    pill.addEventListener("click", () => {
      conditionPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeCondition = pill.getAttribute("data-condition");
      executeLiveRobustness(activeCondition);
    });
  });

  async function runDefaultLiveRobustness() {
    if (!currentImageBase64) {
      try {
        const resp = await fetch(getAssetPath("outputs/sample_images/small_lesion.jpg"));
        if (resp.ok) {
          const blob = await resp.blob();
          const reader = new FileReader();
          reader.onload = (e) => {
            currentImageBase64 = e.target.result;
            executeLiveRobustness(activeCondition);
          };
          reader.readAsDataURL(blob);
          return;
        }
      } catch (e) {
        console.warn("Could not load default sample for robustness lab", e);
      }
    }
    executeLiveRobustness(activeCondition);
  }

  function renderRobustnessSuccess(data, conditionName) {
    displayImage(robCleanInput, data.clean_image || currentImageBase64);
    displayImage(robDegradedInput, data.degraded_image);
    displayImage(robBaseMask, (data.baseline && data.baseline.overlay) || data.unet_overlay || (data.baseline && data.baseline.mask) || data.unet_mask);
    displayImage(robRbMask, (data.rb_unet && data.rb_unet.overlay) || data.rb_unet_overlay || (data.rb_unet && data.rb_unet.mask) || data.rb_unet_mask);

    if (robConsistencyVal) robConsistencyVal.textContent = `${data.prediction_consistency !== undefined ? data.prediction_consistency.toFixed(1) : 100.0}%`;
    if (robDeltaAreaVal) robDeltaAreaVal.textContent = `${data.area_delta_pct !== undefined ? data.area_delta_pct.toFixed(1) : 0.0}%`;
    
    const bArea = (data.baseline && data.baseline.lesion_area_pct !== undefined) ? data.baseline.lesion_area_pct : data.unet_area_pct;
    const bMs = (data.baseline && data.baseline.inference_time_ms !== undefined) ? data.baseline.inference_time_ms : data.unet_inference_time_ms;
    const rbArea = (data.rb_unet && data.rb_unet.lesion_area_pct !== undefined) ? data.rb_unet.lesion_area_pct : data.rb_unet_area_pct;
    const rbMs = (data.rb_unet && data.rb_unet.inference_time_ms !== undefined) ? data.rb_unet.inference_time_ms : data.rb_unet_inference_time_ms;

    if (robBaseAreaVal) robBaseAreaVal.textContent = `${bArea !== undefined ? bArea : "--"}%`;
    if (robBaseMsVal) robBaseMsVal.textContent = `Latency: ${bMs !== undefined ? bMs : "--"} ms`;
    if (robRbAreaVal) robRbAreaVal.textContent = `${rbArea !== undefined ? rbArea : "--"}%`;
    if (robRbMsVal) robRbMsVal.textContent = `Latency: ${rbMs !== undefined ? rbMs : "--"} ms`;
    if (robStatusTag) robStatusTag.textContent = conditionName === "Clean" ? "Unperturbed Reference" : "Corrupted Variant";
  }

  async function executeLiveRobustness(conditionName) {
    robustnessHasRun = true;

    if (robCurrentCondition) robCurrentCondition.textContent = `Active Degradation: ${conditionName}`;
    if (robActiveCondLabel) robActiveCondLabel.textContent = conditionName;
    if (robStatusTag) robStatusTag.textContent = "Processing Dual Inference...";

    // 1. Try Live API if backend is connected
    if ((isLiveBackend || getApiBase()) && currentImageBase64) {
      try {
        const resp = await fetch(apiUrl("/api/robustness"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: currentImageBase64,
            condition: conditionName
          })
        });

        if (resp.ok) {
          const data = await resp.json();
          if (data.success) {
            renderRobustnessSuccess(data, conditionName);
            return;
          }
        }
      } catch (err) {
        console.warn("Live robustness API failed, falling back to showcase cohort:", err);
      }
    }

    // 2. Showcase Precomputed Benchmark Cohort Fallback
    const condSlugMap = {
      "Clean": "clean",
      "Gaussian Noise": "gaussian_noise",
      "Gaussian Blur": "gaussian_blur",
      "Contrast Shift": "contrast",
      "Brightness Shift": "brightness"
    };
    const condSlug = condSlugMap[conditionName] || "clean";

    const cleanImgUrl = getAssetPath("outputs/robustness/clean/input.png");
    const degradedImgUrl = getAssetPath(`outputs/robustness/${condSlug}/input.png`);
    const baseMaskUrl = getAssetPath(`outputs/robustness/${condSlug}/baseline_prediction.png`);
    const rbMaskUrl = getAssetPath(`outputs/robustness/${condSlug}/rb_unet_prediction.png`);

    const statsMap = {
      "Clean": { consistency: 100.0, deltaArea: 0.0, baseArea: 3.4, rbArea: 1.3, baseMs: 27.8, rbMs: 28.5 },
      "Gaussian Noise": { consistency: 93.3, deltaArea: 0.2, baseArea: 43.2, rbArea: 1.1, baseMs: 27.5, rbMs: 28.2 },
      "Gaussian Blur": { consistency: 97.8, deltaArea: 0.1, baseArea: 3.2, rbArea: 1.3, baseMs: 27.4, rbMs: 28.0 },
      "Contrast Shift": { consistency: 65.5, deltaArea: 0.7, baseArea: 2.5, rbArea: 0.6, baseMs: 27.6, rbMs: 28.1 },
      "Brightness Shift": { consistency: 83.6, deltaArea: 0.5, baseArea: 47.7, rbArea: 1.8, baseMs: 27.9, rbMs: 28.4 }
    };
    const stats = statsMap[conditionName] || statsMap["Clean"];

    displayImage(robCleanInput, cleanImgUrl);
    displayImage(robDegradedInput, degradedImgUrl);
    displayImage(robBaseMask, baseMaskUrl);
    displayImage(robRbMask, rbMaskUrl);

    if (robConsistencyVal) robConsistencyVal.textContent = `${stats.consistency.toFixed(1)}%`;
    if (robDeltaAreaVal) robDeltaAreaVal.textContent = `${stats.deltaArea.toFixed(1)}%`;
    if (robBaseAreaVal) robBaseAreaVal.textContent = `${stats.baseArea}%`;
    if (robBaseMsVal) robBaseMsVal.textContent = `Latency: ${stats.baseMs} ms`;
    if (robRbAreaVal) robRbAreaVal.textContent = `${stats.rbArea}%`;
    if (robRbMsVal) robRbMsVal.textContent = `Latency: ${stats.rbMs} ms`;
    if (robStatusTag) robStatusTag.textContent = conditionName === "Clean" ? "Unperturbed Reference" : "Corrupted Variant";
  }

  // =========================================================================
  // RESEARCH RESULTS & COHORT BENCHMARK DATA FETCH
  // =========================================================================
  let researchDataLoaded = false;
  let cachedResults = null;
  let robustnessChartInstance = null;
  let ablationChartInstance = null;

  async function fetchResearchResults() {
    if (researchDataLoaded && cachedResults) return;

    let data = null;

    // 1. Try Live API first if backend configured
    if (isLiveBackend || getApiBase()) {
      try {
        const resp = await fetch(apiUrl("/api/results"));
        if (resp.ok) {
          data = await resp.json();
        }
      } catch (e) {
        console.warn("Could not fetch /api/results from backend:", e);
      }
    }

    // 2. Fallback to static JSON file (guaranteed on GitHub Pages / Vercel)
    if (!data) {
      try {
        const resp = await fetch(getAssetPath("outputs/api_results.json"));
        if (resp.ok) {
          data = await resp.json();
        }
      } catch (e) {
        console.warn("Could not load static api_results.json:", e);
      }
    }

    if (data) {
      // Normalize relative image URLs in qualitative suite
      if (data.qualitative) {
        data.qualitative.forEach(c => {
          c.original_url = getAssetPath(c.original_url || "");
          c.gt_url = getAssetPath(c.gt_url || "");
          c.baseline_url = getAssetPath(c.baseline_url || "");
          c.rb_unet_url = getAssetPath(c.rb_unet_url || "");
        });
      }

      cachedResults = data;
      researchDataLoaded = true;

      renderResearchPerformance(data);
      renderAblationStudy(data);
      renderRobustnessBenchmarkTable(data);
      renderSacredTestResults(data);
      renderQualitativeGallery(data);
    }
  }

  // 1. Overall Performance (Clean Validation Split)
  function renderResearchPerformance(data) {
    const abl = data.ablation || [];
    const baseExp = abl.find(a => a.experiment === "baseline_unet");
    const rbExp = abl.find(a => a.experiment === "full_rb_unet");

    const pBaseDice = document.getElementById("perf-base-dice");
    const pBaseIou = document.getElementById("perf-base-iou");
    const pRbDice = document.getElementById("perf-rb-dice");
    const pRbIou = document.getElementById("perf-rb-iou");
    const pDeltaDice = document.getElementById("perf-delta-dice");

    if (baseExp && pBaseDice && pBaseIou) {
      pBaseDice.textContent = (baseExp.best_val_dice || 0).toFixed(4);
      pBaseIou.textContent = (baseExp.best_val_iou || 0).toFixed(4);
    }
    if (rbExp && pRbDice && pRbIou) {
      pRbDice.textContent = (rbExp.best_val_dice || 0).toFixed(4);
      pRbIou.textContent = (rbExp.best_val_iou || 0).toFixed(4);
    }
    if (baseExp && rbExp && pDeltaDice) {
      const delta = (rbExp.best_val_dice || 0) - (baseExp.best_val_dice || 0);
      const sign = delta >= 0 ? "+" : "";
      pDeltaDice.textContent = `${sign}${delta.toFixed(4)} (${sign}${(delta * 100).toFixed(2)}%)`;
    }
  }

  // 2. Ablation Suite Table & Chart
  function renderAblationStudy(data) {
    const tableBody = document.getElementById("table-ablation-body");
    const abl = data.ablation || [];
    if (!tableBody || abl.length === 0) return;

    const stages = [
      { id: "baseline_unet", name: "Standard U-Net Baseline", boundary: "—", consistency: "—" },
      { id: "ablation_boundary", name: "U-Net + Boundary Supervision", boundary: "✓ (Aux Head)", consistency: "—" },
      { id: "ablation_consistency", name: "U-Net + Corruption Consistency", boundary: "—", consistency: "✓ (L2 Consistency)" },
      { id: "full_rb_unet", name: "Full Proposed RB-UNet", boundary: "✓ (Aux Head)", consistency: "✓ (Both Mechanisms)", isFull: true }
    ];

    let html = "";
    const chartLabels = [];
    const chartDice = [];
    const chartIou = [];

    stages.forEach(st => {
      const match = abl.find(a => a.experiment === st.id) || {};
      const d = match.best_val_dice !== undefined ? match.best_val_dice.toFixed(4) : "—";
      const i = match.best_val_iou !== undefined ? match.best_val_iou.toFixed(4) : "—";

      chartLabels.push(st.name.replace("Standard ", "").replace("U-Net + ", "+ "));
      chartDice.push(match.best_val_dice || 0);
      chartIou.push(match.best_val_iou || 0);

      const rowClass = st.isFull ? "class='highlight-row'" : "";
      html += `
        <tr ${rowClass}>
          <td><strong>${st.name}</strong></td>
          <td>${st.boundary}</td>
          <td>${st.consistency}</td>
          <td class="mono ${st.isFull ? 'text-accent font-semibold' : ''}">${d}</td>
          <td class="mono ${st.isFull ? 'text-accent' : ''}">${i}</td>
        </tr>
      `;
    });
    tableBody.innerHTML = html;

    // Render Chart.js Ablation Bar Chart
    const ctxAbl = document.getElementById("chart-ablation");
    if (ctxAbl && typeof Chart !== "undefined") {
      if (ablationChartInstance) ablationChartInstance.destroy();
      ablationChartInstance = new Chart(ctxAbl, {
        type: "bar",
        data: {
          labels: chartLabels,
          datasets: [
            {
              label: "Validation Dice",
              data: chartDice,
              backgroundColor: "rgba(14, 165, 233, 0.85)",
              borderColor: "#0EA5E9",
              borderWidth: 1,
              borderRadius: 4
            },
            {
              label: "Validation IoU",
              data: chartIou,
              backgroundColor: "rgba(16, 185, 129, 0.85)",
              borderColor: "#10B981",
              borderWidth: 1,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: false,
              min: 0.65,
              max: 0.80,
              grid: { color: "#F1F5F9" },
              ticks: { color: "#64748B", font: { family: "JetBrains Mono" } }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#334155", font: { size: 10 } }
            }
          },
          plugins: {
            legend: {
              labels: { color: "#0F172A", font: { family: "Inter", size: 11 } }
            }
          }
        }
      });
    }
  }

  // 3. Controlled Robustness Benchmark (Cohort Aggregate)
  function renderRobustnessBenchmarkTable(data) {
    const rob = data.robustness || [];
    const tableBody = document.getElementById("table-robustness-body");
    if (!tableBody || rob.length === 0) return;

    let tblHtml = "";
    rob.forEach(item => {
      const delta = (item.delta_dice !== undefined) ? item.delta_dice : (item.rb_unet_dice - item.baseline_dice);
      const sign = delta >= 0 ? "+" : "";
      const deltaClass = delta >= 0 ? "text-accent font-semibold" : "text-muted";
      const advDesc = delta > 0.15 ? "High Noise Resistance" : (delta > 0.05 ? "Robust Contour Fit" : "High Precision");

      tblHtml += `
        <tr>
          <td><strong>${item.condition}</strong></td>
          <td class="mono">${(item.baseline_dice || 0).toFixed(4)}</td>
          <td class="mono text-accent font-semibold">${(item.rb_unet_dice || 0).toFixed(4)}</td>
          <td class="mono ${deltaClass}">${sign}${delta.toFixed(4)}</td>
          <td><span class="badge-tag highlight">${advDesc}</span></td>
        </tr>
      `;
    });
    tableBody.innerHTML = tblHtml;

    // Render Cohort Robustness Chart
    const ctxRob = document.getElementById("chart-robustness");
    if (ctxRob && typeof Chart !== "undefined") {
      if (robustnessChartInstance) robustnessChartInstance.destroy();
      const labels = rob.map(r => r.condition);
      const baseScores = rob.map(r => r.baseline_dice || 0);
      const rbScores = rob.map(r => r.rb_unet_dice || 0);

      robustnessChartInstance = new Chart(ctxRob, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Standard U-Net Baseline",
              data: baseScores,
              backgroundColor: "rgba(148, 163, 184, 0.7)",
              borderColor: "#94A3B8",
              borderWidth: 1,
              borderRadius: 4
            },
            {
              label: "RB-UNet (Proposed)",
              data: rbScores,
              backgroundColor: "rgba(2, 132, 199, 0.85)",
              borderColor: "#0284C7",
              borderWidth: 1,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: false,
              min: 0.45,
              max: 0.85,
              grid: { color: "#F1F5F9" },
              ticks: { color: "#64748B", font: { family: "JetBrains Mono" } }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#334155", font: { size: 10 } }
            }
          },
          plugins: {
            legend: {
              labels: { color: "#0F172A", font: { family: "Inter", size: 11 } }
            }
          }
        }
      });
    }
  }

  // 4. Qualitative Case Gallery
  function renderQualitativeGallery(data) {
    const gallery = document.getElementById("qualitative-gallery");
    const qual = data.qualitative || [];
    if (!gallery || qual.length === 0) return;

    let html = "";
    qual.forEach(c => {
      const sign = c.delta_dice >= 0 ? "+" : "";
      html += `
        <div class="case-card">
          <div class="case-header">
            <div>
              <span class="case-title">${c.label}</span>
              <span class="text-muted" style="font-size: 0.75rem; margin-left: 8px;">(${c.id} — ${c.difficulty})</span>
            </div>
            <div class="case-metrics-badge mono">
              Baseline: ${c.baseline_dice.toFixed(4)} → RB-UNet: <strong>${c.rb_unet_dice.toFixed(4)}</strong> (${sign}${c.delta_dice.toFixed(4)})
            </div>
          </div>
          <div class="case-row">
            <div class="case-item">
              <span class="case-item-label">ORIGINAL</span>
              <div class="case-item-img">
                <img src="${c.original_url}" alt="${c.id} Original" loading="lazy">
              </div>
            </div>
            <div class="case-item">
              <span class="case-item-label">GROUND TRUTH</span>
              <div class="case-item-img">
                <img src="${c.gt_url}" alt="${c.id} Ground Truth" loading="lazy">
              </div>
            </div>
            <div class="case-item">
              <span class="case-item-label">STANDARD U-NET</span>
              <div class="case-item-img">
                <img src="${c.baseline_url}" alt="${c.id} Baseline Mask" loading="lazy">
              </div>
            </div>
            <div class="case-item">
              <span class="case-item-label text-accent">RB-UNET PREDICTION</span>
              <div class="case-item-img" style="border-color: rgba(2, 132, 199, 0.4);">
                <img src="${c.rb_unet_url}" alt="${c.id} RB-UNet Mask" loading="lazy">
              </div>
            </div>
          </div>
        </div>
      `;
    });
    gallery.innerHTML = html;
  }

  // 5. Sacred Final Test Performance
  function renderSacredTestResults(data) {
    const test = data.final_test || {};
    const tBaseDice = document.getElementById("test-base-dice");
    const tRbDice = document.getElementById("test-rb-dice");
    const tDeltaDice = document.getElementById("test-delta-dice");

    if (test.baseline_unet && tBaseDice) {
      tBaseDice.textContent = (test.baseline_unet.test_dice || 0).toFixed(4);
    }
    if (test.rb_unet && tRbDice) {
      tRbDice.textContent = (test.rb_unet.test_dice || 0).toFixed(4);
    }
    if (test.delta_dice !== undefined && tDeltaDice) {
      const sign = test.delta_dice >= 0 ? "+" : "";
      tDeltaDice.textContent = `${sign}${test.delta_dice.toFixed(4)} (${sign}${(test.delta_dice * 100).toFixed(2)}%)`;
    }
  }
});
