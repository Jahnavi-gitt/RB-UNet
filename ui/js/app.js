// RB-UNet Research Workstation Orchestrator

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
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
        // If robustness has not run yet, run it on current image or load default preset
        if (!robustnessHasRun) {
          runDefaultLiveRobustness();
        }
      } else if (targetId === "section-results") {
        fetchResearchResults();
      }
    });
  });

  // System Status Check
  const statusIndicatorDot = document.getElementById("status-indicator-dot");
  const systemStatusText = document.getElementById("system-status-text");

  async function checkSystemStatus() {
    try {
      const resp = await fetch("/api/status");
      if (resp.ok) {
        const data = await resp.json();
        if (data.best_checkpoint_exists && data.baseline_checkpoint_exists) {
          systemStatusText.textContent = "Model Ready";
          statusIndicatorDot.style.backgroundColor = "var(--accent-green)";
          statusIndicatorDot.style.boxShadow = "0 0 6px rgba(16, 185, 129, 0.5)";
        } else if (data.best_checkpoint_exists) {
          systemStatusText.textContent = "RB-UNet Loaded";
          statusIndicatorDot.style.backgroundColor = "var(--accent-green)";
        } else {
          systemStatusText.textContent = "Weights Pending";
          statusIndicatorDot.style.backgroundColor = "var(--accent-amber)";
        }
      }
    } catch (e) {
      systemStatusText.textContent = "Offline";
      statusIndicatorDot.style.backgroundColor = "var(--accent-red)";
    }
  }
  checkSystemStatus();

  // =========================================================================
  // STATE MANAGEMENT
  // =========================================================================
  let currentImageBase64 = null;
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

  // Safe image display helper to ensure clean display without broken icons
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
    const reader = new FileReader();
    reader.onload = (event) => {
      currentImageBase64 = event.target.result;
      if (btnSegment) btnSegment.disabled = false;
      if (stateTag) stateTag.textContent = "Image Loaded";

      // Show temporary preview
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
      if (stateTag) stateTag.textContent = "Loading Preset...";
      const resp = await fetch(`/outputs/sample_images/${presetName}.jpg`);
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

        // Automatically trigger live segmentation
        await executeLiveInference();
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error("Error loading preset:", err);
      if (stateTag) stateTag.textContent = "Preset Error";
      alert("Could not load preset dermoscopy image: " + err.message);
    }
  }

  // Trigger Live Segmentation Inference
  if (btnSegment) {
    btnSegment.addEventListener("click", executeLiveInference);
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

    try {
      const response = await fetch("/api/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: currentImageBase64 })
      });

      const data = await response.json();
      if (data.success) {
        displayImage(imgOriginal, data.original || data.image);
        displayImage(imgMask, data.mask);
        displayImage(imgOverlay, data.overlay);
        if (data.contour && imgContour) {
          displayImage(imgContour, data.contour);
        }
        if (data.boundary && imgBoundary) {
          displayImage(imgBoundary, data.boundary);
        }

        if (dimOriginal) dimOriginal.textContent = data.original_dimensions || "";
        if (valLesionArea) valLesionArea.textContent = `${data.lesion_area_pct}%`;
        if (valInferTime) valInferTime.textContent = `${data.inference_time_ms} ms`;
        if (valMeanFgProb && data.mean_foreground_prob !== undefined) {
          valMeanFgProb.textContent = data.mean_foreground_prob.toFixed(3);
        }
        if (valOperatingThTag && data.threshold !== undefined) {
          valOperatingThTag.textContent = data.threshold.toFixed(2);
        }
        if (valOperatingThDisplay && data.threshold !== undefined) {
          valOperatingThDisplay.textContent = `${data.threshold.toFixed(2)} (Val-Sweep)`;
        }
        if (stateTag) stateTag.textContent = "Segmentation Complete";
      } else {
        if (stateTag) stateTag.textContent = "Inference Failed";
        alert("Segmentation could not be generated: " + (data.error || "Please check image."));
      }
    } catch (err) {
      console.error("Inference request error:", err);
      if (stateTag) stateTag.textContent = "Network Error";
      alert("Failed to communicate with segmentation backend.");
    } finally {
      if (btnSegment) btnSegment.disabled = false;
      if (segmentSpinner) segmentSpinner.style.display = "none";
      if (btnText) btnText.textContent = "Segment Lesion";
    }
  }

  // =========================================================================
  // LIVE ROBUSTNESS LAB (INTERACTIVE STRESS-TEST ON ACTIVE USER IMAGE)
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
  const btnSyncLiveImage = document.getElementById("btn-sync-live-image");

  if (btnSyncLiveImage) {
    btnSyncLiveImage.addEventListener("click", () => {
      if (currentImageBase64) {
        executeLiveRobustness(activeCondition);
      } else {
        runDefaultLiveRobustness();
      }
    });
  }

  // Robustness condition pills
  const robPills = document.querySelectorAll("#corruption-pills-bar .pill");
  robPills.forEach(pill => {
    pill.addEventListener("click", () => {
      robPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeCondition = pill.getAttribute("data-condition");
      executeLiveRobustness(activeCondition);
    });
  });

  async function runDefaultLiveRobustness() {
    if (!currentImageBase64) {
      try {
        const resp = await fetch("/outputs/sample_images/small_lesion.jpg");
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
    } else {
      executeLiveRobustness(activeCondition);
    }
  }

  async function executeLiveRobustness(conditionName) {
    if (!currentImageBase64) return;
    robustnessHasRun = true;

    if (robCurrentCondition) robCurrentCondition.textContent = `Active Degradation: ${conditionName}`;
    if (robActiveCondLabel) robActiveCondLabel.textContent = conditionName;
    if (robStatusTag) robStatusTag.textContent = "Processing Dual Inference...";

    try {
      const resp = await fetch("/api/robustness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: currentImageBase64,
          condition: conditionName
        })
      });

      const data = await resp.json();
      if (data.success) {
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
      } else {
        if (robStatusTag) robStatusTag.textContent = "Evaluation Failed";
      }
    } catch (err) {
      console.error("Live robustness error:", err);
      if (robStatusTag) robStatusTag.textContent = "Network Error";
    }
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

    try {
      const resp = await fetch("/api/results");
      if (!resp.ok) throw new Error("Failed to fetch /api/results");

      const data = await resp.json();
      cachedResults = data;
      researchDataLoaded = true;

      renderResearchPerformance(data);
      renderAblationStudy(data);
      renderRobustnessBenchmarkTable(data);
      renderSacredTestResults(data);
      renderQualitativeGallery(data);

    } catch (e) {
      console.error("Error fetching research results:", e);
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

    // Render Ablation Chart
    const ctx = document.getElementById("chart-ablation");
    if (ctx && typeof Chart !== "undefined") {
      if (ablationChartInstance) ablationChartInstance.destroy();
      ablationChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: chartLabels,
          datasets: [
            {
              label: "Validation Dice",
              data: chartDice,
              backgroundColor: "rgba(2, 132, 199, 0.8)",
              borderColor: "#0284C7",
              borderWidth: 1,
              borderRadius: 4
            },
            {
              label: "Validation IoU",
              data: chartIou,
              backgroundColor: "rgba(13, 148, 136, 0.7)",
              borderColor: "#0D9488",
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
              min: 0.5,
              max: 0.85,
              grid: { color: "#F1F5F9" },
              ticks: { color: "#64748B", font: { family: "JetBrains Mono" } }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#334155", font: { size: 11 } }
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
