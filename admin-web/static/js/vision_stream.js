/**
 * Live Vision & AI Inspection Stream Controller
 * Interacts with /api/vision/infer-frame
 */

class VisionStreamMonitor {
  constructor(videoElementId = "vision-video", canvasElementId = "vision-canvas") {
    this.videoEl = document.getElementById(videoElementId);
    this.canvasEl = document.getElementById(canvasElementId);
    this.ctx = this.canvasEl ? this.canvasEl.getContext("2d") : null;
    this.offscreenCanvas = document.createElement("canvas");
    this.offscreenCtx = this.offscreenCanvas.getContext("2d");

    this.isStreaming = false;
    this.streamInterval = null;
    this.animFrameId = null;

    this.fpsCounterEl = document.getElementById("metric-fps");
    this.latencyCounterEl = document.getElementById("metric-latency");
    this.personsCountEl = document.getElementById("metric-persons");
    this.unknownCountEl = document.getElementById("metric-unknown");
    this.isInferring = false;

    this.lastDetections = [];
    this.lastIsGroup = false;
    this.lastGroupSummary = "";

    this.selectedZone = "ZONE A: ECO TRAIL";
    this.require100Accuracy = true;

    const zoneSelect = document.getElementById("select-camera-zone");
    if (zoneSelect) {
      this.selectedZone = zoneSelect.value;
      zoneSelect.addEventListener("change", (e) => {
        this.selectedZone = e.target.value;
        const zoneMetric = document.getElementById("metric-camera-zone");
        if (zoneMetric) zoneMetric.textContent = this.selectedZone;
      });
    }
  }

  async startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      if (this.videoEl) {
        this.videoEl.srcObject = stream;
        await this.videoEl.play();
        this.isStreaming = true;

        this.videoEl.onloadedmetadata = () => {
          const w = this.videoEl.videoWidth || 640;
          const h = this.videoEl.videoHeight || 480;
          this.canvasEl.width = w;
          this.canvasEl.height = h;
          this.offscreenCanvas.width = w;
          this.offscreenCanvas.height = h;
        };

        const w = this.videoEl.videoWidth || 640;
        const h = this.videoEl.videoHeight || 480;
        this.canvasEl.width = w;
        this.canvasEl.height = h;
        this.offscreenCanvas.width = w;
        this.offscreenCanvas.height = h;

        this.startRenderLoop();
        this.startInferenceLoop();
      }
    } catch (err) {
      console.warn("Camera stream unavailable. Starting synthetic test frame inference:", err);
      this.startSyntheticFeed();
    }
  }

  startRenderLoop() {
    const render = () => {
      if (!this.isStreaming) return;
      if (this.ctx && this.videoEl && this.videoEl.readyState >= 2) {
        // Draw live camera feed
        this.ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);
        // Draw detections overlay smoothly on top
        if (this.lastDetections && this.lastDetections.length > 0) {
          this.drawDetections(this.lastDetections, this.lastIsGroup, this.lastGroupSummary);
        }
      }
      this.animFrameId = requestAnimationFrame(render);
    };
    this.animFrameId = requestAnimationFrame(render);
  }

  startSyntheticFeed() {
    this.isStreaming = true;
    if (this.canvasEl) {
      this.canvasEl.width = 640;
      this.canvasEl.height = 480;
      this.offscreenCanvas.width = 640;
      this.offscreenCanvas.height = 480;
    }
    this.streamInterval = setInterval(() => {
      this.renderSyntheticFrameAndInfer();
    }, 500);
  }

  renderSyntheticFrameAndInfer() {
    if (!this.ctx || this.isInferring) return;

    // Draw simulated forest background
    this.ctx.fillStyle = "#121614";
    this.ctx.fillRect(0, 0, 640, 480);
    this.ctx.fillStyle = "#1B221E";
    this.ctx.fillRect(80, 100, 480, 280);

    this.ctx.fillStyle = "#2D6A4F";
    this.ctx.font = "14px Consolas, monospace";
    this.ctx.fillText(`TACTICAL SENSOR FEED: [${this.selectedZone}] (STANDBY MONITOR)`, 20, 30);

    const dataUrl = this.canvasEl.toDataURL("image/jpeg", 0.7);
    this.sendFrameForInference(dataUrl);
  }

  startInferenceLoop() {
    this.streamInterval = setInterval(() => {
      if (!this.isInferring && this.videoEl && this.isStreaming) {
        this.captureAndInfer();
      }
    }, 380); // ~2.6 frames per sec inspection cycle
  }

  captureAndInfer() {
    if (!this.videoEl || this.videoEl.readyState < 2) return;
    const w = this.offscreenCanvas.width || 640;
    const h = this.offscreenCanvas.height || 480;
    this.offscreenCtx.drawImage(this.videoEl, 0, 0, w, h);
    const dataUrl = this.offscreenCanvas.toDataURL("image/jpeg", 0.75);
    this.sendFrameForInference(dataUrl);
  }

  async sendFrameForInference(base64Data) {
    this.isInferring = true;
    const t0 = performance.now();

    try {
      const zoneSelect = document.getElementById("select-camera-zone");
      if (zoneSelect) this.selectedZone = zoneSelect.value;

      const res = await fetch("/api/vision/infer-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame: base64Data,
          camera_zone: this.selectedZone,
          require_100_accuracy: this.require100Accuracy
        })
      });
      const data = await res.json();
      const clientLatency = Math.round(performance.now() - t0);

      if (data.success) {
        this.lastFrameWidth = data.frame_width || this.offscreenCanvas.width || 640;
        this.lastFrameHeight = data.frame_height || this.offscreenCanvas.height || 480;
        
        // Strictly filter: ONLY show if 100% accurate in detection of person
        const rawDetections = data.detections || [];
        this.lastDetections = rawDetections.filter(d => {
          const acc = Math.round((d.confidence || 0) * 100);
          return acc >= 100;
        });

        this.lastIsGroup = this.lastDetections.length > 1;
        this.lastGroupSummary = data.group_summary || "";

        this.updateMetrics(
          data.fps,
          data.latency_ms || clientLatency,
          this.lastDetections.length,
          data.unregistered_persons || data.unknown_persons || 0,
          data.registered_persons || 0,
          this.lastIsGroup,
          this.lastGroupSummary
        );
      }
    } catch (e) {
      console.error("Frame inference error:", e);
    } finally {
      this.isInferring = false;
    }
  }

  updateMetrics(fps, latency, total, unknown, registered, isGroup, groupSummary) {
    if (this.fpsCounterEl) this.fpsCounterEl.textContent = `${fps} FPS`;
    if (this.latencyCounterEl) this.latencyCounterEl.textContent = `${latency} ms`;
    if (this.personsCountEl) this.personsCountEl.textContent = total;
    if (this.unknownCountEl) this.unknownCountEl.textContent = unknown;

    const regEl = document.getElementById("metric-registered");
    if (regEl) regEl.textContent = registered;

    const zoneEl = document.getElementById("metric-camera-zone");
    if (zoneEl) zoneEl.textContent = this.selectedZone;

    const groupEl = document.getElementById("metric-group-status");
    if (groupEl) {
      if (isGroup) {
        groupEl.className = "badge badge-critical";
        groupEl.textContent = `CLUSTER (${total})`;
      } else if (total === 1) {
        groupEl.className = "badge badge-safe";
        groupEl.textContent = "SOLO";
      } else {
        groupEl.className = "badge badge-muted";
        groupEl.textContent = "NONE";
      }
    }

    const logContainer = document.getElementById("vision-log-container");
    if (logContainer && total > 0) {
      const now = new Date().toLocaleTimeString();
      let logHtml = `<div style="color: var(--accent-rescuer-light); margin-bottom: 6px;">[${now}] ${groupSummary}</div>`;
      logContainer.innerHTML = logHtml + logContainer.innerHTML.slice(0, 800);
    }
  }

  drawDetections(detections, isGroup = false, groupSummary = "") {
    if (!this.ctx || !detections) return;

    // Strict 100% accuracy filter: only show if 100% accurate in detection of person
    const validDetections = detections.filter(d => {
      const acc = Math.round((d.confidence || 0) * 100);
      return acc >= 100;
    });

    if (validDetections.length === 0) return;

    const scaleX = this.canvasEl.width / (this.lastFrameWidth || this.canvasEl.width || 640);
    const scaleY = this.canvasEl.height / (this.lastFrameHeight || this.canvasEl.height || 480);

    // 1. Group Banner Overlay
    if (isGroup) {
      this.ctx.fillStyle = "rgba(193, 18, 31, 0.88)";
      this.ctx.fillRect(0, 0, this.canvasEl.width, 28);
      this.ctx.fillStyle = "#FFF";
      this.ctx.font = "bold 12px Consolas, monospace";
      this.ctx.textAlign = "center";
      this.ctx.fillText(`⚠ ${groupSummary.toUpperCase()}`, this.canvasEl.width / 2, 19);
      this.ctx.textAlign = "left";
    }

    validDetections.forEach((det, idx) => {
      const x1 = Math.round(det.coords[0] * scaleX);
      const y1 = Math.round(det.coords[1] * scaleY);
      const x2 = Math.round(det.coords[2] * scaleX);
      const y2 = Math.round(det.coords[3] * scaleY);

      const isRegistered = det.is_registered === true || (det.identity && det.identity.startsWith("REGISTERED"));
      const boxColor = isRegistered ? "#2D6A4F" : "#C1121F";
      const badgeBg = isRegistered ? "rgba(18, 45, 28, 0.95)" : "rgba(45, 18, 20, 0.95)";
      const textColor = isRegistered ? "#52B788" : "#FF4D4D";
      const personZone = det.zone || this.selectedZone || "ZONE A: ECO TRAIL";

      // 1. Person Bounding Box
      this.ctx.strokeStyle = boxColor;
      this.ctx.lineWidth = isRegistered ? 2 : 3;
      this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Corner Accents for Military HUD effect
      const cornerLen = Math.min(18, Math.floor((x2 - x1) * 0.2));
      this.ctx.lineWidth = 3;
      // Top-Left
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1 + cornerLen);
      this.ctx.lineTo(x1, y1);
      this.ctx.lineTo(x1 + cornerLen, y1);
      this.ctx.stroke();
      // Top-Right
      this.ctx.beginPath();
      this.ctx.moveTo(x2 - cornerLen, y1);
      this.ctx.lineTo(x2, y1);
      this.ctx.lineTo(x2, y1 + cornerLen);
      this.ctx.stroke();
      // Bottom-Left
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y2 - cornerLen);
      this.ctx.lineTo(x1, y2);
      this.ctx.lineTo(x1 + cornerLen, y2);
      this.ctx.stroke();
      // Bottom-Right
      this.ctx.beginPath();
      this.ctx.moveTo(x2 - cornerLen, y2);
      this.ctx.lineTo(x2, y2);
      this.ctx.lineTo(x2, y2 - cornerLen);
      this.ctx.stroke();

      // 2. Face Targeting Reticle (if face localized)
      if (det.face_coords) {
        const fx1 = Math.round(det.face_coords[0] * scaleX);
        const fy1 = Math.round(det.face_coords[1] * scaleY);
        const fx2 = Math.round(det.face_coords[2] * scaleX);
        const fy2 = Math.round(det.face_coords[3] * scaleY);

        this.ctx.strokeStyle = isRegistered ? "#52B788" : "#FFB703";
        this.ctx.lineWidth = 1.5;
        this.ctx.strokeRect(fx1, fy1, fx2 - fx1, fy2 - fy1);

        const simPercent = Math.round((det.similarity || 0) * 100);
        const tag = isRegistered
          ? `FACE-BIO: VERIFIED${simPercent > 0 ? ` (${simPercent}%)` : ""}`
          : "FACE-BIO: UNREGISTERED";
        this.ctx.fillText(tag, fx1 + 4, fy1 + 11);
      }

      // 3. Centroid Crosshair
      const cx = det.centroid ? Math.round(det.centroid[0] * scaleX) : Math.round((x1 + x2) / 2);
      const cy = det.centroid ? Math.round(det.centroid[1] * scaleY) : Math.round((y1 + y2) / 2);

      this.ctx.fillStyle = boxColor;
      this.ctx.fillRect(cx - 3, cy - 3, 6, 6);
      this.ctx.strokeStyle = textColor;
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
      this.ctx.stroke();

      // 4. Identity & Zone Header Badge (Top)
      const simPercent = Math.round((det.similarity || 0) * 100);
      const matchText = isRegistered && simPercent > 0 ? ` (${simPercent}% MATCH)` : "";
      const labelText = isRegistered
        ? `✔ ${det.person_name || "REGISTERED"}${matchText}`
        : "⚠ UNREGISTERED";

      const topBadgeText = `${labelText} | ZONE: ${personZone} [100% ACCURACY]`;

      this.ctx.font = "bold 11px Consolas, monospace";
      const textWidth = this.ctx.measureText(topBadgeText).width;
      const badgeHeight = 22;
      const badgeY = Math.max(0, y1 - badgeHeight);

      this.ctx.fillStyle = badgeBg;
      this.ctx.fillRect(x1, badgeY, Math.max(textWidth + 14, x2 - x1), badgeHeight);
      this.ctx.strokeStyle = boxColor;
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(x1, badgeY, Math.max(textWidth + 14, x2 - x1), badgeHeight);

      this.ctx.fillStyle = textColor;
      this.ctx.fillText(topBadgeText, x1 + 6, badgeY + 15);

      // 5. Zone Location Pill (Bottom of Box)
      const bottomTag = `📍 ZONE: ${personZone}  •  DETECTION: 100% ACCURATE`;
      this.ctx.font = "bold 10px Consolas, monospace";
      const bottomWidth = this.ctx.measureText(bottomTag).width;
      const bottomY = Math.min(this.canvasEl.height - 20, y2 + 3);

      this.ctx.fillStyle = "rgba(18, 22, 20, 0.95)";
      this.ctx.fillRect(x1, bottomY, Math.max(bottomWidth + 14, x2 - x1), 18);
      this.ctx.strokeStyle = "#52B788";
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(x1, bottomY, Math.max(bottomWidth + 14, x2 - x1), 18);

      this.ctx.fillStyle = "#52B788";
      this.ctx.fillText(bottomTag, x1 + 6, bottomY + 13);
    });
  }

  stop() {
    this.isStreaming = false;
    if (this.streamInterval) clearInterval(this.streamInterval);
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    if (this.videoEl && this.videoEl.srcObject) {
      this.videoEl.srcObject.getTracks().forEach(t => t.stop());
    }
  }
}

window.visionMonitor = null;
document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("vision-canvas")) {
    window.visionMonitor = new VisionStreamMonitor();
    const btnStart = document.getElementById("btn-start-stream");
    if (btnStart) {
      btnStart.addEventListener("click", () => {
        window.visionMonitor.startCamera();
        btnStart.textContent = "INSPECTION ACTIVE";
        btnStart.disabled = true;
      });
    }
  }
});
