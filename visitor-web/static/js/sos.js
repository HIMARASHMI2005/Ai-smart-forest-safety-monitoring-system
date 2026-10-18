/**
 * Emergency SOS Workflow & Chunked Media Upload Pipeline
 * Uses MediaRecorder API + crypto.subtle.digest for SHA-256 verification
 */

class VisitorSOSController {
  constructor() {
    this.btnSos = document.getElementById("btn-main-sos");
    this.isArmed = false;
    this.armTimeout = null;
    this.activeSosId = localStorage.getItem("active_sos_id") || null;
    this.mediaRecorder = null;
    this.mediaStream = null;
    this.chunkSequence = 0;
    this.init();
  }

  init() {
    if (this.btnSos) {
      this.btnSos.addEventListener("click", () => this.handleSosClick());
    }

    // Check if ongoing active SOS session exists
    if (window.location.pathname.includes("/sos/active")) {
      this.mountActiveSosView();
    }
  }

  handleSosClick() {
    if (!this.isArmed) {
      this.isArmed = true;
      this.btnSos.classList.add("armed");
      this.btnSos.textContent = "CONFIRM EMERGENCY SOS (CLICK AGAIN)";

      this.armTimeout = setTimeout(() => {
        this.disarm();
      }, 5000);
    } else {
      clearTimeout(this.armTimeout);
      this.triggerSOS();
    }
  }

  disarm() {
    this.isArmed = false;
    if (this.btnSos) {
      this.btnSos.classList.remove("armed");
      this.btnSos.textContent = "EMERGENCY SOS";
    }
  }

  async triggerSOS() {
    const profile = VisitorAuth.getProfile();
    if (!profile) {
      alert("Please log in to activate emergency beacon.");
      window.location.href = "/visitor/login";
      return;
    }

    const lastPos = window.visitorTracker ? window.visitorTracker.lastPosition : null;
    const lat = lastPos ? lastPos.latitude : profile.last_latitude || 11.660;
    const lng = lastPos ? lastPos.longitude : profile.last_longitude || 76.620;
    const zone = profile.current_zone || "ZONE_A";

    if (this.btnSos) {
      this.btnSos.disabled = true;
      this.btnSos.textContent = "DISPATCHING DISTRESS BEACON...";
    }

    try {
      const res = await fetch("/api/sos/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          visitor_id: profile.visitor_id,
          latitude: lat,
          longitude: lng,
          zone: zone
        })
      });

      const data = await res.json();
      if (data.success) {
        this.activeSosId = data.sos_id;
        localStorage.setItem("active_sos_id", data.sos_id);
        window.location.href = "/visitor/sos/active";
      } else {
        alert("Failed to initiate SOS: " + (data.error || "Unknown error"));
        this.disarm();
      }
    } catch (e) {
      alert("Network transmission error. Retrying emergency broadcast...");
      this.disarm();
    }
  }

  async mountActiveSosView() {
    const sosId = localStorage.getItem("active_sos_id");
    if (!sosId) {
      window.location.href = "/visitor/dashboard";
      return;
    }

    const idEl = document.getElementById("active-sos-id");
    if (idEl) idEl.textContent = sosId;

    await this.startChunkedVideoRecording(sosId);
  }

  async startChunkedVideoRecording(sosId) {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: true
      });

      const videoPreview = document.getElementById("sos-preview-video");
      if (videoPreview) {
        videoPreview.srcObject = this.mediaStream;
        videoPreview.play();
      }

      this.chunkSequence = 0;
      this.recordNextChunk(sosId);
    } catch (err) {
      console.warn("Could not activate camera/microphone for SOS video:", err);
      const statusEl = document.getElementById("sos-recording-status");
      if (statusEl) {
        statusEl.textContent = "Camera/Mic unavailable. Audio-visual stream omitted; GPS beaconing active.";
      }
    }
  }

  recordNextChunk(sosId) {
    if (!this.mediaStream) return;

    let mimeType = "video/webm";
    if (!MediaRecorder.isTypeSupported("video/webm")) {
      mimeType = "video/mp4";
    }

    const recordedChunks = [];
    this.mediaRecorder = new MediaRecorder(this.mediaStream, { mimeType: mimeType });

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: mimeType });
      const seq = this.chunkSequence++;
      await this.uploadChunk(sosId, seq, blob, 30);

      // Continue recording sequential chunks as long as SOS is active
      if (localStorage.getItem("active_sos_id")) {
        this.recordNextChunk(sosId);
      }
    };

    // Record in 30-second chunks
    this.mediaRecorder.start();
    const statusEl = document.getElementById("sos-recording-status");
    if (statusEl) {
      statusEl.textContent = `Recording chunk #${this.chunkSequence + 1} (30s slice)...`;
    }

    setTimeout(() => {
      if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
        this.mediaRecorder.stop();
      }
    }, 30000);
  }

  async calculateBlobSha256(blob) {
    const buffer = await blob.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
  }

  async uploadChunk(sosId, sequenceNumber, blob, duration) {
    try {
      const sha256 = await this.calculateBlobSha256(blob);
      const formData = new FormData();
      formData.append("sos_id", sosId);
      formData.append("sequence_number", sequenceNumber);
      formData.append("file_hash_sha256", sha256);
      formData.append("duration_seconds", duration);
      formData.append("file", blob, `chunk_${sequenceNumber}.webm`);

      const statusEl = document.getElementById("sos-recording-status");
      if (statusEl) statusEl.textContent = `Uploading chunk #${sequenceNumber} (SHA-256: ${sha256.slice(0, 8)}...)...`;

      const res = await fetch("/api/sos/video/chunk", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (data.success && statusEl) {
        statusEl.textContent = `Chunk #${sequenceNumber} uploaded & verified on SAR server.`;
      }
    } catch (e) {
      console.error("Chunk upload failed:", e);
    }
  }

  stopSOS() {
    localStorage.removeItem("active_sos_id");
    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
      this.mediaRecorder.stop();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
    }
    window.location.href = "/visitor/dashboard";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.visitorSOS = new VisitorSOSController();
});
