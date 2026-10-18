/**
 * 1-Click Biometric Face Login via Browser Camera
 */

class FaceLoginController {
  constructor() {
    this.videoEl = document.getElementById("login-face-video");
    this.canvasEl = document.getElementById("login-face-canvas");
    this.feedbackEl = document.getElementById("login-face-feedback");
    this.btnScan = document.getElementById("btn-verify-face");
    this.stream = null;
    this.init();
  }

  async init() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      if (this.videoEl) {
        this.videoEl.srcObject = this.stream;
        this.videoEl.play();
      }
    } catch (err) {
      if (this.feedbackEl) {
        this.feedbackEl.textContent = "Camera permission denied. Enable browser camera permission for face login.";
        this.feedbackEl.style.color = "#FF4D4D";
      }
    }

    if (this.btnScan) {
      this.btnScan.addEventListener("click", () => this.verifyFace());
    }
  }

  async verifyFace() {
    if (!this.videoEl || this.videoEl.readyState !== 4) return;
    this.btnScan.disabled = true;
    this.btnScan.textContent = "VERIFYING BIOMETRICS...";

    const ctx = this.canvasEl.getContext("2d");
    this.canvasEl.width = this.videoEl.videoWidth || 640;
    this.canvasEl.height = this.videoEl.videoHeight || 480;
    ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);

    const base64Img = this.canvasEl.toDataURL("image/jpeg", 0.85);

    try {
      const res = await fetch("/api/auth/face-verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Img })
      });

      const data = await res.json();
      if (data.verified) {
        this.feedbackEl.style.color = "#52B788";
        this.feedbackEl.textContent = `Identity Verified: ${data.visitor.full_name} (Similarity: ${(data.similarity * 100).toFixed(1)}%). Logging in...`;

        VisitorAuth.setSession(data.token, data.user, data.visitor);
        setTimeout(() => {
          window.location.href = "/visitor/dashboard";
        }, 1200);
      } else {
        this.feedbackEl.style.color = "#FF4D4D";
        this.feedbackEl.textContent = data.message || "Face verification failed. Please try again or use password login.";
      }
    } catch (err) {
      this.feedbackEl.style.color = "#FF4D4D";
      this.feedbackEl.textContent = "Error communicating with biometric gateway.";
    } finally {
      this.btnScan.disabled = false;
      this.btnScan.textContent = "SCAN & VERIFY IDENTITY";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("login-face-video")) {
    new FaceLoginController();
  }
});
