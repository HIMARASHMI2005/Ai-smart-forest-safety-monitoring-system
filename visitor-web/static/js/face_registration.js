/**
 * Guided Biometric Face Registration
 * Captures 5 Angles: Front, Left 15°, Right 15°, Up 10°, Down 10°
 * Enforces quality gating and submits to ForestFaceNet pipeline.
 */

class FaceRegistrationWizard {
  constructor() {
    this.videoEl = document.getElementById("face-video");
    this.canvasEl = document.getElementById("face-canvas");
    this.poseTitleEl = document.getElementById("pose-title");
    this.poseProgressEl = document.getElementById("pose-progress");
    this.feedbackEl = document.getElementById("face-feedback");
    this.btnCapture = document.getElementById("btn-capture-pose");

    this.poses = [
      { index: 0, name: "Pose 1 of 5: Front (Look Directly at Camera)" },
      { index: 1, name: "Pose 2 of 5: Turn Left 15°" },
      { index: 2, name: "Pose 3 of 5: Turn Right 15°" },
      { index: 3, name: "Pose 4 of 5: Tilt Up 10°" },
      { index: 4, name: "Pose 5 of 5: Tilt Down 10°" }
    ];
    this.currentPoseStep = 0;
    this.stream = null;

    this.init();
  }

  async init() {
    this.updateUI();
    await this.startCamera();
    if (this.btnCapture) {
      this.btnCapture.addEventListener("click", () => this.captureAndUpload());
    }
  }

  async startCamera() {
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
        this.feedbackEl.textContent = "Camera permission denied. Camera access is required for biometric face registration.";
        this.feedbackEl.style.color = "#FF4D4D";
      }
    }
  }

  updateUI() {
    if (this.currentPoseStep < this.poses.length) {
      const p = this.poses[this.currentPoseStep];
      if (this.poseTitleEl) this.poseTitleEl.textContent = p.name;
      if (this.poseProgressEl) this.poseProgressEl.style.width = `${((this.currentPoseStep) / 5) * 100}%`;
    }
  }

  async captureAndUpload() {
    if (!this.videoEl || this.videoEl.readyState !== 4) return;
    const token = VisitorAuth.getToken();
    if (!token) {
      alert("Session expired. Please log in first.");
      window.location.href = "/visitor/login";
      return;
    }

    this.btnCapture.disabled = true;
    this.btnCapture.textContent = "PROCESSING EMBEDDING...";

    const ctx = this.canvasEl.getContext("2d");
    this.canvasEl.width = this.videoEl.videoWidth || 640;
    this.canvasEl.height = this.videoEl.videoHeight || 480;
    ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);

    const base64Img = this.canvasEl.toDataURL("image/jpeg", 0.85);

    try {
      const res = await fetch("/api/auth/face-register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({
          image: base64Img,
          face_pose_index: this.currentPoseStep
        })
      });

      const data = await res.json();
      if (data.success) {
        this.feedbackEl.style.color = "#52B788";
        this.feedbackEl.textContent = `Pose ${this.currentPoseStep + 1} accepted (Quality: ${data.quality_score * 100}%).`;

        this.currentPoseStep++;
        if (this.currentPoseStep >= this.poses.length) {
          if (this.poseProgressEl) this.poseProgressEl.style.width = "100%";
          this.feedbackEl.textContent = "All 5 biometric poses registered! ForestFaceNet templates stored.";
          setTimeout(() => {
            window.location.href = "/visitor/dashboard";
          }, 1500);
        } else {
          this.updateUI();
        }
      } else {
        this.feedbackEl.style.color = "#FF4D4D";
        this.feedbackEl.textContent = `Quality check: ${data.error || "Please reposition face inside the guide."}`;
      }
    } catch (err) {
      this.feedbackEl.style.color = "#FF4D4D";
      this.feedbackEl.textContent = "Connection error. Retrying...";
    } finally {
      this.btnCapture.disabled = false;
      this.btnCapture.textContent = "CAPTURE & ENROLL POSE";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("face-video")) {
    window.faceWizard = new FaceRegistrationWizard();
  }
});
