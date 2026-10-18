/**
 * Officer Tactical GPS Tracking
 * Continuous geolocation broadcast via navigator.geolocation.watchPosition()
 */

class OfficerGPSTracker {
  constructor() {
    this.watchId = null;
    this.profile = OfficerAuth.getProfile();
    this.currentPosition = null;
    this.init();
  }

  init() {
    if (!this.profile) return;
    if ("geolocation" in navigator) {
      this.watchId = navigator.geolocation.watchPosition(
        (pos) => this.handlePosition(pos),
        (err) => console.warn("Officer GPS warning:", err.message),
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 3000 }
      );
    }
  }

  handlePosition(pos) {
    const coords = pos.coords;
    this.currentPosition = {
      latitude: coords.latitude,
      longitude: coords.longitude,
      accuracy: coords.accuracy
    };

    // Update UI if present
    const latEl = document.getElementById("officer-lat");
    const lngEl = document.getElementById("officer-lng");
    if (latEl) latEl.textContent = coords.latitude.toFixed(5);
    if (lngEl) lngEl.textContent = coords.longitude.toFixed(5);

    // Transmit to gateway
    fetch("/api/officer/update-location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        officer_id: this.profile.officer_id,
        latitude: coords.latitude,
        longitude: coords.longitude,
        accuracy: coords.accuracy || 5.0,
        speed: (coords.speed || 0) * 3.6
      })
    }).catch(e => console.error("Officer ping failed:", e));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.officerTracker = new OfficerGPSTracker();
});
