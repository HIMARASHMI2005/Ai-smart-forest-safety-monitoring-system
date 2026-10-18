/**
 * Continuous Web GPS Tracking & Store-and-Forward Offline Queue
 * Uses navigator.geolocation.watchPosition()
 */

class VisitorGPSTracker {
  constructor() {
    this.watchId = null;
    this.visitorProfile = VisitorAuth.getProfile();
    this.offlineQueueKey = "visitor_offline_queue";
    this.lastPosition = null;
    this.isTracking = false;
    this.init();
  }

  init() {
    if (!this.visitorProfile) {
      console.warn("No active visitor profile found in session.");
      return;
    }

    // Flush any pending offline queue on init
    this.syncOfflineQueue();
    window.addEventListener("online", () => this.syncOfflineQueue());

    this.startTracking();
  }

  startTracking() {
    if (!("geolocation" in navigator)) {
      this.updateGPSUI({ error: "Geolocation API not supported by browser." });
      return;
    }

    const options = {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 2000
    };

    this.watchId = navigator.geolocation.watchPosition(
      (pos) => this.handlePosition(pos),
      (err) => this.handleError(err),
      options
    );
    this.isTracking = true;
  }

  toggleSimulation() {
    if (this.isSimulating) {
      this.stopSimulation();
    } else {
      this.startSimulation();
    }
  }

  startSimulation() {
    this.isSimulating = true;
    console.log("[GPS] Starting simulated trail patrol walk...");

    // Initial trail waypoint in Zone A (Eco Trail)
    let curLat = 11.6605;
    let curLng = 76.6210;
    let step = 0;

    const btn = document.getElementById("btn-toggle-sim-gps");
    if (btn) {
      btn.textContent = "STOP PATROL SIM";
      btn.className = "btn btn-danger";
    }

    this.simInterval = setInterval(() => {
      step++;
      // Walk along the trail toward Zone B
      curLat += 0.00035 + (Math.random() - 0.5) * 0.00008;
      curLng += 0.00028 + (Math.random() - 0.5) * 0.00008;

      const payload = {
        visitor_id: this.visitorProfile.visitor_id,
        latitude: curLat,
        longitude: curLng,
        altitude: 850.0,
        accuracy: 4.2,
        speed: 4.8 + Math.random() * 0.5,
        bearing: 42.0,
        timestamp: new Date().toISOString()
      };

      this.lastPosition = payload;
      this.updateGPSUI(payload);
      this.transmitTelemetry(payload);
    }, 2500);
  }

  stopSimulation() {
    this.isSimulating = false;
    if (this.simInterval) clearInterval(this.simInterval);
    const btn = document.getElementById("btn-toggle-sim-gps");
    if (btn) {
      btn.textContent = "SIMULATE PATROL GPS";
      btn.className = "btn";
    }
    const statusEl = document.getElementById("gps-lock-status");
    if (statusEl) {
      statusEl.textContent = "GPS STANDBY";
      statusEl.className = "badge badge-muted";
    }
  }

  handlePosition(position) {
    if (this.isSimulating) return; // Prefer active simulation if engaged
    const coords = position.coords;
    const payload = {
      visitor_id: this.visitorProfile.visitor_id,
      latitude: coords.latitude,
      longitude: coords.longitude,
      altitude: coords.altitude || 0.0,
      accuracy: coords.accuracy || 5.0,
      speed: coords.speed ? (coords.speed * 3.6) : 0.0, // Convert m/s to km/h
      bearing: coords.heading || 0.0,
      timestamp: new Date(position.timestamp).toISOString()
    };

    this.lastPosition = payload;
    this.updateGPSUI(payload);
    this.transmitTelemetry(payload);
  }

  handleError(err) {
    let msg = "Live GPS unavailable";
    if (err.code === 1) {
      msg = "Location permission denied. Click 'SIMULATE PATROL GPS' to test live movement.";
    } else if (err.code === 2) {
      msg = "Live GPS searching... Click 'SIMULATE PATROL GPS' to test live movement.";
    } else if (err.code === 3) {
      msg = "GPS timed out. Click 'SIMULATE PATROL GPS' to test live movement.";
    }
    this.updateGPSUI({ error: msg });
  }

  async transmitTelemetry(payload) {
    if (!navigator.onLine) {
      this.enqueueOffline(payload);
      return;
    }

    try {
      const res = await fetch("/api/location/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        this.enqueueOffline(payload);
      } else {
        const data = await res.json();
        this.updateZoneRiskUI(data);
      }
    } catch (e) {
      this.enqueueOffline(payload);
    }
  }

  enqueueOffline(payload) {
    const queue = JSON.parse(localStorage.getItem(this.offlineQueueKey) || "[]");
    queue.push(payload);
    localStorage.setItem(this.offlineQueueKey, JSON.stringify(queue));
    const syncBadge = document.getElementById("sync-status-badge");
    if (syncBadge) {
      syncBadge.textContent = `OFFLINE (${queue.length} QUEUED)`;
      syncBadge.className = "badge badge-warning";
    }
  }

  async syncOfflineQueue() {
    const queue = JSON.parse(localStorage.getItem(this.offlineQueueKey) || "[]");
    if (queue.length === 0) return;

    console.log(`[STORE-AND-FORWARD] Synchronizing ${queue.length} buffered telemetry pings...`);
    const remaining = [];

    for (const item of queue) {
      try {
        const res = await fetch("/api/location/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item)
        });
        if (!res.ok) remaining.push(item);
      } catch (e) {
        remaining.push(item);
      }
    }

    localStorage.setItem(this.offlineQueueKey, JSON.stringify(remaining));
    const syncBadge = document.getElementById("sync-status-badge");
    if (syncBadge) {
      if (remaining.length === 0) {
        syncBadge.textContent = "ONLINE & SYNCED";
        syncBadge.className = "badge badge-safe";
      } else {
        syncBadge.textContent = `${remaining.length} PENDING SYNC`;
      }
    }
  }

  updateGPSUI(data) {
    const latEl = document.getElementById("telemetry-lat");
    const lngEl = document.getElementById("telemetry-lng");
    const accEl = document.getElementById("telemetry-acc");
    const spdEl = document.getElementById("telemetry-spd");
    const statusEl = document.getElementById("gps-lock-status");

    if (data.error) {
      if (statusEl) {
        statusEl.textContent = data.error;
        statusEl.className = "badge badge-danger";
      }
      return;
    }

    if (statusEl) {
      statusEl.textContent = "SATELLITE LOCK ACTIVE";
      statusEl.className = "badge badge-safe";
    }
    if (latEl) latEl.textContent = data.latitude.toFixed(5);
    if (lngEl) lngEl.textContent = data.longitude.toFixed(5);
    if (accEl) accEl.textContent = `±${Math.round(data.accuracy)} m`;
    if (spdEl) spdEl.textContent = `${data.speed.toFixed(1)} km/h`;
  }

  updateZoneRiskUI(data) {
    const zoneEl = document.getElementById("telemetry-zone");
    const riskBadge = document.getElementById("telemetry-risk-badge");
    const riskScoreEl = document.getElementById("telemetry-risk-score");
    const riskReasonsEl = document.getElementById("telemetry-risk-reasons");

    if (zoneEl) zoneEl.textContent = data.current_zone;

    if (riskScoreEl) riskScoreEl.textContent = `${data.risk_score}/100`;

    if (riskBadge) {
      riskBadge.textContent = data.risk_level;
      let badgeClass = "badge-safe";
      if (data.risk_level === "CRITICAL") badgeClass = "badge-critical";
      else if (data.risk_level === "HIGH") badgeClass = "badge-danger";
      else if (data.risk_level === "MEDIUM") badgeClass = "badge-warning";
      riskBadge.className = `badge ${badgeClass}`;
    }

    if (riskReasonsEl && data.risk_reasons) {
      riskReasonsEl.innerHTML = data.risk_reasons.map(r => `<div>• ${r}</div>`).join("");
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.visitorTracker = new VisitorGPSTracker();
});
