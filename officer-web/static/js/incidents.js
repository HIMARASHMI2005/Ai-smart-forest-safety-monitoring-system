/**
 * Officer Tactical Incident Radar & SAR Triage
 */

function haversineDistanceMeters(lat1, lon1, lat2, lon2) {
  const R = 6371e3; // Earth radius in meters
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaPhi = ((lat2 - lat1) * Math.PI) / 180;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(deltaPhi / 2) * Math.sin(deltaPhi / 2) +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) * Math.sin(deltaLambda / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

class OfficerIncidents {
  constructor(containerId = "officer-incident-feed") {
    this.container = document.getElementById(containerId);
    this.profile = OfficerAuth.getProfile();
    this.init();
  }

  async init() {
    await this.loadIncidents();

    if (typeof io !== "undefined") {
      const socket = io({ transports: ["websocket", "polling"] });
      socket.on("connect", () => {
        socket.emit("join_room", { room: "admin_control_room" });
        if (this.profile) {
          socket.emit("join_room", { room: `officer_channel_${this.profile.officer_id}` });
        }
      });
      socket.on("emergency_alert", () => this.loadIncidents());
      socket.on("incident_state_changed", () => this.loadIncidents());
    }
  }

  async loadIncidents() {
    if (!this.container) return;
    try {
      const res = await fetch("/api/sos/active");
      const incidents = await res.json();
      this.renderFeed(incidents);
    } catch (e) {
      console.error("Failed to load incidents:", e);
    }
  }

  renderFeed(incidents) {
    if (!this.container) return;
    if (incidents.length === 0) {
      this.container.innerHTML = `
        <div style="color: var(--text-muted); font-family: Consolas, monospace; text-align: center; padding: 24px;">
          NO ACTIVE SEARCH & RESCUE INCIDENTS. SECTOR PATROL NOMINAL.
        </div>
      `;
      return;
    }

    const officerPos = window.officerTracker ? window.officerTracker.currentPosition : null;

    let html = "";
    incidents.forEach(inc => {
      let distText = "DISTANCE CALC...";
      if (officerPos && inc.initial_lat && inc.initial_lng) {
        const dMeters = haversineDistanceMeters(
          officerPos.latitude, officerPos.longitude,
          inc.initial_lat, inc.initial_lng
        );
        distText = dMeters > 1000 ? `${(dMeters / 1000).toFixed(2)} km` : `${Math.round(dMeters)} m`;
      }

      const isAck = inc.status === "ACKNOWLEDGED" || inc.status === "DISPATCHED";
      const oPosParam = (officerPos && officerPos.latitude) ? `&olat=${officerPos.latitude}&olng=${officerPos.longitude}` : '';

      html += `
        <div class="incident-card ${isAck ? 'ack' : ''}">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <strong style="color: #FFF; font-size: 14px;">${inc.sos_id} &bull; ${inc.visitor?.full_name || inc.visitor_id}</strong>
            <span class="badge badge-critical">${inc.status}</span>
          </div>
          <div style="font-family: Consolas, monospace; font-size: 11px; color: var(--text-muted); line-height: 1.6; margin-bottom: 10px;">
            <div>DISTANCE: <strong style="color: #60A5FA;">${distText}</strong> | ZONE: <strong>${inc.initial_zone}</strong></div>
            <div>COORDINATES: [${inc.initial_lat.toFixed(5)}, ${inc.initial_lng.toFixed(5)}]</div>
            <div>PHONE: <strong>${inc.visitor?.phone || "N/A"}</strong> | EMERGENCY: <strong>${inc.visitor?.emergency_contact_phone || "N/A"}</strong></div>
            <div>TRIGGERED: ${new Date(inc.triggered_at).toLocaleTimeString()}</div>
          </div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            ${!isAck ? `
              <button class="btn btn-rescuer" onclick="window.officerIncidents.acknowledge('${inc.sos_id}')">ACKNOWLEDGE DISPATCH</button>
            ` : `
              <span class="badge badge-safe">DISPATCH ACKNOWLEDGED</span>
            `}
            <a href="/officer/map?lat=${inc.initial_lat}&lng=${inc.initial_lng}&id=${inc.sos_id}${oPosParam}" class="btn">RADAR MAP</a>
            <a href="/officer/video?sos_id=${inc.sos_id}" class="btn">REVIEW SOS VIDEO</a>
            <a href="/officer/resolve?sos_id=${inc.sos_id}" class="btn btn-safe">RESOLVE</a>
          </div>
        </div>
      `;
    });
    this.container.innerHTML = html;
  }

  async acknowledge(sosId) {
    if (!this.profile) return;
    try {
      const res = await fetch("/api/officer/acknowledge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sos_id: sosId,
          officer_id: this.profile.officer_id
        })
      });
      if (res.ok) {
        this.loadIncidents();
      }
    } catch (e) {
      console.error("Acknowledge failed:", e);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.officerIncidents = new OfficerIncidents();
});
