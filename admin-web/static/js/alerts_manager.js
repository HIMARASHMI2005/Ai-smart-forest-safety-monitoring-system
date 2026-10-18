/**
 * Operational Alerts & Incident Triage Manager
 */

class AlertsManager {
  constructor(containerId = "alerts-container") {
    this.container = document.getElementById(containerId);
    this.audioAlert = null;
    this.init();
  }

  init() {
    this.loadAlerts();
    if (window.smartSocket) {
      window.smartSocket.onEmergencyAlert((data) => {
        this.renderEmergencyBanner(data);
        this.loadAlerts();
      });
      window.smartSocket.onIncidentState(() => {
        this.loadAlerts();
      });
    }
  }

  async loadAlerts() {
    if (!this.container) return;
    try {
      const res = await fetch("/api/alerts");
      const alerts = await res.json();
      this.renderAlertsList(alerts);
    } catch (e) {
      console.error("Failed to load alerts:", e);
    }
  }

  renderAlertsList(alerts) {
    if (!this.container) return;
    if (alerts.length === 0) {
      this.container.innerHTML = `
        <div style="color: var(--text-muted); font-family: Consolas, monospace; padding: 20px; text-align: center;">
          NO UNRESOLVED OPERATIONAL ALERTS. SYSTEM NOMINAL.
        </div>
      `;
      return;
    }

    let html = "";
    alerts.forEach(a => {
      const isCritical = a.severity === "CRITICAL";
      const isResolved = a.status === "RESOLVED";
      const severityClass = a.severity.toLowerCase();

      html += `
        <div class="incident-item ${severityClass}" id="alert-item-${a.id}">
          <div class="incident-title-row">
            <span class="incident-title">${a.title}</span>
            <span class="badge badge-${severityClass}">${a.severity}</span>
          </div>
          <p style="font-size: 12px; margin-bottom: 6px; color: var(--text-primary);">${a.description}</p>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="incident-meta">${new Date(a.created_at).toLocaleTimeString()} | STATUS: ${a.status}</span>
            <div>
              ${!isResolved ? `
                <button class="btn-tactical btn-safe" onclick="window.alertsManager.resolveAlert(${a.id})" style="padding: 2px 8px; font-size: 10px;">MARK RESOLVED</button>
              ` : `
                <span class="badge badge-neutral">RESOLVED</span>
              `}
            </div>
          </div>
        </div>
      `;
    });
    this.container.innerHTML = html;
  }

  renderEmergencyBanner(data) {
    const alertBox = document.getElementById("emergency-top-banner");
    if (!alertBox) return;

    alertBox.style.display = "flex";
    alertBox.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px; width: 100%; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="badge badge-critical" style="font-size: 12px; animation: pulse 1s infinite;">CRITICAL MAYDAY</span>
          <strong style="font-family: Consolas, monospace; color: #FFF;">
            EMERGENCY SOS: ${data.visitor?.full_name || data.visitor_id} | ZONE: ${data.initial_zone || data.current_zone}
          </strong>
        </div>
        <div>
          <a href="/officer/incidents" class="btn-tactical btn-danger" style="margin-right: 8px;">DISPATCH SAR</a>
          <button class="btn-tactical" onclick="document.getElementById('emergency-top-banner').style.display='none'">DISMISS</button>
        </div>
      </div>
    `;
  }

  async resolveAlert(alertId) {
    try {
      const res = await fetch(`/api/alerts/${alertId}/resolve`, { method: "POST" });
      if (res.ok) {
        this.loadAlerts();
      }
    } catch (e) {
      console.error("Failed to resolve alert:", e);
    }
  }
}

window.alertsManager = null;
document.addEventListener("DOMContentLoaded", () => {
  window.alertsManager = new AlertsManager();
});
