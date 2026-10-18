/**
 * Smart Forest Operational System - WebSocket Client
 * Event-Driven Architecture (Zero Polling)
 */

class SmartForestSocket {
  constructor() {
    this.socket = null;
    this.handlers = {
      visitor_location: [],
      officer_location: [],
      zone_transition: [],
      emergency_alert: [],
      video_chunk: [],
      incident_state: []
    };
    this.connected = false;
  }

  connect(room = "admin_control_room") {
    if (typeof io === "undefined") {
      console.warn("Socket.IO library not loaded yet.");
      return;
    }

    const token = localStorage.getItem("token") || "";
    this.socket = io({
      query: { token: token },
      transports: ["websocket", "polling"]
    });

    this.socket.on("connect", () => {
      this.connected = true;
      console.log("[WSS] Connected to Operational Gateway. Joining room:", room);
      this.socket.emit("join_room", { room: room });
      const indicator = document.getElementById("socket-indicator");
      if (indicator) {
        indicator.classList.remove("alert");
        indicator.classList.add("online");
        indicator.title = "WSS Gateway Connected";
      }
    });

    this.socket.on("disconnect", () => {
      this.connected = false;
      console.warn("[WSS] Gateway connection lost. Reconnecting...");
      const indicator = document.getElementById("socket-indicator");
      if (indicator) {
        indicator.classList.remove("online");
        indicator.classList.add("alert");
        indicator.title = "WSS Disconnected";
      }
    });

    this.socket.on("visitor_location_update", (data) => {
      this.handlers.visitor_location.forEach(cb => cb(data));
    });

    this.socket.on("officer_location_update", (data) => {
      this.handlers.officer_location.forEach(cb => cb(data));
    });

    this.socket.on("zone_transition", (data) => {
      this.handlers.zone_transition.forEach(cb => cb(data));
    });

    this.socket.on("emergency_alert", (data) => {
      this.handlers.emergency_alert.forEach(cb => cb(data));
    });

    this.socket.on("sos_video_chunk_received", (data) => {
      this.handlers.video_chunk.forEach(cb => cb(data));
    });

    this.socket.on("incident_state_changed", (data) => {
      this.handlers.incident_state.forEach(cb => cb(data));
    });
  }

  onVisitorLocation(cb) { this.handlers.visitor_location.push(cb); }
  onOfficerLocation(cb) { this.handlers.officer_location.push(cb); }
  onZoneTransition(cb) { this.handlers.zone_transition.push(cb); }
  onEmergencyAlert(cb) { this.handlers.emergency_alert.push(cb); }
  onVideoChunk(cb) { this.handlers.video_chunk.push(cb); }
  onIncidentState(cb) { this.handlers.incident_state.push(cb); }
}

window.smartSocket = new SmartForestSocket();
