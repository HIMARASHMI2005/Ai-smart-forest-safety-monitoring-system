/**
 * Tactical GIS Map Engine (Leaflet.js + GeoJSON + Real-Time Telemetry)
 */

class TacticalMap {
  constructor(mapContainerId = "tactical-map", initialCenter = [11.668, 76.625], initialZoom = 13) {
    this.containerId = mapContainerId;
    this.center = initialCenter;
    this.zoom = initialZoom;
    this.map = null;
    this.zoneLayer = null;
    this.visitorMarkers = {};
    this.officerMarkers = {};
    this.breadcrumbTrails = {};
    this.initMap();
  }

  initMap() {
    const el = document.getElementById(this.containerId);
    if (!el) return;

    this.map = L.map(this.containerId, {
      center: this.center,
      zoom: this.zoom,
      zoomControl: true
    });

    // Basemap Providers: OpenStreetMap, Google Maps, and Tactical Dark
    this.baseLayers = {
      "Google Satellite (Hybrid)": L.tileLayer("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", {
        attribution: '&copy; Google Maps Satellite / Imagery',
        maxZoom: 20
      }),
      "Google Terrain": L.tileLayer("https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", {
        attribution: '&copy; Google Maps Terrain',
        maxZoom: 20
      }),
      "Google Roadmap": L.tileLayer("https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", {
        attribution: '&copy; Google Maps',
        maxZoom: 20
      }),
      "OpenStreetMap (Standard)": L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        subdomains: "abc",
        maxZoom: 19
      }),
      "Tactical Dark (CartoDB)": L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; CARTO &copy; OpenStreetMap contributors',
        subdomains: "abcd",
        maxZoom: 19
      })
    };

    // Default to Tactical Dark
    this.currentBaseLayer = this.baseLayers["Tactical Dark (CartoDB)"];
    this.currentBaseLayer.addTo(this.map);

    // Leaflet Base Layer Switcher Control
    this.layerControl = L.control.layers(this.baseLayers, null, {
      position: "topright",
      collapsed: true
    }).addTo(this.map);

    // Listen to custom select dropdown if present
    const basemapSelect = document.getElementById("select-basemap");
    if (basemapSelect) {
      basemapSelect.addEventListener("change", (e) => {
        this.setBaseMap(e.target.value);
      });
    }

    this.loadZones();
    this.loadActiveVisitors();
    this.loadActiveOfficers();

    // Periodic telemetry sync fallback every 3 seconds (ensures zero data loss)
    setInterval(() => {
      this.loadActiveVisitors();
      this.loadActiveOfficers();
    }, 3000);
  }

  setBaseMap(providerKey) {
    if (!this.map || !this.baseLayers) return;
    const mapping = {
      "google_satellite": "Google Satellite (Hybrid)",
      "google_terrain": "Google Terrain",
      "google_streets": "Google Roadmap",
      "google_roadmap": "Google Roadmap",
      "google": "Google Satellite (Hybrid)",
      "osm": "OpenStreetMap (Standard)",
      "openstreetmap": "OpenStreetMap (Standard)",
      "tactical_dark": "Tactical Dark (CartoDB)",
      "dark": "Tactical Dark (CartoDB)"
    };
    const key = mapping[providerKey.toLowerCase()] || providerKey;
    if (this.baseLayers[key]) {
      if (this.currentBaseLayer) {
        this.map.removeLayer(this.currentBaseLayer);
      }
      this.currentBaseLayer = this.baseLayers[key];
      this.currentBaseLayer.addTo(this.map);
    }
  }

  async loadZones() {
    try {
      const res = await fetch("/api/zones");
      const zones = await res.json();

      zones.forEach(zone => {
        try {
          const geojson = JSON.parse(zone.polygon_geojson);
          const color = zone.color_hex || "#2D6A4F";

          const poly = L.geoJSON(geojson, {
            style: {
              color: color,
              weight: 2,
              opacity: 0.9,
              fillColor: color,
              fillOpacity: 0.22,
              dashArray: zone.code === "ZONE_B" ? "4, 4" : null
            }
          }).addTo(this.map);

          poly.bindPopup(`
            <div style="font-family: Consolas, monospace; font-size: 11px;">
              <strong style="color: ${color}; text-transform: uppercase;">${zone.name}</strong><br/>
              CODE: <strong>${zone.code}</strong><br/>
              SEVERITY: <strong>${zone.severity}</strong><br/>
              STATUS: <strong>${zone.active ? "ACTIVE ENFORCEMENT" : "INACTIVE"}</strong>
            </div>
          `);
        } catch (e) {
          console.error("Failed to parse zone polygon:", zone.id, e);
        }
      });
    } catch (e) {
      console.error("Failed to load zones:", e);
    }
  }

  async loadActiveVisitors() {
    try {
      const res = await fetch("/api/location/live-all");
      const visitors = await res.json();
      visitors.forEach(v => {
        if (v.last_latitude && v.last_longitude && (Math.abs(v.last_latitude) > 0.01)) {
          this.updateVisitorMarker({
            visitor_id: v.visitor_id,
            full_name: v.full_name,
            phone: v.phone,
            emergency_contact_phone: v.emergency_contact_phone,
            latitude: v.last_latitude,
            longitude: v.last_longitude,
            current_zone: v.current_zone,
            current_risk_level: v.current_risk_level,
            risk_score: v.risk_score || 10,
            speed: v.speed || 0.0
          });
        }
      });
    } catch (e) {
      console.error("Failed to load live visitors:", e);
    }
  }

  updateVisitorMarker(telemetry) {
    if (!this.map) return;
    const { visitor_id, full_name, phone, emergency_contact_phone, latitude, longitude, current_zone, current_risk_level, risk_score, speed } = telemetry;
    if (!latitude || !longitude || (Math.abs(latitude) < 0.01)) return;

    let markerColor = "#2D6A4F";
    if (current_risk_level === "CRITICAL") markerColor = "#D90429";
    else if (current_risk_level === "HIGH") markerColor = "#C1121F";
    else if (current_risk_level === "MEDIUM") markerColor = "#E09F3E";

    // High-visibility Tactical Pin with Name Tag
    const customIcon = L.divIcon({
      className: "custom-visitor-pin",
      html: `
        <div style="position: relative; display: flex; flex-direction: column; align-items: center; transform: translate(-50%, -100%);">
          <div style="background: rgba(18, 22, 20, 0.95); border: 1px solid ${markerColor}; color: #FFF; font-family: Consolas, monospace; font-size: 9px; font-weight: bold; padding: 2px 5px; border-radius: 3px; white-space: nowrap; margin-bottom: 2px; box-shadow: 0 2px 6px rgba(0,0,0,0.8);">
            ${full_name || visitor_id}
          </div>
          <div style="width: 14px; height: 14px; border-radius: 50%; background: ${markerColor}; border: 2px solid #FFF; box-shadow: 0 0 10px ${markerColor};"></div>
        </div>
      `,
      iconSize: [14, 14],
      iconAnchor: [0, 0]
    });

    const popupHtml = `
      <div style="font-family: Consolas, monospace; font-size: 11px; line-height: 1.6;">
        <strong style="color: #FFF; font-size: 12px;">VISITOR: ${full_name || "Registered"}</strong><br/>
        ID: <strong style="color: #93C5FD;">${visitor_id}</strong><br/>
        GPS COORD: [${latitude.toFixed(5)}, ${longitude.toFixed(5)}]<br/>
        ZONE: <strong style="color: ${markerColor}">${current_zone}</strong><br/>
        RISK TIER: <strong style="color: ${markerColor}">${current_risk_level} (${risk_score || 0})</strong><br/>
        SPEED: <strong>${speed ? speed.toFixed(1) + " km/h" : "STATIONARY"}</strong><br/>
        PHONE: <strong>${phone || "N/A"}</strong> | SOS: <strong>${emergency_contact_phone || "N/A"}</strong>
      </div>
    `;

    if (this.visitorMarkers[visitor_id]) {
      this.visitorMarkers[visitor_id].setLatLng([latitude, longitude]);
      this.visitorMarkers[visitor_id].setIcon(customIcon);
      this.visitorMarkers[visitor_id].setPopupContent(popupHtml);
    } else {
      const marker = L.marker([latitude, longitude], { icon: customIcon }).addTo(this.map);
      marker.bindPopup(popupHtml);
      this.visitorMarkers[visitor_id] = marker;
    }

    // Update historical breadcrumb polyline
    if (!this.breadcrumbTrails[visitor_id]) {
      this.breadcrumbTrails[visitor_id] = L.polyline([], {
        color: markerColor,
        weight: 2,
        opacity: 0.65,
        dashArray: "4, 4"
      }).addTo(this.map);
    }
    this.breadcrumbTrails[visitor_id].addLatLng([latitude, longitude]);
  }

  flyToVisitor(visitorId, lat, lng) {
    if (!this.map) return;
    if (this.visitorMarkers[visitorId]) {
      const marker = this.visitorMarkers[visitorId];
      this.map.flyTo(marker.getLatLng(), 15, { duration: 1.2 });
      marker.openPopup();
    } else if (lat && lng) {
      this.map.flyTo([lat, lng], 15, { duration: 1.2 });
    }
  }


  updateOfficerMarker(officer) {
    if (!this.map) return;
    const { officer_id, full_name, badge_number, current_latitude, current_longitude, availability_status } = officer;

    const officerIcon = L.divIcon({
      className: "custom-officer-pin",
      html: `
        <div style="
          width: 16px;
          height: 16px;
          border-radius: 3px;
          background: #1D4ED8;
          border: 2px solid #93C5FD;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 9px;
          font-family: Consolas, monospace;
          color: #FFF;
          font-weight: bold;
        ">R</div>
      `,
      iconSize: [16, 16],
      iconAnchor: [8, 8]
    });

    const popupHtml = `
      <div style="font-family: Consolas, monospace; font-size: 11px;">
        <strong>RESCUER: ${full_name}</strong><br/>
        BADGE: <strong>${badge_number}</strong><br/>
        STATUS: <strong>${availability_status}</strong><br/>
        COORD: [${current_latitude.toFixed(5)}, ${current_longitude.toFixed(5)}]
      </div>
    `;

    if (this.officerMarkers[officer_id]) {
      this.officerMarkers[officer_id].setLatLng([current_latitude, current_longitude]);
      this.officerMarkers[officer_id].setPopupContent(popupHtml);
    } else {
      const marker = L.marker([current_latitude, current_longitude], { icon: officerIcon }).addTo(this.map);
      marker.bindPopup(popupHtml);
      this.officerMarkers[officer_id] = marker;
    }
  }

  async loadActiveOfficers() {
    if (!this.map) return;
    try {
      const res = await fetch("/api/officer/list");
      if (!res.ok) return;
      const officers = await res.json();
      officers.forEach(off => {
        if (off.current_latitude && off.current_longitude &&
            (Math.abs(off.current_latitude) > 0.01 || Math.abs(off.current_longitude) > 0.01)) {
          this.updateOfficerMarker(off);
        }
      });
    } catch (e) {
      console.warn("Failed to load active officers:", e);
    }
  }
}

window.tacticalMap = null;
document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("tactical-map")) {
    window.tacticalMap = new TacticalMap();

    // Hook socket listener
    if (window.smartSocket) {
      window.smartSocket.onVisitorLocation((data) => {
        window.tacticalMap.updateVisitorMarker(data);
      });
      window.smartSocket.onOfficerLocation((data) => {
        window.tacticalMap.updateOfficerMarker(data);
      });
    }
  }
});
