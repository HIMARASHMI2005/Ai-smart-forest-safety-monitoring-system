/**
 * Officer Navigation & SAR Radar Map (Leaflet.js + Spherical Geodesics)
 * Computes Haversine Distance, Forward Bearing Azimuth, and Tactical Navigation Vectors.
 */

function calcHaversineMeters(lat1, lon1, lat2, lon2) {
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

function calcBearingDegrees(lat1, lon1, lat2, lon2) {
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180;

  const y = Math.sin(deltaLambda) * Math.cos(phi2);
  const x = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  const theta = Math.atan2(y, x);
  return (theta * 180 / Math.PI + 360) % 360;
}

function getCompassSector(deg) {
  const sectors = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  const idx = Math.round(deg / 22.5) % 16;
  return sectors[idx];
}

class OfficerRescueMap {
  constructor(containerId = "officer-nav-map") {
    this.container = document.getElementById(containerId);
    this.map = null;
    this.officerMarker = null;
    this.targetMarker = null;
    this.vectorLine = null;
    this.bearingArrow = null;
    this.officersLayer = null;
    this.fieldOfficerMarkers = {};

    // Default Outpost Station in Bandipur Reserve
    this.defaultOutpost = { lat: 11.6620, lng: 76.6180 };
    this.rescuerPos = { ...this.defaultOutpost };
    this.targetPos = { lat: 11.6740, lng: 76.6310 };
    this.sosId = "ACTIVE-SAR";

    this.init();
  }

  async init() {
    if (!this.container) return;

    const urlParams = new URLSearchParams(window.location.search);
    const paramLat = parseFloat(urlParams.get("lat"));
    const paramLng = parseFloat(urlParams.get("lng"));
    const paramOLat = parseFloat(urlParams.get("olat"));
    const paramOLng = parseFloat(urlParams.get("olng"));
    this.sosId = urlParams.get("id") || "INCIDENT";

    if (!isNaN(paramLat) && !isNaN(paramLng)) {
      this.targetPos = { lat: paramLat, lng: paramLng };
    } else {
      // Try to fetch latest active SOS
      try {
        const res = await fetch("/api/sos/active");
        const activeList = await res.json();
        if (activeList.length > 0 && activeList[0].initial_lat && activeList[0].initial_lng) {
          this.targetPos = { lat: activeList[0].initial_lat, lng: activeList[0].initial_lng };
          this.sosId = activeList[0].sos_id;
        }
      } catch (e) {
        console.warn("Could not fetch active SOS, using default tactical target:", e);
      }
    }

    if (!isNaN(paramOLat) && !isNaN(paramOLng)) {
      this.rescuerPos = { lat: paramOLat, lng: paramOLng };
    }

    // Initialize map
    this.map = L.map(this.container).setView([this.targetPos.lat, this.targetPos.lng], 15);

    // Basemap Providers: OpenStreetMap, Google Maps, and Tactical Dark
    this.baseLayers = {
      "Google Satellite (Hybrid)": L.tileLayer("https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", {
        attribution: '&copy; Google Maps Satellite / Imagery',
        subdomains: ["mt0", "mt1", "mt2", "mt3"],
        maxZoom: 20
      }),
      "Google Terrain": L.tileLayer("https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", {
        attribution: '&copy; Google Maps Terrain',
        subdomains: ["mt0", "mt1", "mt2", "mt3"],
        maxZoom: 20
      }),
      "Google Roadmap": L.tileLayer("https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", {
        attribution: '&copy; Google Maps',
        subdomains: ["mt0", "mt1", "mt2", "mt3"],
        maxZoom: 20
      }),
      "OpenStreetMap (Standard)": L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        subdomains: ["a", "b", "c"],
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

    // Dedicated Field Officers Layer Group
    this.officersLayer = L.layerGroup().addTo(this.map);

    // Leaflet Base Layer Switcher Control
    this.layerControl = L.control.layers(this.baseLayers, {
      "Field Officers": this.officersLayer
    }, {
      position: "topright",
      collapsed: true
    }).addTo(this.map);

    // Ensure Leaflet recalculates dimensions once mounted
    window.addEventListener("resize", () => {
      if (this.map) this.map.invalidateSize();
    });
    setTimeout(() => {
      if (this.map) this.map.invalidateSize();
    }, 250);

    // Listen to custom select dropdown if present
    const basemapSelect = document.getElementById("select-officer-basemap");
    if (basemapSelect) {
      basemapSelect.addEventListener("change", (e) => {
        this.setBaseMap(e.target.value);
      });
    }

    // Load Zones
    try {
      const res = await fetch("/api/zones");
      const zones = await res.json();
      zones.forEach(z => {
        try {
          const geojson = JSON.parse(z.polygon_geojson);
          L.geoJSON(geojson, {
            style: { color: z.color_hex, weight: 2, fillOpacity: 0.15 }
          }).addTo(this.map);
        } catch (e) {}
      });
    } catch (e) {
      console.warn("Zone load warning:", e);
    }

    // Render Distressed Visitor Target Marker
    const distressIcon = L.divIcon({
      className: "distress-pin",
      html: `
        <div style="position: relative; width: 24px; height: 24px;">
          <div style="position: absolute; width: 24px; height: 24px; border-radius: 50%; background: rgba(217, 4, 41, 0.4); animation: pulse 1.5s infinite;"></div>
          <div style="position: absolute; top: 4px; left: 4px; width: 16px; height: 16px; border-radius: 50%; background: #D90429; border: 2px solid #FFF; box-shadow: 0 0 12px #D90429;"></div>
        </div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    this.targetMarker = L.marker([this.targetPos.lat, this.targetPos.lng], { icon: distressIcon })
      .addTo(this.map)
      .bindPopup(`<strong>INCIDENT: ${this.sosId}</strong><br/>DISTRESSED VISITOR TARGET<br/>[${this.targetPos.lat.toFixed(5)}, ${this.targetPos.lng.toFixed(5)}]`)
      .openPopup();

    // Determine Rescuer Initial Position
    this.resolveRescuerPosition();

    // Load All Active Field Officers immediately and poll every 4s
    this.loadFieldOfficers();
    setInterval(() => this.loadFieldOfficers(), 4000);
  }

  resolveRescuerPosition() {
    const handleSuccess = (lat, lng) => {
      this.rescuerPos = { lat, lng };
      this.renderRescuerAndVector();
    };

    // 1. If explicit olat / olng given in URL query params, use immediately
    const urlParams = new URLSearchParams(window.location.search);
    const pOLat = parseFloat(urlParams.get("olat"));
    const pOLng = parseFloat(urlParams.get("olng"));
    if (!isNaN(pOLat) && !isNaN(pOLng)) {
      handleSuccess(pOLat, pOLng);
      return;
    }

    // 2. Check window.officerTracker current position
    const trackerPos = window.officerTracker ? window.officerTracker.currentPosition : null;
    if (trackerPos && trackerPos.latitude && trackerPos.longitude) {
      handleSuccess(trackerPos.latitude, trackerPos.longitude);
      return;
    }

    // 3. Check browser Geolocation
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => handleSuccess(pos.coords.latitude, pos.coords.longitude),
        (err) => {
          console.warn("[SAR RADAR] Geolocation note (" + err.message + "). Checking profile or fallback.");
          const prof = typeof OfficerAuth !== "undefined" ? OfficerAuth.getProfile() : null;
          if (prof && prof.current_latitude && prof.current_longitude && Math.abs(prof.current_latitude) > 0.01) {
            handleSuccess(prof.current_latitude, prof.current_longitude);
          } else {
            handleSuccess(this.defaultOutpost.lat, this.defaultOutpost.lng);
          }
        },
        { enableHighAccuracy: true, timeout: 6000, maximumAge: 5000 }
      );
    } else {
      handleSuccess(this.defaultOutpost.lat, this.defaultOutpost.lng);
    }
  }

  async loadFieldOfficers() {
    if (!this.map || !this.officersLayer) return;
    try {
      const res = await fetch("/api/officer/list");
      if (!res.ok) return;
      const officers = await res.json();

      const currentProfile = typeof OfficerAuth !== "undefined" ? OfficerAuth.getProfile() : null;
      let activeCount = 0;
      let withGpsCount = 0;
      const seenIds = new Set();

      officers.forEach((off) => {
        const isCurrentRescuer = currentProfile && currentProfile.officer_id === off.officer_id;
        const hasValidCoords = off.current_latitude && off.current_longitude &&
          (Math.abs(off.current_latitude) > 0.01 || Math.abs(off.current_longitude) > 0.01);

        activeCount++;
        if (hasValidCoords) withGpsCount++;

        if (!hasValidCoords) {
          return;
        }

        seenIds.add(off.officer_id);

        const lat = off.current_latitude;
        const lng = off.current_longitude;

        const isAvailable = off.availability_status === "AVAILABLE";
        const isDispatched = off.availability_status === "DISPATCHED";
        const bg = isCurrentRescuer ? "#2563EB" : (isDispatched ? "#D97706" : "#059669");
        const border = isCurrentRescuer ? "#93C5FD" : (isDispatched ? "#FDE68A" : "#6EE7B7");
        const roleLabel = isCurrentRescuer ? "YOU" : (off.badge_number || "OFF");

        const pinIcon = L.divIcon({
          className: "tactical-officer-pin",
          html: `
            <div style="position: relative; display: flex; flex-direction: column; align-items: center; cursor: pointer;">
              <div style="
                padding: 3px 6px;
                border-radius: 4px;
                background: ${bg};
                border: 2px solid ${border};
                color: #FFF;
                font-family: Consolas, monospace;
                font-weight: bold;
                font-size: 10px;
                line-height: 12px;
                text-align: center;
                white-space: nowrap;
                box-shadow: 0 0 10px ${bg};
                display: flex;
                align-items: center;
                gap: 4px;
              ">
                <span style="font-size: 11px;">👮</span>
                <span>${roleLabel}</span>
              </div>
              <div style="
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid ${bg};
                margin-top: -1px;
              "></div>
            </div>
          `,
          iconSize: [60, 26],
          iconAnchor: [30, 26],
          popupAnchor: [0, -24]
        });

        const distToSos = calcHaversineMeters(lat, lng, this.targetPos.lat, this.targetPos.lng);
        const distText = distToSos >= 1000 ? `${(distToSos / 1000).toFixed(2)} km` : `${Math.round(distToSos)} m`;
        const statusColor = isAvailable ? "#10B981" : (isDispatched ? "#F59E0B" : "#94A3B8");

        const popupHtml = `
          <div style="font-family: Consolas, monospace; font-size: 11px; min-width: 200px; color: #1E293B;">
            <div style="color: #0F172A; font-weight: bold; font-size: 12px; border-bottom: 1px solid #CBD5E1; padding-bottom: 4px; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
              <span>👮 ${off.full_name}</span>
              <span style="background: ${bg}; color: #FFF; font-size: 9px; padding: 1px 5px; border-radius: 3px;">${isCurrentRescuer ? "YOU" : off.badge_number}</span>
            </div>
            <div style="margin-bottom: 3px;">BADGE: <strong>${off.badge_number}</strong></div>
            <div style="margin-bottom: 3px;">DEPT: <strong>${off.department || "Wildland SAR"}</strong></div>
            <div style="margin-bottom: 3px;">DUTY: <strong style="color: ${statusColor};">${off.availability_status}</strong></div>
            <div style="margin-bottom: 3px;">COORD: [${lat.toFixed(5)}, ${lng.toFixed(5)}]</div>
            <div style="margin-bottom: 3px; color: #2563EB;">DISTANCE TO SOS: <strong>${distText}</strong></div>
            <div style="font-size: 10px; color: #64748B;">PHONE: ${off.phone || "N/A"}</div>
          </div>
        `;

        if (this.fieldOfficerMarkers[off.officer_id]) {
          const marker = this.fieldOfficerMarkers[off.officer_id];
          marker.setLatLng([lat, lng]);
          marker.setIcon(pinIcon);
          marker.setPopupContent(popupHtml);
        } else {
          const marker = L.marker([lat, lng], { icon: pinIcon })
            .addTo(this.officersLayer)
            .bindPopup(popupHtml)
            .bindTooltip(`👮 ${off.badge_number} • ${off.full_name} [${off.availability_status}]`, {
              direction: "top",
              offset: [0, -24]
            });
          this.fieldOfficerMarkers[off.officer_id] = marker;
        }
      });

      // Cleanup stale markers
      Object.keys(this.fieldOfficerMarkers).forEach(id => {
        if (!seenIds.has(id)) {
          this.officersLayer.removeLayer(this.fieldOfficerMarkers[id]);
          delete this.fieldOfficerMarkers[id];
        }
      });

      // Update HUD counter
      const elCount = document.getElementById("hud-officers-count");
      if (elCount) {
        elCount.textContent = `${activeCount} UNITS (${withGpsCount} ON RADAR)`;
      }
    } catch (e) {
      console.warn("[SAR RADAR] Failed to load field officers:", e);
    }
  }

  renderRescuerAndVector() {
    const oLat = this.rescuerPos.lat;
    const oLng = this.rescuerPos.lng;
    const tLat = this.targetPos.lat;
    const tLng = this.targetPos.lng;

    // Rescuer Marker
    const officerIcon = L.divIcon({
      className: "officer-pin",
      html: `<div style="width: 22px; height: 22px; border-radius: 4px; background: #2563EB; border: 2px solid #93C5FD; color: #FFF; font-weight: bold; text-align: center; font-size: 11px; line-height: 18px; box-shadow: 0 0 10px #2563EB;">R</div>`,
      iconSize: [22, 22],
      iconAnchor: [11, 11]
    });

    if (this.officerMarker) {
      this.officerMarker.setLatLng([oLat, oLng]);
    } else {
      this.officerMarker = L.marker([oLat, oLng], { icon: officerIcon })
        .addTo(this.map)
        .bindPopup(`<strong>SAR RESCUER POSITION (YOU)</strong><br/>[${oLat.toFixed(5)}, ${oLng.toFixed(5)}]`);
    }

    // Mathematical Navigation Vector
    const distanceMeters = calcHaversineMeters(oLat, oLng, tLat, tLng);
    const bearingDeg = calcBearingDegrees(oLat, oLng, tLat, tLng);
    const cardinal = getCompassSector(bearingDeg);
    const etaMinutes = Math.max(1, Math.round(distanceMeters / 75)); // 4.5 km/h ≈ 75 m/min

    // Draw Vector Polyline
    if (this.vectorLine) {
      this.vectorLine.setLatLngs([[oLat, oLng], [tLat, tLng]]);
    } else {
      this.vectorLine = L.polyline([[oLat, oLng], [tLat, tLng]], {
        color: "#60A5FA",
        weight: 3,
        opacity: 0.9,
        dashArray: "6, 6"
      }).addTo(this.map);
    }

    // Mid-point Bearing Indicator Marker
    const midLat = (oLat + tLat) / 2;
    const midLng = (oLng + tLng) / 2;
    const arrowIcon = L.divIcon({
      className: "vector-arrow",
      html: `
        <div style="transform: rotate(${Math.round(bearingDeg)}deg); font-size: 16px; color: #60A5FA; font-weight: bold; text-shadow: 0 0 6px #000; width: 16px; height: 16px; line-height: 16px; text-align: center;">
          ➤
        </div>
      `,
      iconSize: [16, 16],
      iconAnchor: [8, 8]
    });

    if (this.bearingArrow) {
      this.bearingArrow.setLatLng([midLat, midLng]);
      this.bearingArrow.setIcon(arrowIcon);
    } else {
      this.bearingArrow = L.marker([midLat, midLng], { icon: arrowIcon }).addTo(this.map);
    }

    // Fit map bounds cleanly
    if (distanceMeters < 60) {
      this.map.setView([midLat, midLng], 18);
    } else {
      this.map.fitBounds([[oLat, oLng], [tLat, tLng]], { padding: [60, 60], maxZoom: 18 });
    }

    // Update Floating HUD Overlay
    const distText = distanceMeters >= 1000
      ? `${(distanceMeters / 1000).toFixed(2)} km`
      : `${Math.round(distanceMeters)} m`;

    const elTarget = document.getElementById("hud-target-id");
    const elBearing = document.getElementById("hud-bearing");
    const elDistance = document.getElementById("hud-distance");
    const elEta = document.getElementById("hud-eta");
    const elRescuer = document.getElementById("hud-rescuer-coord");

    if (elTarget) elTarget.textContent = this.sosId;
    if (elBearing) elBearing.textContent = `${Math.round(bearingDeg).toString().padStart(3, '0')}° ${cardinal}`;
    if (elDistance) elDistance.textContent = distText;
    if (elEta) elEta.textContent = `~${etaMinutes} MIN FOOT MARCH (~4.5 km/h)`;
    if (elRescuer) elRescuer.textContent = `[${oLat.toFixed(5)}, ${oLng.toFixed(5)}]`;
  }

  simulateMarch() {
    // Advance rescuer 20% closer to distressed target along vector
    const oLat = this.rescuerPos.lat;
    const oLng = this.rescuerPos.lng;
    const tLat = this.targetPos.lat;
    const tLng = this.targetPos.lng;

    const remaining = calcHaversineMeters(oLat, oLng, tLat, tLng);
    if (remaining < 25) {
      alert("RESCUER HAS ARRIVED AT INCIDENT TARGET SITE!");
      return;
    }

    this.rescuerPos.lat = oLat + (tLat - oLat) * 0.25;
    this.rescuerPos.lng = oLng + (tLng - oLng) * 0.25;
    this.renderRescuerAndVector();
  }

  snapToOutpost() {
    this.rescuerPos = { ...this.defaultOutpost };
    this.renderRescuerAndVector();
    if (this.map) {
      setTimeout(() => this.map.invalidateSize(), 100);
    }
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
      setTimeout(() => {
        if (this.map) this.map.invalidateSize();
      }, 100);
    }
  }

  openCompass() {
    window.location.href = `/officer/navigation?lat=${this.targetPos.lat}&lng=${this.targetPos.lng}&id=${this.sosId}&olat=${this.rescuerPos.lat}&olng=${this.rescuerPos.lng}`;
  }
}

window.officerMap = null;
document.addEventListener("DOMContentLoaded", () => {
  window.officerMap = new OfficerRescueMap();
});
