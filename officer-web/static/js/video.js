/**
 * Officer Tactical SOS Video Assessment & Streamed Chunk Player
 */

class OfficerVideoReview {
  constructor() {
    this.videoEl = document.getElementById("sos-playback-video");
    this.playlistEl = document.getElementById("chunks-playlist");
    this.statusEl = document.getElementById("video-load-status");
    this.chunks = [];
    this.currentIndex = 0;
    this.init();
  }

  async init() {
    const urlParams = new URLSearchParams(window.location.search);
    const sosId = urlParams.get("sos_id");
    if (!sosId) {
      if (this.statusEl) this.statusEl.textContent = "No SOS ID specified in query.";
      return;
    }

    const sosHeaderEl = document.getElementById("video-sos-id");
    if (sosHeaderEl) sosHeaderEl.textContent = sosId;

    await this.loadVideoChunks(sosId);

    if (this.videoEl) {
      this.videoEl.addEventListener("ended", () => {
        if (this.currentIndex + 1 < this.chunks.length) {
          this.playChunk(this.currentIndex + 1);
        }
      });
    }
  }

  async loadVideoChunks(sosId) {
    try {
      const res = await fetch(`/api/sos/${sosId}/videos`);
      this.chunks = await res.json();

      if (this.chunks.length === 0) {
        if (this.statusEl) this.statusEl.textContent = "No video chunks uploaded yet for this incident. Standby for field stream...";
        return;
      }

      if (this.statusEl) this.statusEl.textContent = `Found ${this.chunks.length} video chunk(s).`;
      this.renderPlaylist();
      this.playChunk(0);
    } catch (e) {
      console.error("Failed to load chunks:", e);
    }
  }

  renderPlaylist() {
    if (!this.playlistEl) return;
    let html = "";
    this.chunks.forEach((c, idx) => {
      html += `
        <div onclick="window.officerVideo.playChunk(${idx})" style="padding: 8px; border-bottom: 1px solid var(--border-color); cursor: pointer; font-family: var(--font-mono); font-size: 11px;">
          <strong>CHUNK #${c.sequence_number}</strong> (${c.duration_seconds}s) &bull; SHA-256: ${c.file_hash_sha256.slice(0, 8)}...
        </div>
      `;
    });
    this.playlistEl.innerHTML = html;
  }

  playChunk(index) {
    if (index >= this.chunks.length) return;
    this.currentIndex = index;
    const chunk = this.chunks[index];

    if (this.videoEl) {
      this.videoEl.src = `/api/sos/video/stream/${chunk.id}`;
      this.videoEl.play();
    }
    if (this.statusEl) {
      this.statusEl.textContent = `Playing Chunk #${chunk.sequence_number} (${index + 1}/${this.chunks.length})`;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.officerVideo = new OfficerVideoReview();
});
