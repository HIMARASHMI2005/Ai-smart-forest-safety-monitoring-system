import os
from pathlib import Path
from flask import Flask, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
from backend.app.config import Config
from backend.app.database.connection import init_db
from backend.app.websocket.events import socketio
from backend.app.api import register_api_routes

def create_app():
    # Base paths
    base_dir = Config.BASE_DIR
    admin_web_dir = base_dir / "admin-web"
    visitor_web_dir = base_dir / "visitor-web"
    officer_web_dir = base_dir / "officer-web"

    app = Flask(
        __name__,
        template_folder=str(admin_web_dir / "templates"),
        static_folder=str(admin_web_dir / "static")
    )
    app.config.from_object(Config)

    # Enable CORS for all REST APIs
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize Database Schema & Tables
    init_db()

    # Register API blueprints
    register_api_routes(app)

    # Initialize Flask-SocketIO
    socketio.init_app(app)

    # Static assets routes for Visitor & Officer web apps
    @app.route("/visitor/static/<path:filename>")
    def visitor_static(filename):
        return send_from_directory(visitor_web_dir / "static", filename)

    @app.route("/officer/static/<path:filename>")
    def officer_static(filename):
        return send_from_directory(officer_web_dir / "static", filename)

    @app.route("/admin/static/<path:filename>")
    def admin_static(filename):
        return send_from_directory(admin_web_dir / "static", filename)

    # =========================================================================
    # ADMIN WEB APPLICATION ROUTES
    # =========================================================================
    @app.route("/")
    @app.route("/admin")
    @app.route("/admin/login")
    @app.route("/login")
    @app.route("/dashboard")
    def admin_dashboard():
        return send_from_directory(admin_web_dir / "templates", "dashboard.html")

    @app.route("/live-map")
    def admin_live_map():
        return send_from_directory(admin_web_dir / "templates", "live_map.html")

    @app.route("/vision-monitor")
    def admin_vision_monitor():
        return send_from_directory(admin_web_dir / "templates", "vision_monitor.html")

    @app.route("/zones")
    def admin_zones():
        return send_from_directory(admin_web_dir / "templates", "zones.html")

    @app.route("/alerts")
    def admin_alerts():
        return send_from_directory(admin_web_dir / "templates", "alerts.html")

    @app.route("/visitors")
    def admin_visitors():
        return send_from_directory(admin_web_dir / "templates", "visitors.html")

    @app.route("/officers")
    @app.route("/officer-approvals")
    def admin_officers():
        return send_from_directory(admin_web_dir / "templates", "officers.html")

    @app.route("/ai-models")
    def admin_ai_models():
        return send_from_directory(admin_web_dir / "templates", "ai_models.html")

    @app.route("/logs")
    def admin_logs():
        return send_from_directory(admin_web_dir / "templates", "logs.html")

    # =========================================================================
    # VISITOR WEB APPLICATION ROUTES
    # =========================================================================
    @app.route("/visitor")
    @app.route("/visitor/dashboard")
    def visitor_dashboard():
        return send_from_directory(visitor_web_dir / "templates", "dashboard.html")

    @app.route("/visitor/login")
    def visitor_login():
        return send_from_directory(visitor_web_dir / "templates", "login.html")

    @app.route("/visitor/register")
    def visitor_register():
        return send_from_directory(visitor_web_dir / "templates", "register.html")

    @app.route("/visitor/face-registration")
    def visitor_face_registration():
        return send_from_directory(visitor_web_dir / "templates", "face_registration.html")

    @app.route("/visitor/face-login")
    def visitor_face_login():
        return send_from_directory(visitor_web_dir / "templates", "face_login.html")

    @app.route("/visitor/gps")
    def visitor_gps():
        return send_from_directory(visitor_web_dir / "templates", "gps.html")

    @app.route("/visitor/zone-status")
    def visitor_zone_status():
        return send_from_directory(visitor_web_dir / "templates", "zone_status.html")

    @app.route("/visitor/risk")
    def visitor_risk():
        return send_from_directory(visitor_web_dir / "templates", "risk.html")

    @app.route("/visitor/sos")
    def visitor_sos():
        return send_from_directory(visitor_web_dir / "templates", "sos.html")

    @app.route("/visitor/sos/active")
    def visitor_sos_active():
        return send_from_directory(visitor_web_dir / "templates", "sos_active.html")

    @app.route("/visitor/sos/history")
    def visitor_sos_history():
        return send_from_directory(visitor_web_dir / "templates", "sos_history.html")

    @app.route("/visitor/profile")
    def visitor_profile():
        return send_from_directory(visitor_web_dir / "templates", "profile.html")

    # =========================================================================
    # OFFICER WEB APPLICATION ROUTES
    # =========================================================================
    @app.route("/officer")
    @app.route("/officer/dashboard")
    def officer_dashboard():
        return send_from_directory(officer_web_dir / "templates", "dashboard.html")

    @app.route("/officer/login")
    def officer_login():
        return send_from_directory(officer_web_dir / "templates", "login.html")

    @app.route("/officer/register")
    def officer_register():
        return send_from_directory(officer_web_dir / "templates", "register.html")

    @app.route("/officer/approval")
    def officer_approval():
        return send_from_directory(officer_web_dir / "templates", "approval.html")

    @app.route("/officer/incidents")
    def officer_incidents():
        return send_from_directory(officer_web_dir / "templates", "incidents.html")

    @app.route("/officer/incidents/<incident_id>")
    def officer_incident_detail(incident_id):
        return send_from_directory(officer_web_dir / "templates", "incident_details.html")

    @app.route("/officer/map")
    def officer_map():
        return send_from_directory(officer_web_dir / "templates", "map.html")

    @app.route("/officer/navigation")
    def officer_navigation():
        return send_from_directory(officer_web_dir / "templates", "navigation.html")

    @app.route("/officer/video")
    def officer_video():
        return send_from_directory(officer_web_dir / "templates", "video.html")

    @app.route("/officer/resolve")
    def officer_resolve():
        return send_from_directory(officer_web_dir / "templates", "resolve.html")

    @app.route("/officer/profile")
    def officer_profile():
        return send_from_directory(officer_web_dir / "templates", "profile.html")

    return app
