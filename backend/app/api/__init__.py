from backend.app.api.auth_routes import auth_bp
from backend.app.api.location_routes import location_bp
from backend.app.api.zone_routes import zone_bp
from backend.app.api.sos_routes import sos_bp
from backend.app.api.officer_routes import officer_bp
from backend.app.api.visitor_routes import visitor_bp
from backend.app.api.vision_routes import vision_bp
from backend.app.api.alert_routes import alert_bp

def register_api_routes(app):
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(location_bp, url_prefix="/api/location")
    app.register_blueprint(zone_bp, url_prefix="/api/zones")
    app.register_blueprint(sos_bp, url_prefix="/api/sos")
    app.register_blueprint(officer_bp, url_prefix="/api/officer")
    app.register_blueprint(visitor_bp, url_prefix="/api/visitor")
    app.register_blueprint(vision_bp, url_prefix="/api/vision")
    app.register_blueprint(alert_bp, url_prefix="/api/alerts")
