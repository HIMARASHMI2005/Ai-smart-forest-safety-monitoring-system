import jwt
from functools import wraps
from datetime import datetime, timedelta
from flask import request, jsonify, g
from sqlalchemy.orm import joinedload
from backend.app.config import Config
from backend.app.database.connection import SessionLocal
from backend.app.database.models import User

def generate_jwt_token(user: User) -> str:
    payload = {
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "exp": datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        token = None

        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            elif len(parts) == 1:
                token = parts[0]

        if not token:
            # Check query param as fallback for stream/socket
            token = request.args.get("token")

        if not token:
            return jsonify({"error": "Authorization token is missing"}), 401

        try:
            payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
            db = SessionLocal()
            try:
                user = db.query(User).options(
                    joinedload(User.visitor),
                    joinedload(User.officer)
                ).filter(User.id == payload["user_id"]).first()
                if not user:
                    return jsonify({"error": "User account no longer exists"}), 401
                if user.status == "SUSPENDED":
                    return jsonify({"error": "Account has been suspended"}), 403
                g.current_user = user
            finally:
                db.close()
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except Exception as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated

def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        @token_required
        def decorated(*args, **kwargs):
            if g.current_user.role not in allowed_roles:
                return jsonify({"error": f"Access forbidden. Requires one of roles: {allowed_roles}"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
