import jwt
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
from backend.app.config import Config

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

def decode_token(token: str):
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
    except Exception:
        return None

@socketio.on("connect")
def handle_connect():
    token = request.args.get("token")
    user_info = decode_token(token) if token else None
    print(f"WebSocket client connected: sid={request.sid}, user={user_info.get('email') if user_info else 'Anonymous'}")
    emit("connected_ack", {"status": "connected", "sid": request.sid})

@socketio.on("disconnect")
def handle_disconnect():
    print(f"WebSocket client disconnected: sid={request.sid}")

@socketio.on("join_room")
def handle_join_room(data):
    room_name = data.get("room")
    if room_name:
        join_room(room_name)
        print(f"Client {request.sid} joined room: {room_name}")
        emit("room_joined", {"room": room_name, "status": "active"}, room=request.sid)

@socketio.on("leave_room")
def handle_leave_room(data):
    room_name = data.get("room")
    if room_name:
        leave_room(room_name)
        emit("room_left", {"room": room_name}, room=request.sid)

# Broadcast helpers
def broadcast_visitor_location(location_data: dict):
    """Emits real-time location update to admin control room and visitor room."""
    socketio.emit("visitor_location_update", location_data, room="admin_control_room")
    v_id = location_data.get("visitor_id")
    if v_id:
        socketio.emit("visitor_location_update", location_data, room=f"visitor_{v_id}")

def broadcast_officer_location(officer_data: dict):
    """Emits officer location update to admin room."""
    socketio.emit("officer_location_update", officer_data, room="admin_control_room")

def broadcast_zone_transition(transition_data: dict):
    """Emits zone boundary transition event."""
    socketio.emit("zone_transition", transition_data, room="admin_control_room")
    v_id = transition_data.get("visitor_id")
    if v_id:
        socketio.emit("zone_transition", transition_data, room=f"visitor_{v_id}")

def broadcast_emergency_alert(alert_data: dict):
    """Broadcasts high-priority emergency alert to admin control room and all officers."""
    socketio.emit("emergency_alert", alert_data, room="admin_control_room")
    socketio.emit("emergency_alert", alert_data)  # Global broadcast for emergency priority

def broadcast_sos_video_chunk(chunk_data: dict):
    """Notifies admin and assigned officers as sequential video chunks are ingested."""
    socketio.emit("sos_video_chunk_received", chunk_data, room="admin_control_room")
    officer_id = chunk_data.get("officer_id")
    if officer_id:
        socketio.emit("sos_video_chunk_received", chunk_data, room=f"officer_channel_{officer_id}")

def broadcast_incident_state_changed(incident_data: dict):
    """Synchronizes acknowledged / resolved state across all consoles."""
    socketio.emit("incident_state_changed", incident_data, room="admin_control_room")
    socketio.emit("incident_state_changed", incident_data)
