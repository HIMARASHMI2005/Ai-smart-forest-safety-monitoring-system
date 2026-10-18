import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    String, Integer, Float, DateTime, Boolean, ForeignKey, LargeBinary, Text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database.connection import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'ADMIN', 'OFFICER', 'VISITOR'
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")  # 'ACTIVE', 'PENDING', 'SUSPENDED'
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    visitor: Mapped[Optional["Visitor"]] = relationship("Visitor", back_populates="user", uselist=False, cascade="all, delete-orphan")
    officer: Mapped[Optional["Officer"]] = relationship("Officer", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Visitor(Base):
    __tablename__ = "visitors"

    visitor_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    phone: Mapped[str] = mapped_column(String(30), nullable=False)
    emergency_contact_name: Mapped[str] = mapped_column(String(120), nullable=False)
    emergency_contact_phone: Mapped[str] = mapped_column(String(30), nullable=False)
    current_zone: Mapped[str] = mapped_column(String(30), default="OUT_OF_BOUNDS")
    current_risk_level: Mapped[str] = mapped_column(String(20), default="LOW")
    last_latitude: Mapped[float] = mapped_column(Float, default=0.0)
    last_longitude: Mapped[float] = mapped_column(Float, default=0.0)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(20), default="REGISTERED")

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="visitor")
    biometric_templates: Mapped[List["BiometricTemplate"]] = relationship("BiometricTemplate", back_populates="visitor", cascade="all, delete-orphan")
    locations: Mapped[List["Location"]] = relationship("Location", back_populates="visitor", cascade="all, delete-orphan")
    zone_events: Mapped[List["ZoneEvent"]] = relationship("ZoneEvent", back_populates="visitor", cascade="all, delete-orphan")
    risk_events: Mapped[List["RiskEvent"]] = relationship("RiskEvent", back_populates="visitor", cascade="all, delete-orphan")
    sos_events: Mapped[List["SosEvent"]] = relationship("SosEvent", back_populates="visitor", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "visitor_id": self.visitor_id,
            "user_id": self.user_id,
            "full_name": self.full_name,
            "phone": self.phone,
            "emergency_contact_name": self.emergency_contact_name,
            "emergency_contact_phone": self.emergency_contact_phone,
            "current_zone": self.current_zone,
            "current_risk_level": self.current_risk_level,
            "last_latitude": self.last_latitude,
            "last_longitude": self.last_longitude,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "status": self.status
        }


class Officer(Base):
    __tablename__ = "officers"

    officer_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    badge_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    phone: Mapped[str] = mapped_column(String(30), nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=False)
    approval_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    availability_status: Mapped[str] = mapped_column(String(20), default="AVAILABLE")
    current_latitude: Mapped[float] = mapped_column(Float, default=0.0)
    current_longitude: Mapped[float] = mapped_column(Float, default=0.0)
    last_ping: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="officer")

    def to_dict(self):
        return {
            "officer_id": self.officer_id,
            "user_id": self.user_id,
            "badge_number": self.badge_number,
            "full_name": self.full_name,
            "phone": self.phone,
            "department": self.department,
            "approval_status": self.approval_status,
            "availability_status": self.availability_status,
            "current_latitude": self.current_latitude,
            "current_longitude": self.current_longitude,
            "last_ping": self.last_ping.isoformat() if self.last_ping else None
        }


class BiometricTemplate(Base):
    __tablename__ = "biometric_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    visitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("visitors.visitor_id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # 128 x 32-bit float array bytes
    model_version: Mapped[str] = mapped_column(String(50), default="ForestFaceNet_v1")
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)
    face_pose_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship
    visitor: Mapped["Visitor"] = relationship("Visitor", back_populates="biometric_templates")

    def to_dict(self):
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "model_version": self.model_version,
            "quality_score": self.quality_score,
            "face_pose_index": self.face_pose_index,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("visitors.visitor_id", ondelete="CASCADE"), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    altitude: Mapped[float] = mapped_column(Float, default=0.0)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    speed: Mapped[float] = mapped_column(Float, default=0.0)
    bearing: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationship
    visitor: Mapped["Visitor"] = relationship("Visitor", back_populates="locations")

    def to_dict(self):
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "accuracy": self.accuracy,
            "speed": self.speed,
            "bearing": self.bearing,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    code: Mapped[str] = mapped_column(String(30), nullable=False)  # ZONE_A, ZONE_B, ZONE_C, ZONE_D
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    polygon_geojson: Mapped[str] = mapped_column(Text, nullable=False)
    min_lat: Mapped[float] = mapped_column(Float, nullable=False)
    max_lat: Mapped[float] = mapped_column(Float, nullable=False)
    min_lng: Mapped[float] = mapped_column(Float, nullable=False)
    max_lng: Mapped[float] = mapped_column(Float, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    color_hex: Mapped[str] = mapped_column(String(10), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "severity": self.severity,
            "polygon_geojson": self.polygon_geojson,
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lng": self.min_lng,
            "max_lng": self.max_lng,
            "active": self.active,
            "color_hex": self.color_hex
        }


class ZoneEvent(Base):
    __tablename__ = "zone_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("visitors.visitor_id", ondelete="CASCADE"), nullable=False, index=True)
    zone_id: Mapped[str] = mapped_column(String(50), ForeignKey("zones.id", ondelete="CASCADE"), nullable=False)
    previous_zone: Mapped[Optional[str]] = mapped_column(String(50))
    current_zone: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)  # ENTRY, EXIT, TRANSITION
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationship
    visitor: Mapped["Visitor"] = relationship("Visitor", back_populates="zone_events")

    def to_dict(self):
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "zone_id": self.zone_id,
            "previous_zone": self.previous_zone,
            "current_zone": self.current_zone,
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("visitors.visitor_id", ondelete="CASCADE"), nullable=False, index=True)
    score: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 - 100
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    reasons_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array of active risk factors
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationship
    visitor: Mapped["Visitor"] = relationship("Visitor", back_populates="risk_events")

    def to_dict(self):
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "score": self.score,
            "severity": self.severity,
            "reasons_json": self.reasons_json,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class SosEvent(Base):
    __tablename__ = "sos_events"

    sos_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    visitor_id: Mapped[str] = mapped_column(String(36), ForeignKey("visitors.visitor_id", ondelete="CASCADE"), nullable=False, index=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    initial_lat: Mapped[float] = mapped_column(Float, nullable=False)
    initial_lng: Mapped[float] = mapped_column(Float, nullable=False)
    initial_zone: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="TRIGGERED", index=True)  # TRIGGERED, ACKNOWLEDGED, DISPATCHED, RESOLVED, CANCELLED
    acknowledged_by_officer_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("officers.officer_id", ondelete="SET NULL"), nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    visitor: Mapped["Visitor"] = relationship("Visitor", back_populates="sos_events")
    chunks: Mapped[List["SosVideoChunk"]] = relationship("SosVideoChunk", back_populates="sos_event", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "sos_id": self.sos_id,
            "visitor_id": self.visitor_id,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "initial_lat": self.initial_lat,
            "initial_lng": self.initial_lng,
            "initial_zone": self.initial_zone,
            "status": self.status,
            "acknowledged_by_officer_id": self.acknowledged_by_officer_id,
            "resolution_notes": self.resolution_notes
        }


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)  # ZONE_VIOLATION, CRITICAL_ZONE, UNKNOWN_PERSON, SOS_TRIGGERED, PROLONGED_STATIONARY
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # WARNING, DANGER, CRITICAL
    visitor_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("visitors.visitor_id", ondelete="CASCADE"), nullable=True)
    officer_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("officers.officer_id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="UNREAD", index=True)  # UNREAD, ACKNOWLEDGED, RESOLVED
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "visitor_id": self.visitor_id,
            "officer_id": self.officer_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class SosVideoChunk(Base):
    __tablename__ = "sos_video_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sos_id: Mapped[str] = mapped_column(String(36), ForeignKey("sos_events.sos_id", ondelete="CASCADE"), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship
    sos_event: Mapped["SosEvent"] = relationship("SosEvent", back_populates="chunks")

    def to_dict(self):
        return {
            "id": self.id,
            "sos_id": self.sos_id,
            "sequence_number": self.sequence_number,
            "storage_path": self.storage_path,
            "file_hash_sha256": self.file_hash_sha256,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None
        }
