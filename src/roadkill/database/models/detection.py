"""Detection event model."""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class VerificationStatus(str, PyEnum):
    """Verification status enum."""

    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    MANUAL_REVIEW = "manual_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class SourceType(str, PyEnum):
    """Detection source type."""

    DASHCAM = "dashcam"
    DRONE = "drone"
    SENSOR = "sensor"
    CITIZEN = "citizen"
    MANUAL = "manual"


class DetectionEvent(Base, TimestampMixin):
    """Roadkill detection event."""

    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Location
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    gps_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    road_segment_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Species
    species_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    species_scientific: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    species_common: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    carcass_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_threatened: Mapped[bool] = mapped_column(Boolean, default=False)

    # Source
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType), default=SourceType.DASHCAM
    )
    device_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    vehicle_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    speed_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    heading: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Media
    image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    video_clip_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Verification
    verification_status: Mapped[VerificationStatus] = mapped_column(
        Enum(VerificationStatus), default=VerificationStatus.PENDING, index=True
    )
    verified_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    verification_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Sync status
    nva_synced: Mapped[bool] = mapped_column(Boolean, default=False)
    nva_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    list_synced: Mapped[bool] = mapped_column(Boolean, default=False)
    list_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Additional data
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DetectionEvent(id={self.id}, event_id={self.event_id}, "
            f"species={self.species_code}, confidence={self.confidence:.2f})>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "gps_accuracy": self.gps_accuracy,
            "species_code": self.species_code,
            "species_scientific": self.species_scientific,
            "species_common": self.species_common,
            "confidence": self.confidence,
            "carcass_probability": self.carcass_probability,
            "is_threatened": self.is_threatened,
            "source_type": self.source_type.value if self.source_type else None,
            "device_id": self.device_id,
            "vehicle_id": self.vehicle_id,
            "verification_status": (
                self.verification_status.value if self.verification_status else None
            ),
            "image_path": self.image_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
