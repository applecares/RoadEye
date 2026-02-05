"""Hotspot prediction model."""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class RiskCategory(str, PyEnum):
    """Risk category enum."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Hotspot(Base, TimestampMixin):
    """Predicted roadkill hotspot."""

    __tablename__ = "hotspots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    road_segment_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )

    # Location
    center_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    center_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    length_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    road_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Risk assessment
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category: Mapped[RiskCategory] = mapped_column(
        Enum(RiskCategory), nullable=False, index=True
    )

    # Statistics
    incident_count_30d: Mapped[int] = mapped_column(Integer, default=0)
    incident_count_365d: Mapped[int] = mapped_column(Integer, default=0)
    primary_species: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Features used for prediction
    speed_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    road_curvature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vegetation_density: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_to_water: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Virtual fence recommendation
    fence_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    fence_priority: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fence_estimated_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Calculation metadata
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Hotspot(id={self.id}, segment={self.road_segment_id}, "
            f"risk={self.risk_category.value}, score={self.risk_score:.1f})>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "road_segment_id": self.road_segment_id,
            "center_latitude": self.center_latitude,
            "center_longitude": self.center_longitude,
            "length_km": self.length_km,
            "road_name": self.road_name,
            "risk_score": self.risk_score,
            "risk_category": self.risk_category.value,
            "incident_count_30d": self.incident_count_30d,
            "incident_count_365d": self.incident_count_365d,
            "primary_species": self.primary_species,
            "fence_recommended": self.fence_recommended,
            "fence_priority": self.fence_priority,
            "calculated_at": (
                self.calculated_at.isoformat() if self.calculated_at else None
            ),
        }
