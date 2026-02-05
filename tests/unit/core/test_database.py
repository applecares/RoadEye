"""Tests for database models."""

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from roadkill.database.models import (
    Base,
    DetectionEvent,
    Hotspot,
    RiskCategory,
    SourceType,
    VerificationStatus,
)


def get_test_session() -> Session:
    """Create an in-memory test database session."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


class TestDetectionEvent:
    """Tests for DetectionEvent model."""

    def test_create_detection_event(self):
        """Test creating a detection event."""
        session = get_test_session()

        event = DetectionEvent(
            event_id="TEST_001",
            timestamp=datetime.now(timezone.utc),
            latitude=-42.8821,
            longitude=147.3272,
            species_code="DEVIL",
            confidence=0.95,
        )

        session.add(event)
        session.commit()

        assert event.id is not None
        assert event.event_id == "TEST_001"
        assert event.species_code == "DEVIL"
        assert event.confidence == 0.95

        session.close()

    def test_detection_event_defaults(self):
        """Test detection event default values."""
        session = get_test_session()

        event = DetectionEvent(
            event_id="TEST_002",
            timestamp=datetime.now(timezone.utc),
            latitude=-42.0,
            longitude=147.0,
            species_code="WALBY",
            confidence=0.8,
        )

        session.add(event)
        session.commit()

        assert event.source_type == SourceType.DASHCAM
        assert event.verification_status == VerificationStatus.PENDING
        assert event.is_threatened is False
        assert event.nva_synced is False
        assert event.list_synced is False

        session.close()

    def test_detection_event_with_all_fields(self):
        """Test detection event with all fields populated."""
        session = get_test_session()

        event = DetectionEvent(
            event_id="TEST_003",
            timestamp=datetime.now(timezone.utc),
            latitude=-41.0500,
            longitude=144.6500,
            gps_accuracy=5.0,
            road_segment_id="ROAD_001",
            species_code="DEVIL",
            species_scientific="Sarcophilus harrisii",
            species_common="Tasmanian Devil",
            confidence=0.98,
            carcass_probability=0.95,
            is_threatened=True,
            source_type=SourceType.DASHCAM,
            device_id="FLEET001",
            vehicle_id="VEH001",
            speed_kmh=65.0,
            heading=180.0,
            image_path="/uploads/test.jpg",
            verification_status=VerificationStatus.AUTO_APPROVED,
        )

        session.add(event)
        session.commit()

        assert event.species_scientific == "Sarcophilus harrisii"
        assert event.is_threatened is True
        assert event.carcass_probability == 0.95
        assert event.verification_status == VerificationStatus.AUTO_APPROVED

        session.close()

    def test_detection_event_to_dict(self):
        """Test converting detection event to dictionary."""
        session = get_test_session()

        event = DetectionEvent(
            event_id="TEST_004",
            timestamp=datetime.now(timezone.utc),
            latitude=-42.0,
            longitude=147.0,
            species_code="PADEM",
            confidence=0.75,
        )

        session.add(event)
        session.commit()

        data = event.to_dict()

        assert data["event_id"] == "TEST_004"
        assert data["species_code"] == "PADEM"
        assert data["confidence"] == 0.75
        assert data["latitude"] == -42.0
        assert "timestamp" in data
        assert "created_at" in data

        session.close()

    def test_query_detection_events_by_species(self):
        """Test querying detection events by species."""
        session = get_test_session()

        # Create multiple events
        for i, species in enumerate(["DEVIL", "WALBY", "WALBY", "PADEM"]):
            event = DetectionEvent(
                event_id=f"TEST_{i:03d}",
                timestamp=datetime.now(timezone.utc),
                latitude=-42.0,
                longitude=147.0,
                species_code=species,
                confidence=0.8,
            )
            session.add(event)

        session.commit()

        # Query by species
        wallaby_events = (
            session.query(DetectionEvent)
            .filter(DetectionEvent.species_code == "WALBY")
            .all()
        )

        assert len(wallaby_events) == 2

        session.close()


class TestHotspot:
    """Tests for Hotspot model."""

    def test_create_hotspot(self):
        """Test creating a hotspot."""
        session = get_test_session()

        hotspot = Hotspot(
            road_segment_id="SEG_001",
            center_latitude=-41.0500,
            center_longitude=144.6500,
            risk_score=85.0,
            risk_category=RiskCategory.HIGH,
            calculated_at=datetime.now(timezone.utc),
        )

        session.add(hotspot)
        session.commit()

        assert hotspot.id is not None
        assert hotspot.road_segment_id == "SEG_001"
        assert hotspot.risk_score == 85.0
        assert hotspot.risk_category == RiskCategory.HIGH

        session.close()

    def test_hotspot_with_fence_recommendation(self):
        """Test hotspot with virtual fence recommendation."""
        session = get_test_session()

        hotspot = Hotspot(
            road_segment_id="SEG_002",
            center_latitude=-41.8000,
            center_longitude=145.5000,
            risk_score=95.0,
            risk_category=RiskCategory.CRITICAL,
            calculated_at=datetime.now(timezone.utc),
            fence_recommended=True,
            fence_priority=1,
            fence_estimated_cost=15000.0,
            primary_species="DEVIL",
        )

        session.add(hotspot)
        session.commit()

        assert hotspot.fence_recommended is True
        assert hotspot.fence_priority == 1
        assert hotspot.fence_estimated_cost == 15000.0
        assert hotspot.primary_species == "DEVIL"

        session.close()

    def test_hotspot_to_dict(self):
        """Test converting hotspot to dictionary."""
        session = get_test_session()

        hotspot = Hotspot(
            road_segment_id="SEG_003",
            center_latitude=-42.0,
            center_longitude=147.0,
            road_name="Huon Highway",
            risk_score=60.0,
            risk_category=RiskCategory.MEDIUM,
            incident_count_30d=5,
            incident_count_365d=45,
            calculated_at=datetime.now(timezone.utc),
        )

        session.add(hotspot)
        session.commit()

        data = hotspot.to_dict()

        assert data["road_segment_id"] == "SEG_003"
        assert data["road_name"] == "Huon Highway"
        assert data["risk_score"] == 60.0
        assert data["risk_category"] == "MEDIUM"
        assert data["incident_count_30d"] == 5

        session.close()

    def test_query_hotspots_by_risk_category(self):
        """Test querying hotspots by risk category."""
        session = get_test_session()

        # Create hotspots with different risk levels
        risk_levels = [
            (RiskCategory.LOW, 20.0),
            (RiskCategory.MEDIUM, 50.0),
            (RiskCategory.HIGH, 75.0),
            (RiskCategory.CRITICAL, 95.0),
        ]

        for i, (category, score) in enumerate(risk_levels):
            hotspot = Hotspot(
                road_segment_id=f"SEG_{i:03d}",
                center_latitude=-42.0,
                center_longitude=147.0,
                risk_score=score,
                risk_category=category,
                calculated_at=datetime.now(timezone.utc),
            )
            session.add(hotspot)

        session.commit()

        # Query critical hotspots
        critical = (
            session.query(Hotspot)
            .filter(Hotspot.risk_category == RiskCategory.CRITICAL)
            .all()
        )

        assert len(critical) == 1
        assert critical[0].risk_score == 95.0

        # Query high risk and above
        high_risk = (
            session.query(Hotspot)
            .filter(
                Hotspot.risk_category.in_([RiskCategory.HIGH, RiskCategory.CRITICAL])
            )
            .all()
        )

        assert len(high_risk) == 2

        session.close()


class TestVerificationStatus:
    """Tests for verification status enum."""

    def test_verification_status_values(self):
        """Test verification status enum values."""
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.AUTO_APPROVED.value == "auto_approved"
        assert VerificationStatus.MANUAL_REVIEW.value == "manual_review"
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.REJECTED.value == "rejected"
        assert VerificationStatus.ARCHIVED.value == "archived"


class TestSourceType:
    """Tests for source type enum."""

    def test_source_type_values(self):
        """Test source type enum values."""
        assert SourceType.DASHCAM.value == "dashcam"
        assert SourceType.DRONE.value == "drone"
        assert SourceType.SENSOR.value == "sensor"
        assert SourceType.CITIZEN.value == "citizen"
        assert SourceType.MANUAL.value == "manual"


class TestRiskCategory:
    """Tests for risk category enum."""

    def test_risk_category_values(self):
        """Test risk category enum values."""
        assert RiskCategory.LOW.value == "LOW"
        assert RiskCategory.MEDIUM.value == "MEDIUM"
        assert RiskCategory.HIGH.value == "HIGH"
        assert RiskCategory.CRITICAL.value == "CRITICAL"
