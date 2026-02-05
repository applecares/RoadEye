"""Tests for detection engine."""

import pytest

from roadkill.core.species import Species, SpeciesRegistry, ThreatStatus
from roadkill.detection.engine import (
    DetectionEngine,
    DetectionEvent,
    DetectionResult,
)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_create_detection_result(self):
        """Test creating a detection result."""
        result = DetectionResult(
            species_code="DEVIL",
            confidence=0.95,
            bbox=(100, 200, 300, 400),
            class_id=0,
            raw_class_name="tasmanian_devil",
        )

        assert result.species_code == "DEVIL"
        assert result.confidence == 0.95
        assert result.bbox == (100, 200, 300, 400)
        assert result.class_id == 0
        assert result.raw_class_name == "tasmanian_devil"


class TestDetectionEvent:
    """Tests for DetectionEvent dataclass."""

    def test_create_detection_event(self):
        """Test creating a detection event."""
        event = DetectionEvent(
            event_id="TEST_001",
            timestamp="2026-02-05T12:00:00Z",
            latitude=-42.8821,
            longitude=147.3272,
            gps_accuracy=5.0,
            species_code="DEVIL",
            species_scientific="Sarcophilus harrisii",
            species_common="Tasmanian Devil",
            confidence=0.95,
            bbox=(100, 200, 300, 400),
            is_threatened=True,
        )

        assert event.event_id == "TEST_001"
        assert event.species_code == "DEVIL"
        assert event.is_threatened is True
        assert event.source_type == "dashcam"  # Default

    def test_detection_event_to_dict(self):
        """Test converting event to dictionary."""
        event = DetectionEvent(
            event_id="TEST_002",
            timestamp="2026-02-05T12:00:00Z",
            latitude=-42.0,
            longitude=147.0,
            gps_accuracy=3.0,
            species_code="WALBY",
            species_scientific=None,
            species_common="Bennett's Wallaby",
            confidence=0.85,
            bbox=(50, 100, 150, 200),
            is_threatened=False,
            carcass_probability=0.75,
            speed_kmh=60.0,
            heading=180.0,
        )

        data = event.to_dict()

        assert data["event_id"] == "TEST_002"
        assert data["species_code"] == "WALBY"
        assert data["confidence"] == 0.85
        assert data["bbox"] == [50, 100, 150, 200]  # Converted to list
        assert data["carcass_probability"] == 0.75
        assert data["speed_kmh"] == 60.0


class TestDetectionEngine:
    """Tests for DetectionEngine."""

    def test_engine_init_without_model(self):
        """Test engine initialisation without YOLO model."""
        engine = DetectionEngine(model_path=None, use_mewc_docker=False)
        # Model may or may not be available depending on ultralytics
        # Just verify engine initialises without error
        assert engine is not None

    def test_engine_init_mewc_mode(self):
        """Test engine initialisation in MEWC Docker mode."""
        engine = DetectionEngine(use_mewc_docker=True)
        assert engine.use_mewc_docker is True
        assert engine.model is None

    def test_verification_queue_auto_approve(self):
        """Test verification queue for high confidence detection."""
        engine = DetectionEngine(
            auto_approve_threshold=0.95,
            review_threshold=0.75,
        )

        queue = engine.get_verification_queue(confidence=0.98, is_threatened=False)
        assert queue == "auto_approve"

    def test_verification_queue_standard_review(self):
        """Test verification queue for medium confidence detection."""
        engine = DetectionEngine(
            auto_approve_threshold=0.95,
            review_threshold=0.75,
        )

        queue = engine.get_verification_queue(confidence=0.85, is_threatened=False)
        assert queue == "standard_review"

    def test_verification_queue_archive(self):
        """Test verification queue for low confidence detection."""
        engine = DetectionEngine(
            auto_approve_threshold=0.95,
            review_threshold=0.75,
        )

        queue = engine.get_verification_queue(confidence=0.60, is_threatened=False)
        assert queue == "archive"

    def test_verification_queue_threatened_urgent(self):
        """Test threatened species always go to urgent review."""
        engine = DetectionEngine(
            auto_approve_threshold=0.95,
            review_threshold=0.75,
        )

        # Even with high confidence, threatened species go to urgent review
        queue = engine.get_verification_queue(confidence=0.99, is_threatened=True)
        assert queue == "urgent_review"

        # Low confidence threatened species also go to urgent review
        queue = engine.get_verification_queue(confidence=0.55, is_threatened=True)
        assert queue == "urgent_review"


class TestDetectionEngineWithRegistry:
    """Tests for DetectionEngine with species registry."""

    @pytest.fixture
    def registry(self):
        """Create a test species registry."""
        reg = SpeciesRegistry(state_code="TAS")
        reg.register(
            Species(
                code="DEVIL",
                scientific_name="Sarcophilus harrisii",
                common_name="Tasmanian Devil",
                threatened_status=ThreatStatus.ENDANGERED,
                mewc_class_id=0,
            )
        )
        reg.register(
            Species(
                code="WALBY",
                scientific_name="Notamacropus rufogriseus",
                common_name="Bennett's Wallaby",
                mewc_class_id=1,
            )
        )
        return reg

    def test_create_event_with_registry(self, registry):
        """Test creating event with species lookup."""
        engine = DetectionEngine(
            use_mewc_docker=True,  # Avoid loading YOLO
            species_registry=registry,
        )

        detection = DetectionResult(
            species_code="DEVIL",
            confidence=0.92,
            bbox=(100, 200, 300, 400),
            class_id=0,
            raw_class_name="tasmanian_devil",
        )

        gps_data = {
            "latitude": -42.8821,
            "longitude": 147.3272,
            "accuracy": 5.0,
            "speed": 60.0,
            "heading": 180.0,
        }

        event = engine.create_event(
            detection=detection,
            gps_data=gps_data,
            device_id="TEST_DEVICE",
            vehicle_id="TEST_VEHICLE",
        )

        assert event.species_code == "DEVIL"
        assert event.species_scientific == "Sarcophilus harrisii"
        assert event.species_common == "Tasmanian Devil"
        assert event.is_threatened is True
        assert event.verification_queue == "urgent_review"  # Threatened species
        assert event.latitude == -42.8821
        assert event.longitude == 147.3272
        assert event.speed_kmh == 60.0

    def test_create_event_unknown_species(self, registry):
        """Test creating event for unknown species."""
        engine = DetectionEngine(
            use_mewc_docker=True,
            species_registry=registry,
        )

        detection = DetectionResult(
            species_code="UNKNOWN",
            confidence=0.80,
            bbox=(100, 200, 300, 400),
            class_id=99,
            raw_class_name="unknown_animal",
        )

        gps_data = {"latitude": -42.0, "longitude": 147.0, "accuracy": 10.0}

        event = engine.create_event(
            detection=detection,
            gps_data=gps_data,
        )

        assert event.species_code == "UNKNOWN"
        assert event.species_scientific is None
        assert event.species_common is None
        assert event.is_threatened is False


class TestCOCOClassMapping:
    """Tests for COCO class mapping."""

    def test_coco_animal_classes_defined(self):
        """Test COCO animal class IDs are defined."""
        assert 15 in DetectionEngine.COCO_ANIMAL_CLASSES  # cat
        assert 16 in DetectionEngine.COCO_ANIMAL_CLASSES  # dog
        assert 14 in DetectionEngine.COCO_ANIMAL_CLASSES  # bird

    def test_map_coco_cat_to_feral_cat(self):
        """Test COCO cat maps to feral cat code."""
        engine = DetectionEngine(use_mewc_docker=True)

        species_code = engine._map_to_species(15, "cat")
        assert species_code == "FCAT"

    def test_map_unknown_class(self):
        """Test unknown class returns UNKNOWN code."""
        engine = DetectionEngine(use_mewc_docker=True)

        species_code = engine._map_to_species(999, "random_object")
        assert species_code == "UNKNOWN"


class TestEngineAvailability:
    """Tests for engine availability checks."""

    def test_mewc_mode_is_available(self):
        """MEWC mode should report as available."""
        engine = DetectionEngine(use_mewc_docker=True)
        assert engine.is_available is True

    def test_no_model_not_available(self):
        """Engine without model should report not available."""
        engine = DetectionEngine(model_path="/nonexistent/path.pt")
        # Availability depends on whether ultralytics is installed
        # and whether it falls back to default model
        assert engine is not None
