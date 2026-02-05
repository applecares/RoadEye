"""Detection engine for wildlife/roadkill detection.

Supports YOLO-based detection with optional MEWC Docker pipeline for
production deployments.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from roadkill.core.species import SpeciesRegistry, get_species_registry

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from a single detection inference."""

    species_code: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    raw_class_name: str


@dataclass
class DetectionEvent:
    """A complete detection event with all metadata."""

    event_id: str
    timestamp: str
    latitude: float
    longitude: float
    gps_accuracy: float
    species_code: str
    species_scientific: Optional[str]
    species_common: Optional[str]
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_threatened: bool
    carcass_probability: Optional[float] = None
    source_type: str = "dashcam"
    device_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    speed_kmh: Optional[float] = None
    heading: Optional[float] = None
    image_path: Optional[str] = None
    verification_queue: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "gps_accuracy": self.gps_accuracy,
            "species_code": self.species_code,
            "species_scientific": self.species_scientific,
            "species_common": self.species_common,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "is_threatened": self.is_threatened,
            "carcass_probability": self.carcass_probability,
            "source_type": self.source_type,
            "device_id": self.device_id,
            "vehicle_id": self.vehicle_id,
            "speed_kmh": self.speed_kmh,
            "heading": self.heading,
            "image_path": self.image_path,
            "verification_queue": self.verification_queue,
            "metadata": self.metadata,
        }


class DetectionEngine:
    """Wildlife detection engine using YOLO or MEWC Docker.

    The engine supports two modes:
    - YOLO mode: Direct inference using ultralytics YOLO model
    - MEWC Docker mode: Production pipeline using MegaDetector containers

    Confidence thresholds determine verification routing:
    - >= auto_approve_threshold: Direct sync to NVA/LIST
    - >= review_threshold: Human verification queue
    - < review_threshold: Archive only
    """

    # COCO animal class IDs for testing with pre-trained models
    COCO_ANIMAL_CLASSES = {
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mewc_docker: bool = False,
        species_registry: Optional[SpeciesRegistry] = None,
        auto_approve_threshold: float = 0.95,
        review_threshold: float = 0.75,
        min_confidence: float = 0.50,
    ):
        """Initialise detection engine.

        Args:
            model_path: Path to YOLO model file (.pt)
            use_mewc_docker: Use MEWC Docker containers instead of local YOLO
            species_registry: Species registry for classification
            auto_approve_threshold: Confidence for auto-approval
            review_threshold: Confidence for review queue
            min_confidence: Minimum confidence to create event
        """
        self.use_mewc_docker = use_mewc_docker
        self.model = None
        self.model_path = model_path

        # Thresholds
        self.auto_approve_threshold = auto_approve_threshold
        self.review_threshold = review_threshold
        self.min_confidence = min_confidence

        # Species registry
        self.species_registry = species_registry or get_species_registry()

        # Initialise model
        if not use_mewc_docker:
            self._init_yolo(model_path)
        else:
            logger.info("Detection engine initialised (MEWC Docker mode)")
            logger.info("Ensure MEWC containers are running:")
            logger.info("  - mewc-detect (MegaDetector)")
            logger.info("  - mewc-snip")
            logger.info("  - mewc-predict")

    def _init_yolo(self, model_path: Optional[str]) -> None:
        """Initialise YOLO model."""
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info("Loaded YOLO model: %s", model_path)
            else:
                # Fall back to pre-trained model for testing
                logger.info("Loading YOLOv8n for testing (no custom model provided)")
                self.model = YOLO("yolov8n.pt")

            logger.info("Detection engine initialised (YOLO mode)")

        except ImportError:
            logger.warning(
                "ultralytics not installed. Detection engine will be inactive."
            )
            self.model = None
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            self.model = None

    def detect(
        self,
        frame: Any,
        gps_data: Dict[str, float],
        device_id: Optional[str] = None,
        vehicle_id: Optional[str] = None,
    ) -> List[DetectionResult]:
        """Run detection on a single frame.

        Args:
            frame: BGR image array (numpy ndarray)
            gps_data: Dict with latitude, longitude, accuracy, speed, heading
            device_id: Device identifier
            vehicle_id: Vehicle identifier

        Returns:
            List of DetectionResult objects
        """
        if self.use_mewc_docker:
            return self._detect_mewc(frame)
        else:
            return self._detect_yolo(frame)

    def _detect_yolo(self, frame: Any) -> List[DetectionResult]:
        """Run YOLO inference on frame."""
        if self.model is None:
            return []

        results = self.model(frame, conf=self.min_confidence, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Get raw class name from model
                raw_class = self.model.names.get(cls_id, f"class_{cls_id}")

                # Map to species code if possible
                species_code = self._map_to_species(cls_id, raw_class)

                detections.append(
                    DetectionResult(
                        species_code=species_code,
                        confidence=round(conf, 4),
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        raw_class_name=raw_class,
                    )
                )

        return detections

    def _detect_mewc(self, frame: Any) -> List[DetectionResult]:
        """Run MEWC Docker pipeline detection.

        In production, this calls the MEWC Docker containers:
        - zaandahl/mewc-detect (MegaDetector)
        - zaandahl/mewc-snip
        - zaandahl/mewc-predict
        """
        logger.warning("MEWC Docker detection not yet implemented")
        return []

    def _map_to_species(self, class_id: int, raw_class: str) -> str:
        """Map model class to species code.

        For COCO pre-trained models, maps animal classes to generic codes.
        For custom MEWC models, uses the MEWC class mapping.
        """
        # Check if it's a MEWC class (custom trained)
        species = self.species_registry.get_by_mewc_class(class_id)
        if species:
            return species.code

        # Check COCO animal classes
        if class_id in self.COCO_ANIMAL_CLASSES:
            coco_name = self.COCO_ANIMAL_CLASSES[class_id]
            # Map common COCO animals to closest Tasmania species for testing
            coco_to_species = {
                "cat": "FCAT",  # Feral cat
                "dog": "DEVIL",  # Placeholder
                "bird": "CURRA",  # Currawong
            }
            return coco_to_species.get(coco_name, "UNKNOWN")

        return "UNKNOWN"

    def create_event(
        self,
        detection: DetectionResult,
        gps_data: Dict[str, float],
        device_id: Optional[str] = None,
        vehicle_id: Optional[str] = None,
        frame: Optional[Any] = None,
        save_image: bool = False,
        output_dir: Optional[Path] = None,
    ) -> DetectionEvent:
        """Create a detection event from a detection result.

        Args:
            detection: Detection result from inference
            gps_data: GPS data dict
            device_id: Device identifier
            vehicle_id: Vehicle identifier
            frame: Original frame for image saving
            save_image: Whether to save detection image
            output_dir: Directory for saved images

        Returns:
            DetectionEvent with all metadata
        """
        # Generate event ID
        timestamp = datetime.now(timezone.utc)
        event_id = (
            f"{vehicle_id or 'DEV'}_{timestamp.strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )

        # Get species info
        species = self.species_registry.get(detection.species_code)
        species_scientific = species.scientific_name if species else None
        species_common = species.common_name if species else None
        is_threatened = species.is_threatened if species else False

        # Calculate carcass probability if frame available
        carcass_prob = None
        if frame is not None:
            carcass_prob = self._estimate_carcass_probability(frame, detection.bbox)

        # Determine verification queue
        verification_queue = self.get_verification_queue(
            detection.confidence, is_threatened
        )

        # Save image if requested
        image_path = None
        if save_image and frame is not None and output_dir:
            image_path = self._save_detection_image(
                frame, detection.bbox, event_id, output_dir
            )

        return DetectionEvent(
            event_id=event_id,
            timestamp=timestamp.isoformat(),
            latitude=gps_data.get("latitude", 0.0),
            longitude=gps_data.get("longitude", 0.0),
            gps_accuracy=gps_data.get("accuracy", 10.0),
            species_code=detection.species_code,
            species_scientific=species_scientific,
            species_common=species_common,
            confidence=detection.confidence,
            bbox=detection.bbox,
            is_threatened=is_threatened,
            carcass_probability=carcass_prob,
            source_type="dashcam",
            device_id=device_id,
            vehicle_id=vehicle_id,
            speed_kmh=gps_data.get("speed"),
            heading=gps_data.get("heading"),
            image_path=image_path,
            verification_queue=verification_queue,
        )

    def get_verification_queue(
        self, confidence: float, is_threatened: bool
    ) -> str:
        """Determine verification queue based on confidence and species.

        Threatened species always go to urgent review regardless of confidence.

        Args:
            confidence: Detection confidence score
            is_threatened: Whether the species is threatened

        Returns:
            Queue name: 'urgent_review', 'auto_approve', 'standard_review', 'archive'
        """
        if is_threatened:
            return "urgent_review"
        elif confidence >= self.auto_approve_threshold:
            return "auto_approve"
        elif confidence >= self.review_threshold:
            return "standard_review"
        else:
            return "archive"

    def _estimate_carcass_probability(
        self, frame: Any, bbox: Tuple[int, int, int, int]
    ) -> float:
        """Estimate probability that detection is a carcass.

        Uses simple heuristics based on:
        - Road surface context (grey tones)
        - Object position (lower third of frame = on road)

        In production, a dedicated carcass classifier would be used.

        Args:
            frame: BGR image array
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Probability between 0 and 1
        """
        try:
            import numpy as np

            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.5

            # Heuristic 1: Road surface tends to be grey
            # Check colour variance (road = low variance)
            r_g_diff = np.abs(roi[:, :, 0].astype(float) - roi[:, :, 1].astype(float))
            grey_ratio = float(np.mean(r_g_diff < 20))

            # Heuristic 2: Object in lower third of frame = on road
            center_y = (y1 + y2) / 2
            on_road_factor = 0.2 if center_y > h * 0.6 else 0.0

            probability = min(0.95, 0.5 + grey_ratio * 0.3 + on_road_factor)
            return round(probability, 3)

        except Exception as e:
            logger.debug("Error estimating carcass probability: %s", e)
            return 0.5

    def _save_detection_image(
        self,
        frame: Any,
        bbox: Tuple[int, int, int, int],
        event_id: str,
        output_dir: Path,
    ) -> Optional[str]:
        """Save cropped detection image.

        Args:
            frame: BGR image array
            bbox: Bounding box (x1, y1, x2, y2)
            event_id: Event identifier for filename
            output_dir: Output directory

        Returns:
            Path to saved image or None if failed
        """
        try:
            import cv2

            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Add padding around detection
            pad = 50
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # Crop and save
            crop = frame[y1:y2, x1:x2]
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path = output_dir / f"{event_id}.jpg"
            cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

            return str(image_path)

        except Exception as e:
            logger.error("Failed to save detection image: %s", e)
            return None

    @property
    def is_available(self) -> bool:
        """Check if detection engine is available."""
        return self.model is not None or self.use_mewc_docker
