"""Detection module for wildlife/roadkill detection."""

from .deduplication import DeduplicationFilter, haversine_distance
from .engine import DetectionEngine, DetectionResult

__all__ = [
    "DeduplicationFilter",
    "DetectionEngine",
    "DetectionResult",
    "haversine_distance",
]
