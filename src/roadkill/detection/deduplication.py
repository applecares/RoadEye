"""Deduplication logic for detection events.

Prevents tagging the same carcass multiple times by checking spatial and
temporal proximity of detections.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate distance between two GPS points in metres.

    Uses the haversine formula for calculating great-circle distance
    between two points on a sphere.

    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees

    Returns:
        Distance in metres
    """
    earth_radius = 6371000  # Earth's radius in metres

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


@dataclass
class RecentDetection:
    """Record of a recent detection for deduplication."""

    latitude: float
    longitude: float
    species_code: str
    timestamp: float  # Unix timestamp


class DeduplicationFilter:
    """Filters duplicate detections based on spatial and temporal proximity.

    A detection is considered a duplicate if another detection of the same
    species occurred within the configured distance and time window.

    Attributes:
        distance_threshold_m: Maximum distance in metres for duplicate detection
        time_window_s: Maximum time in seconds for duplicate detection
        max_history: Maximum number of recent detections to track
    """

    def __init__(
        self,
        distance_threshold_m: float = 50.0,
        time_window_s: float = 300.0,
        max_history: int = 100,
    ):
        """Initialise deduplication filter.

        Args:
            distance_threshold_m: Distance threshold in metres (default: 50m)
            time_window_s: Time window in seconds (default: 300s / 5 minutes)
            max_history: Maximum detections to track (default: 100)
        """
        self.distance_threshold_m = distance_threshold_m
        self.time_window_s = time_window_s
        self.max_history = max_history
        self._recent: Deque[RecentDetection] = deque(maxlen=max_history)

    def is_duplicate(
        self,
        latitude: float,
        longitude: float,
        species_code: str,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Check if a detection is a duplicate of a recent event.

        Args:
            latitude: Detection latitude
            longitude: Detection longitude
            species_code: Species code of the detection
            timestamp: Unix timestamp (defaults to current time)

        Returns:
            True if this detection is a duplicate, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()

        for recent in self._recent:
            # Check time window
            time_diff = timestamp - recent.timestamp
            if time_diff > self.time_window_s:
                continue

            # Check species match
            if recent.species_code != species_code:
                continue

            # Check distance
            distance = haversine_distance(
                latitude, longitude, recent.latitude, recent.longitude
            )
            if distance < self.distance_threshold_m:
                return True

        return False

    def add_detection(
        self,
        latitude: float,
        longitude: float,
        species_code: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a detection for future deduplication checks.

        Args:
            latitude: Detection latitude
            longitude: Detection longitude
            species_code: Species code of the detection
            timestamp: Unix timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        self._recent.append(
            RecentDetection(
                latitude=latitude,
                longitude=longitude,
                species_code=species_code,
                timestamp=timestamp,
            )
        )

    def check_and_add(
        self,
        latitude: float,
        longitude: float,
        species_code: str,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Check for duplicate and add if unique.

        Convenience method that checks if a detection is a duplicate and,
        if not, adds it to the history.

        Args:
            latitude: Detection latitude
            longitude: Detection longitude
            species_code: Species code of the detection
            timestamp: Unix timestamp (defaults to current time)

        Returns:
            True if detection is unique (was added), False if duplicate
        """
        if self.is_duplicate(latitude, longitude, species_code, timestamp):
            return False

        self.add_detection(latitude, longitude, species_code, timestamp)
        return True

    def clear(self) -> None:
        """Clear all recent detections."""
        self._recent.clear()

    def prune_old(self, current_time: Optional[float] = None) -> int:
        """Remove detections older than the time window.

        Args:
            current_time: Current timestamp (defaults to now)

        Returns:
            Number of detections removed
        """
        if current_time is None:
            current_time = time.time()

        cutoff = current_time - self.time_window_s
        original_len = len(self._recent)

        # Filter to only keep recent detections
        self._recent = deque(
            (d for d in self._recent if d.timestamp >= cutoff),
            maxlen=self.max_history,
        )

        return original_len - len(self._recent)

    @property
    def count(self) -> int:
        """Number of detections currently tracked."""
        return len(self._recent)
