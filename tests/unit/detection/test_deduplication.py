"""Tests for deduplication logic."""

import time

import pytest

from roadkill.detection.deduplication import (
    DeduplicationFilter,
    haversine_distance,
)


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_zero_distance(self):
        """Same point should return zero distance."""
        distance = haversine_distance(-42.8821, 147.3272, -42.8821, 147.3272)
        assert distance == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        """Test with known distance between two points."""
        # Hobart to Launceston is approximately 200km
        hobart = (-42.8821, 147.3272)
        launceston = (-41.4332, 147.1441)
        distance = haversine_distance(
            hobart[0], hobart[1], launceston[0], launceston[1]
        )
        # Should be roughly 160-170km (straight line)
        assert 150000 < distance < 180000

    def test_small_distance_metres(self):
        """Test small distances in metres."""
        # Two points ~50m apart
        lat1, lon1 = -42.8821, 147.3272
        # Approximately 0.00045 degrees = ~50m at this latitude
        lat2, lon2 = -42.8825, 147.3272
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        assert 40 < distance < 60  # Should be ~44m

    def test_commutative(self):
        """Distance should be same regardless of point order."""
        p1 = (-42.8821, 147.3272)
        p2 = (-41.4332, 147.1441)
        d1 = haversine_distance(p1[0], p1[1], p2[0], p2[1])
        d2 = haversine_distance(p2[0], p2[1], p1[0], p1[1])
        assert d1 == pytest.approx(d2, abs=0.001)


class TestDeduplicationFilter:
    """Tests for DeduplicationFilter."""

    def test_empty_filter_not_duplicate(self):
        """New detection on empty filter is not a duplicate."""
        filter_ = DeduplicationFilter()
        is_dup = filter_.is_duplicate(-42.0, 147.0, "DEVIL")
        assert is_dup is False

    def test_same_detection_is_duplicate(self):
        """Same species at same location is a duplicate."""
        filter_ = DeduplicationFilter()

        # Add first detection
        filter_.add_detection(-42.0, 147.0, "DEVIL")

        # Check same location and species
        is_dup = filter_.is_duplicate(-42.0, 147.0, "DEVIL")
        assert is_dup is True

    def test_different_species_not_duplicate(self):
        """Different species at same location is not a duplicate."""
        filter_ = DeduplicationFilter()

        filter_.add_detection(-42.0, 147.0, "DEVIL")
        is_dup = filter_.is_duplicate(-42.0, 147.0, "WALBY")
        assert is_dup is False

    def test_distant_location_not_duplicate(self):
        """Same species at distant location is not a duplicate."""
        filter_ = DeduplicationFilter(distance_threshold_m=50.0)

        filter_.add_detection(-42.0, 147.0, "DEVIL")
        # ~11km away
        is_dup = filter_.is_duplicate(-42.1, 147.0, "DEVIL")
        assert is_dup is False

    def test_within_threshold_is_duplicate(self):
        """Detection within distance threshold is a duplicate."""
        filter_ = DeduplicationFilter(distance_threshold_m=100.0)

        filter_.add_detection(-42.0, 147.0, "DEVIL")
        # ~44m away (within 100m threshold)
        is_dup = filter_.is_duplicate(-42.0004, 147.0, "DEVIL")
        assert is_dup is True

    def test_old_detection_not_duplicate(self):
        """Detection outside time window is not a duplicate."""
        filter_ = DeduplicationFilter(time_window_s=60.0)

        # Add detection 120 seconds ago
        old_time = time.time() - 120
        filter_.add_detection(-42.0, 147.0, "DEVIL", timestamp=old_time)

        # New detection at same spot
        is_dup = filter_.is_duplicate(-42.0, 147.0, "DEVIL")
        assert is_dup is False

    def test_recent_detection_is_duplicate(self):
        """Detection within time window is a duplicate."""
        filter_ = DeduplicationFilter(time_window_s=300.0)

        # Add detection 30 seconds ago
        recent_time = time.time() - 30
        filter_.add_detection(-42.0, 147.0, "DEVIL", timestamp=recent_time)

        # New detection at same spot
        is_dup = filter_.is_duplicate(-42.0, 147.0, "DEVIL")
        assert is_dup is True

    def test_check_and_add_unique(self):
        """check_and_add returns True and adds unique detection."""
        filter_ = DeduplicationFilter()

        result = filter_.check_and_add(-42.0, 147.0, "DEVIL")
        assert result is True
        assert filter_.count == 1

    def test_check_and_add_duplicate(self):
        """check_and_add returns False for duplicate and does not add."""
        filter_ = DeduplicationFilter()

        filter_.add_detection(-42.0, 147.0, "DEVIL")
        result = filter_.check_and_add(-42.0, 147.0, "DEVIL")
        assert result is False
        assert filter_.count == 1  # Still only one detection

    def test_clear(self):
        """clear removes all detections."""
        filter_ = DeduplicationFilter()

        filter_.add_detection(-42.0, 147.0, "DEVIL")
        filter_.add_detection(-43.0, 148.0, "WALBY")
        assert filter_.count == 2

        filter_.clear()
        assert filter_.count == 0

    def test_prune_old(self):
        """prune_old removes old detections."""
        filter_ = DeduplicationFilter(time_window_s=60.0)

        current_time = time.time()

        # Add old detection (120s ago)
        filter_.add_detection(-42.0, 147.0, "DEVIL", timestamp=current_time - 120)

        # Add recent detection (30s ago)
        filter_.add_detection(-43.0, 148.0, "WALBY", timestamp=current_time - 30)

        assert filter_.count == 2

        # Prune old
        removed = filter_.prune_old(current_time)
        assert removed == 1
        assert filter_.count == 1

    def test_max_history_limit(self):
        """Filter respects max_history limit."""
        filter_ = DeduplicationFilter(max_history=5)

        # Add more than max_history detections
        for i in range(10):
            filter_.add_detection(-42.0 + i * 0.1, 147.0, "DEVIL")

        assert filter_.count == 5

    def test_default_thresholds(self):
        """Default thresholds match config values."""
        filter_ = DeduplicationFilter()
        assert filter_.distance_threshold_m == 50.0
        assert filter_.time_window_s == 300.0
        assert filter_.max_history == 100


class TestDeduplicationRealWorld:
    """Real-world scenario tests for deduplication."""

    def test_vehicle_driving_past_same_carcass(self):
        """Simulate vehicle detecting same carcass multiple times."""
        filter_ = DeduplicationFilter(
            distance_threshold_m=50.0,
            time_window_s=300.0,
        )

        base_time = time.time()
        base_lat, base_lon = -42.0, 147.0

        # First detection - should be unique
        is_unique = filter_.check_and_add(
            base_lat, base_lon, "DEVIL", timestamp=base_time
        )
        assert is_unique is True

        # Second detection 2 seconds later, 5m further - should be duplicate
        is_unique = filter_.check_and_add(
            base_lat + 0.00005,
            base_lon,
            "DEVIL",
            timestamp=base_time + 2,
        )
        assert is_unique is False

        # Third detection 5 seconds later, 10m further - should be duplicate
        is_unique = filter_.check_and_add(
            base_lat + 0.0001,
            base_lon,
            "DEVIL",
            timestamp=base_time + 5,
        )
        assert is_unique is False

    def test_multiple_species_same_location(self):
        """Different species at same location should all be recorded."""
        filter_ = DeduplicationFilter()

        is_unique = filter_.check_and_add(-42.0, 147.0, "DEVIL")
        assert is_unique is True

        is_unique = filter_.check_and_add(-42.0, 147.0, "WALBY")
        assert is_unique is True

        is_unique = filter_.check_and_add(-42.0, 147.0, "PADEM")
        assert is_unique is True

        assert filter_.count == 3

    def test_return_journey(self):
        """Detection on return journey after time window should be unique."""
        filter_ = DeduplicationFilter(time_window_s=300.0)

        base_time = time.time()

        # Detection on outward journey
        is_unique = filter_.check_and_add(
            -42.0, 147.0, "DEVIL", timestamp=base_time
        )
        assert is_unique is True

        # Same location on return journey 10 minutes later
        is_unique = filter_.check_and_add(
            -42.0, 147.0, "DEVIL", timestamp=base_time + 600
        )
        assert is_unique is True  # New detection after time window
