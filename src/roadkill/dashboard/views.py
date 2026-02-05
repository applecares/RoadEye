"""Dashboard views and routes."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from roadkill.core.species import get_species_registry
from roadkill.database import get_db
from roadkill.database.models import DetectionEvent, VerificationStatus

# Set up templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

# Tasmania species colours and icons
SPECIES_DISPLAY = {
    "DEVIL": {"colour": "#ff6b6b", "icon": "😈"},
    "FCAT": {"colour": "#868e96", "icon": "🐱"},
    "PADEM": {"colour": "#ffa94d", "icon": "🦘"},
    "WALBY": {"colour": "#fab005", "icon": "🦘"},
    "WOMBAT": {"colour": "#845ef7", "icon": "🐻"},
    "BPOSM": {"colour": "#22b8cf", "icon": "🐿️"},
    "FDEER": {"colour": "#be4bdb", "icon": "🦌"},
    "BANDI": {"colour": "#51cf66", "icon": "🐀"},
    "CURRA": {"colour": "#339af0", "icon": "🐦"},
    "BRONZ": {"colour": "#f06595", "icon": "🐦"},
    "UNKNOWN": {"colour": "#495057", "icon": "🐾"},
}


def get_time_filter(time_str: str) -> datetime:
    """Convert time filter string to datetime."""
    now = datetime.now(timezone.utc)
    filters = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "all": timedelta(days=3650),
    }
    return now - filters.get(time_str, timedelta(days=7))


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    time: str = Query("7d", description="Time filter"),
    species: str = Query("all", description="Species filter"),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Serve the main dashboard."""
    now = datetime.now(timezone.utc)
    start_time = get_time_filter(time)

    # Build query
    query = db.query(DetectionEvent).filter(DetectionEvent.timestamp >= start_time)

    # Species filter
    if species == "threatened":
        query = query.filter(DetectionEvent.is_threatened.is_(True))
    elif species != "all":
        query = query.filter(
            func.upper(DetectionEvent.species_code) == species.upper()
        )

    events = query.order_by(DetectionEvent.timestamp.desc()).limit(500).all()

    # Calculate stats
    total = db.query(DetectionEvent).count()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today = (
        db.query(DetectionEvent)
        .filter(DetectionEvent.timestamp >= today_start)
        .count()
    )
    threatened = (
        db.query(DetectionEvent)
        .filter(DetectionEvent.is_threatened.is_(True))
        .filter(DetectionEvent.timestamp >= start_time)
        .count()
    )
    vehicles = (
        db.query(DetectionEvent.vehicle_id)
        .filter(DetectionEvent.timestamp >= now - timedelta(hours=24))
        .filter(DetectionEvent.vehicle_id.isnot(None))
        .distinct()
        .count()
    )

    # Species counts for chart
    species_counts: Dict[str, int] = {}
    for e in events:
        code = e.species_code or "UNKNOWN"
        species_counts[code] = species_counts.get(code, 0) + 1

    if not species_counts:
        species_counts = {"No data": 1}

    # Get species list for dropdown
    registry = get_species_registry()
    species_list = registry.list_all()

    # Build events JSON for map
    events_json = [
        {
            "event_id": e.event_id,
            "latitude": e.latitude,
            "longitude": e.longitude,
            "species_code": e.species_code,
            "species_common": e.species_common,
            "confidence": e.confidence,
            "is_threatened": e.is_threatened,
            "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M") if e.timestamp else "",
        }
        for e in events
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": {
                "total": total,
                "today": today,
                "threatened": threatened,
                "vehicles": vehicles,
            },
            "filters": {"time": time, "species": species},
            "events": events,
            "events_json": events_json,
            "species_counts": species_counts,
            "species_colours": SPECIES_DISPLAY,
            "species_list": species_list,
        },
    )


@router.post("/api/demo-data")
async def create_demo_data(
    count: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Create demo detection events for testing."""
    registry = get_species_registry()
    species_list = registry.list_all()

    # Tasmania hotspot locations
    hotspots = [
        (-41.0500, 144.6500, "Arthur River Road"),
        (-41.8000, 145.5000, "Murchison Highway"),
        (-43.0500, 147.0500, "Huon Highway"),
        (-42.1500, 145.8500, "Lyell Highway"),
        (-41.1500, 146.2500, "Bass Highway"),
        (-42.8821, 147.3272, "Hobart Area"),
        (-41.4332, 147.1441, "Launceston Area"),
        (-41.1800, 146.3600, "Devonport Area"),
    ]

    created = 0
    for _ in range(count):
        # Pick random hotspot
        lat, lon, road = random.choice(hotspots)

        # Add some variation
        lat += random.uniform(-0.1, 0.1)
        lon += random.uniform(-0.1, 0.1)

        # Pick random species (weighted towards common ones)
        species = random.choice(species_list)

        # Generate confidence
        confidence = random.uniform(0.5, 0.98)

        # Determine verification status based on confidence
        if confidence >= 0.95:
            status = VerificationStatus.AUTO_APPROVED
        elif confidence >= 0.75:
            status = VerificationStatus.MANUAL_REVIEW
        else:
            status = VerificationStatus.PENDING

        event = DetectionEvent(
            event_id=f"DEMO_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc) - timedelta(
                hours=random.randint(0, 168)
            ),
            latitude=lat,
            longitude=lon,
            gps_accuracy=random.uniform(3, 15),
            species_code=species.code,
            species_scientific=species.scientific_name,
            species_common=species.common_name,
            confidence=round(confidence, 3),
            is_threatened=species.is_threatened,
            source_type="dashcam",
            vehicle_id=f"DEMO_VEH_{random.randint(1, 5):02d}",
            speed_kmh=random.uniform(40, 100),
            verification_status=status,
        )

        db.add(event)
        created += 1

    db.commit()

    return {
        "status": "success",
        "events_created": created,
        "message": f"Created {created} demo detection events",
    }


@router.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get platform statistics."""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total = db.query(DetectionEvent).count()
    today = (
        db.query(DetectionEvent)
        .filter(DetectionEvent.timestamp >= today_start)
        .count()
    )
    week = (
        db.query(DetectionEvent)
        .filter(DetectionEvent.timestamp >= now - timedelta(days=7))
        .count()
    )
    threatened = (
        db.query(DetectionEvent)
        .filter(DetectionEvent.is_threatened.is_(True))
        .count()
    )
    vehicles = (
        db.query(DetectionEvent.vehicle_id)
        .filter(DetectionEvent.vehicle_id.isnot(None))
        .distinct()
        .count()
    )

    # Species breakdown
    species_query = (
        db.query(DetectionEvent.species_code, func.count(DetectionEvent.id))
        .group_by(DetectionEvent.species_code)
        .all()
    )
    species_counts = dict(species_query)

    return {
        "total_events": total,
        "today": today,
        "this_week": week,
        "threatened_species_events": threatened,
        "total_vehicles": vehicles,
        "species_breakdown": species_counts,
    }


@router.get("/api/events")
async def get_events(
    limit: int = Query(100, ge=1, le=1000),
    species: Optional[str] = None,
    days: int = Query(7, ge=1, le=365),
    threatened_only: bool = False,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get detection events."""
    start_time = datetime.now(timezone.utc) - timedelta(days=days)

    query = db.query(DetectionEvent).filter(DetectionEvent.timestamp >= start_time)

    if species:
        query = query.filter(
            func.upper(DetectionEvent.species_code) == species.upper()
        )

    if threatened_only:
        query = query.filter(DetectionEvent.is_threatened.is_(True))

    events = query.order_by(DetectionEvent.timestamp.desc()).limit(limit).all()

    return [
        {
            "event_id": e.event_id,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "latitude": e.latitude,
            "longitude": e.longitude,
            "species_code": e.species_code,
            "species_common": e.species_common,
            "confidence": e.confidence,
            "is_threatened": e.is_threatened,
            "verification_status": e.verification_status.value if e.verification_status else None,
            "vehicle_id": e.vehicle_id,
        }
        for e in events
    ]
