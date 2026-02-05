"""Main CLI entry point for RoadEye."""

from __future__ import annotations

from typing import Optional

import click

from roadkill import __version__


@click.group()
@click.version_option(version=__version__, prog_name="roadeye")
def cli() -> None:
    """RoadEye - Wildlife Roadkill Detection Platform.

    Detect, map, and analyse wildlife roadkill incidents.
    """
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def server(host: str, port: int, config: Optional[str], reload: bool) -> None:
    """Start the API server with dashboard."""
    import uvicorn

    click.echo("RoadEye - Wildlife Detection Platform")
    click.echo(f"Dashboard: http://localhost:{port}")
    click.echo(f"API Docs:  http://localhost:{port}/docs")
    click.echo("Press Ctrl+C to stop")

    uvicorn.run(
        "roadkill.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--source", "-s", default="0", help="Video source (0 for webcam, or file path)")
@click.option("--device-id", default="DEVICE001", help="Unique device identifier")
@click.option("--offline", is_flag=True, help="Run in offline mode")
def edge(source: str, device_id: str, offline: bool) -> None:
    """Run edge detection device."""
    click.echo(f"Starting edge device {device_id} with source {source}")
    # TODO: Implement edge device


@cli.group()
def demo() -> None:
    """Demo data commands."""
    pass


@demo.command("generate")
@click.option("--count", default=50, help="Number of events to generate")
def demo_generate(count: int) -> None:
    """Generate demo detection events."""
    import random
    import uuid
    from datetime import datetime, timedelta, timezone

    from roadkill.core.species import get_species_registry
    from roadkill.database import get_db_session, init_db
    from roadkill.database.models import DetectionEvent, VerificationStatus

    click.echo(f"Generating {count} demo events...")

    # Ensure database exists
    init_db()

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
    ]

    with get_db_session() as db:
        for i in range(count):
            lat, lon, road = random.choice(hotspots)
            lat += random.uniform(-0.1, 0.1)
            lon += random.uniform(-0.1, 0.1)

            species = random.choice(species_list)
            confidence = random.uniform(0.5, 0.98)

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
                vehicle_id=f"DEMO_VEH_{random.randint(1, 5):02d}",
                speed_kmh=random.uniform(40, 100),
                verification_status=status,
            )
            db.add(event)

            if (i + 1) % 10 == 0:
                click.echo(f"  Created {i + 1}/{count} events...")

    click.echo(f"Successfully created {count} demo events")


@cli.group()
def report() -> None:
    """Report generation commands."""
    pass


@report.command("weekly")
@click.option("--week", required=True, help="ISO week (e.g., 2026-W06)")
def report_weekly(week: str) -> None:
    """Generate weekly report."""
    click.echo(f"Generating weekly report for {week}")
    # TODO: Implement report generation


if __name__ == "__main__":
    cli()
