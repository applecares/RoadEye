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


@cli.group()
def training() -> None:
    """Training data collection and model training commands."""
    pass


@training.command("collect")
@click.option(
    "--source",
    "-s",
    type=click.Choice(["inaturalist", "roboflow"]),
    default="inaturalist",
    help="Data source to collect from",
)
@click.option(
    "--species",
    "-sp",
    multiple=True,
    help="Species codes to collect (e.g., DEVIL FCAT). If not specified, collects all.",
)
@click.option("--max-images", default=500, help="Maximum images per species")
@click.option("--dead-only", is_flag=True, help="Only collect dead specimen observations")
@click.option("--roadkill-only", is_flag=True, help="Only collect roadkill-tagged observations")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./data/training",
    help="Output directory for collected data",
)
def training_collect(
    source: str,
    species: tuple,
    max_images: int,
    dead_only: bool,
    roadkill_only: bool,
    output: str,
) -> None:
    """Collect training images from biodiversity databases.

    Downloads images from iNaturalist or Roboflow and prepares them
    for YOLO training.

    Examples:

        roadeye training collect --species DEVIL FCAT --max-images 100

        roadeye training collect --dead-only --roadkill-only

        roadeye training collect --source roboflow
    """
    from pathlib import Path

    from roadkill.training.data_collection import (
        SPECIES_TAXON_MAP,
        collect_training_data,
    )

    output_dir = Path(output)
    species_list = list(species) if species else None

    if species_list:
        # Validate species codes
        invalid = [s for s in species_list if s not in SPECIES_TAXON_MAP]
        if invalid:
            click.echo(f"Error: Unknown species codes: {invalid}", err=True)
            click.echo(f"Valid codes: {list(SPECIES_TAXON_MAP.keys())}", err=True)
            return

    click.echo("RoadEye Training Data Collection")
    click.echo("=" * 40)
    click.echo(f"Source: {source}")
    click.echo(f"Species: {species_list or 'all'}")
    click.echo(f"Max images per species: {max_images}")
    click.echo(f"Dead only: {dead_only}")
    click.echo(f"Roadkill only: {roadkill_only}")
    click.echo(f"Output: {output_dir}")
    click.echo("=" * 40)
    click.echo()

    if source == "inaturalist":
        data_yaml, observations = collect_training_data(
            output_dir=output_dir,
            species_codes=species_list,
            max_images_per_species=max_images,
            dead_only=dead_only,
            roadkill_only=roadkill_only,
        )

        click.echo()
        click.echo("Collection complete!")
        click.echo(f"  Total observations: {len(observations)}")
        click.echo(f"  Downloaded images: {len([o for o in observations if o.local_path])}")
        click.echo(f"  Dataset YAML: {data_yaml}")
    else:
        click.echo("Roboflow collection requires API key.")
        click.echo("Set ROBOFLOW_API_KEY environment variable and use --workspace/--project options.")


@training.command("list-species")
def training_list_species() -> None:
    """List available species codes for data collection."""
    from roadkill.training.data_collection import SPECIES_TAXON_MAP

    click.echo("Available species codes for training data collection:")
    click.echo()
    click.echo(f"{'Code':<10} {'Scientific Name':<35} {'iNat Taxon ID'}")
    click.echo("-" * 60)
    for code, info in SPECIES_TAXON_MAP.items():
        click.echo(f"{code:<10} {info['scientific']:<35} {info['taxon_id']}")


@training.command("import-observations")
@click.option(
    "--source",
    "-s",
    type=click.Choice(["inaturalist", "zenodo"]),
    default="inaturalist",
    help="Source of observations",
)
@click.option(
    "--metadata-file",
    "-m",
    type=click.Path(exists=True),
    required=False,
    help="Path to observations.json metadata file (required for inaturalist)",
)
@click.option("--roadkill-only", is_flag=True, help="Only import roadkill observations")
@click.option(
    "--countries",
    "-c",
    multiple=True,
    default=["Australia", "New Zealand"],
    help="Countries to import from Zenodo (default: Australia, New Zealand)",
)
def training_import_observations(
    source: str, metadata_file: str, roadkill_only: bool, countries: tuple
) -> None:
    """Import collected observations into RoadEye database.

    This populates the map with historical observations from iNaturalist or Zenodo.
    Observations are marked as CITIZEN source type with VERIFIED status.

    Examples:

        roadeye training import-observations -s inaturalist -m data/training/metadata/observations.json

        roadeye training import-observations -s zenodo -c Australia -c "New Zealand"
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from roadkill.database import get_db_session, init_db
    from roadkill.database.models import DetectionEvent, SourceType, VerificationStatus

    # Ensure database exists
    init_db()

    if source == "zenodo":
        _import_zenodo_observations(countries)
    else:
        if not metadata_file:
            click.echo("Error: --metadata-file is required for iNaturalist import", err=True)
            return
        _import_inaturalist_observations(metadata_file, roadkill_only)


def _import_zenodo_observations(countries: tuple) -> None:
    """Import roadkill observations from Zenodo CSV."""
    import csv
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from urllib.request import urlretrieve

    from roadkill.database import get_db_session
    from roadkill.database.models import DetectionEvent, SourceType, VerificationStatus

    zenodo_url = "https://zenodo.org/records/5878813/files/2014-2020_Rk_Daten_Zenodo.csv?download=1"
    csv_path = Path("/tmp/roadkill_zenodo.csv")

    # Download if not exists
    if not csv_path.exists():
        click.echo("Downloading Zenodo roadkill dataset...")
        urlretrieve(zenodo_url, csv_path)

    click.echo(f"Importing Zenodo observations for: {', '.join(countries)}")

    # Species code mapping for common AU/NZ species
    species_mapping = {
        "Macropodidae Gray, 1821": ("MACRO", "Macropodidae", "Kangaroo/Wallaby"),
        "Vombatidae Burnett, 1830": ("WOMBAT", "Vombatus ursinus", "Wombat"),
        "Trichosurus vulpecula Kerr, 1792": ("BPOSM", "Trichosurus vulpecula", "Brushtail Possum"),
        "Phascolarctos cinereus Goldfuss, 1817": ("KOALA", "Phascolarctos cinereus", "Koala"),
        "Vulpes vulpes Linnaeus, 1758": ("FOX", "Vulpes vulpes", "Red Fox"),
        "Felis catus Linnaeus, 1758": ("FCAT", "Felis catus", "Feral Cat"),
        "Oryctolagus cuniculus Linnaeus, 1758": ("RABBIT", "Oryctolagus cuniculus", "Rabbit"),
        "Erinaceus Linnaeus, 1758": ("HEDGE", "Erinaceus europaeus", "Hedgehog"),
        "Lepus europaeus Pallas, 1778": ("HARE", "Lepus europaeus", "European Hare"),
        "Gymnorhina tibicen Latham, 1802": ("MAGPIE", "Gymnorhina tibicen", "Magpie"),
    }

    imported = 0
    skipped = 0

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")

        with get_db_session() as db:
            for row in reader:
                # Filter by country
                if row.get("CNTRY_NAME") not in countries:
                    continue

                # Skip if missing location
                try:
                    lat = float(row.get("decimalLatitude", 0))
                    lon = float(row.get("decimalLongitude", 0))
                    if lat == 0 or lon == 0:
                        skipped += 1
                        continue
                except (ValueError, TypeError):
                    skipped += 1
                    continue

                # Check if already imported
                event_id = f"ZENODO_{row.get('occurrenceID', '')}"
                existing = db.query(DetectionEvent).filter(
                    DetectionEvent.event_id == event_id
                ).first()

                if existing:
                    skipped += 1
                    continue

                # Parse timestamp
                timestamp = datetime.now(timezone.utc)
                if row.get("eventDate"):
                    try:
                        # Handle DD.MM.YYYY format
                        date_str = row["eventDate"]
                        if "." in date_str:
                            parts = date_str.split(".")
                            if len(parts) == 3:
                                timestamp = datetime(
                                    int(parts[2]), int(parts[1]), int(parts[0]),
                                    tzinfo=timezone.utc
                                )
                    except (ValueError, IndexError):
                        pass

                # Map species
                scientific_name = row.get("scientificName", "")
                species_info = species_mapping.get(
                    scientific_name,
                    ("OTHER", scientific_name.split()[0] if scientific_name else "Unknown", "Unknown")
                )

                event = DetectionEvent(
                    event_id=event_id,
                    timestamp=timestamp,
                    latitude=lat,
                    longitude=lon,
                    gps_accuracy=500.0,  # Zenodo uses 500m uncertainty radius
                    species_code=species_info[0],
                    species_scientific=species_info[1],
                    species_common=species_info[2],
                    confidence=1.0,  # Verified citizen science
                    source_type=SourceType.CITIZEN,
                    verification_status=VerificationStatus.VERIFIED,
                    is_threatened=False,
                    image_path=row.get("associatedMedia") or None,
                    metadata_json=json.dumps({
                        "source": "zenodo_roadkill",
                        "occurrence_id": row.get("occurrenceID"),
                        "country": row.get("CNTRY_NAME"),
                        "region": row.get("name"),
                        "animal_class": row.get("class"),
                        "original_scientific_name": scientific_name,
                    }),
                )
                db.add(event)
                imported += 1

                if imported % 50 == 0:
                    click.echo(f"  Imported {imported} observations...")

    click.echo()
    click.echo(f"Import complete: {imported} imported, {skipped} skipped")


def _import_inaturalist_observations(metadata_file: str, roadkill_only: bool) -> None:
    """Import observations from iNaturalist metadata file."""
    import json
    from datetime import datetime, timezone

    from roadkill.database import get_db_session
    from roadkill.database.models import DetectionEvent, SourceType, VerificationStatus

    click.echo("Importing iNaturalist observations to database...")

    # Load metadata
    with open(metadata_file) as f:
        observations = json.load(f)

    if roadkill_only:
        observations = [o for o in observations if o.get("is_roadkill")]

    click.echo(f"Found {len(observations)} observations to import")

    imported = 0
    skipped = 0

    with get_db_session() as db:
        for obs in observations:
            # Skip if missing location
            if not obs.get("latitude") or not obs.get("longitude"):
                skipped += 1
                continue

            # Check if already imported
            existing = db.query(DetectionEvent).filter(
                DetectionEvent.event_id == f"INAT_{obs['observation_id']}"
            ).first()

            if existing:
                skipped += 1
                continue

            # Parse timestamp
            timestamp = datetime.now(timezone.utc)
            if obs.get("observed_on"):
                try:
                    obs_date = obs["observed_on"]
                    if isinstance(obs_date, str):
                        timestamp = datetime.fromisoformat(obs_date.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            event = DetectionEvent(
                event_id=f"INAT_{obs['observation_id']}",
                timestamp=timestamp,
                latitude=obs["latitude"],
                longitude=obs["longitude"],
                gps_accuracy=100.0,  # iNaturalist doesn't provide accuracy
                species_code=obs["species_code"],
                species_scientific=obs.get("scientific_name"),
                confidence=1.0,  # Research grade = verified
                source_type=SourceType.CITIZEN,
                verification_status=VerificationStatus.VERIFIED,
                is_threatened=False,  # Will be set based on species
                metadata_json=json.dumps({
                    "source": "inaturalist",
                    "observation_id": obs["observation_id"],
                    "quality_grade": obs.get("quality_grade"),
                    "is_dead": obs.get("is_dead", False),
                    "is_roadkill": obs.get("is_roadkill", False),
                }),
            )
            db.add(event)
            imported += 1

            if imported % 50 == 0:
                click.echo(f"  Imported {imported} observations...")

    click.echo()
    click.echo(f"Import complete: {imported} imported, {skipped} skipped")


if __name__ == "__main__":
    cli()
