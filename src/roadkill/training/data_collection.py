"""Data collection module for RoadEye training datasets.

Downloads images from biodiversity databases (iNaturalist, GBIF, Roboflow)
and prepares them for YOLO training.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Tasmania bounding box
TASMANIA_BBOX = {
    "sw_lat": -43.7,
    "sw_lng": 143.5,
    "ne_lat": -40.5,
    "ne_lng": 149.0,
}

# Species mapping: RoadEye code -> iNaturalist taxon ID
# These IDs were verified against iNaturalist API
SPECIES_TAXON_MAP = {
    # Australian species (primary)
    "DEVIL": {"taxon_id": 40232, "scientific": "Sarcophilus harrisii"},
    "FCAT": {"taxon_id": 118552, "scientific": "Felis catus"},
    "PADEM": {"taxon_id": 42970, "scientific": "Thylogale billardierii"},
    "WALBY": {"taxon_id": 1453431, "scientific": "Notamacropus rufogriseus"},
    "WOMBAT": {"taxon_id": 43009, "scientific": "Vombatus ursinus"},
    "BPOSM": {"taxon_id": 42808, "scientific": "Trichosurus vulpecula"},
    "FDEER": {"taxon_id": 42161, "scientific": "Dama dama"},
    "BANDI": {"taxon_id": 43294, "scientific": "Isoodon obesulus"},
    "CURRA": {"taxon_id": 8423, "scientific": "Strepera graculina"},  # Pied Currawong
    "BRONZ": {"taxon_id": 3335, "scientific": "Phaps chalcoptera"},  # Common Bronzewing
    # European species (from Zenodo roadkill.at)
    "FOX": {"taxon_id": 42069, "scientific": "Vulpes vulpes"},
    "HARE": {"taxon_id": 43128, "scientific": "Lepus europaeus"},
    "HEDGE": {"taxon_id": 40790, "scientific": "Erinaceus europaeus"},
}


@dataclass
class ObservationRecord:
    """Record of a downloaded observation."""

    observation_id: str
    species_code: str
    scientific_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    observed_on: Optional[str]
    image_url: str
    local_path: Optional[str] = None
    is_dead: bool = False
    is_roadkill: bool = False
    quality_grade: str = "research"
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class INaturalistCollector:
    """Downloads observations and images from iNaturalist.

    Uses the pyinaturalist library for API access.
    Rate limited to 60 requests/minute.
    """

    def __init__(
        self,
        output_dir: Path,
        rate_limit_delay: float = 1.0,
        bbox: Optional[Dict[str, float]] = None,
    ):
        """Initialise the collector.

        Args:
            output_dir: Directory to save downloaded images
            rate_limit_delay: Seconds between API requests
            bbox: Bounding box for location filtering (defaults to Tasmania)
        """
        self.output_dir = Path(output_dir)
        self.rate_limit_delay = rate_limit_delay
        self.bbox = bbox or TASMANIA_BBOX

    def fetch_observations(
        self,
        species_code: str,
        max_results: int = 500,
        dead_only: bool = False,
        roadkill_only: bool = False,
        quality_grade: str = "research",
    ) -> List[ObservationRecord]:
        """Fetch observations from iNaturalist API.

        Args:
            species_code: RoadEye species code (e.g., 'DEVIL')
            max_results: Maximum observations to fetch
            dead_only: Only fetch dead specimen observations
            roadkill_only: Only fetch roadkill-tagged observations
            quality_grade: Quality filter ('research', 'needs_id', 'any')

        Returns:
            List of ObservationRecord objects
        """
        try:
            from pyinaturalist import get_observations
        except ImportError:
            logger.error("pyinaturalist not installed. Run: pip install pyinaturalist")
            return []

        if species_code not in SPECIES_TAXON_MAP:
            logger.error(f"Unknown species code: {species_code}")
            return []

        taxon_info = SPECIES_TAXON_MAP[species_code]
        taxon_id = taxon_info["taxon_id"]

        # Build query parameters
        params = {
            "taxon_id": taxon_id,
            "photos": True,
            "per_page": min(200, max_results),
            "nelat": self.bbox["ne_lat"],
            "nelng": self.bbox["ne_lng"],
            "swlat": self.bbox["sw_lat"],
            "swlng": self.bbox["sw_lng"],
        }

        if quality_grade != "any":
            params["quality_grade"] = quality_grade

        # Filter for dead specimens using annotation
        # term_id=17 is "Alive or Dead", term_value_id=19 is "Dead"
        if dead_only:
            params["term_id"] = 17
            params["term_value_id"] = 19

        logger.info(f"Fetching {species_code} observations from iNaturalist...")
        logger.info(f"Parameters: {params}")

        try:
            response = get_observations(**params)
            results = response.get("results", [])
            logger.info(f"Found {len(results)} observations for {species_code}")
        except Exception as e:
            logger.error(f"API error: {e}")
            return []

        # Convert to ObservationRecord objects
        observations = []
        for obs in results[:max_results]:
            if not obs.get("photos"):
                continue

            # Get location
            location = obs.get("location")
            lat, lng = None, None
            if location:
                try:
                    lat, lng = map(float, location.split(","))
                except (ValueError, AttributeError):
                    pass

            # Get first photo URL (medium size)
            photo = obs["photos"][0]
            image_url = photo.get("url", "").replace("square", "medium")

            # Check for dead/roadkill annotations
            is_dead = False
            annotations = obs.get("annotations", [])
            for ann in annotations:
                if ann.get("controlled_attribute", {}).get("id") == 17:  # Alive or Dead
                    if ann.get("controlled_value", {}).get("id") == 19:  # Dead
                        is_dead = True
                        break

            # Check observation fields for roadkill tag
            is_roadkill = False
            ofvs = obs.get("ofvs", [])
            for ofv in ofvs:
                field_name = ofv.get("name", "").lower()
                if "roadkill" in field_name or "road kill" in field_name:
                    is_roadkill = True
                    break

            # Skip if roadkill_only but not roadkill
            if roadkill_only and not is_roadkill:
                continue

            record = ObservationRecord(
                observation_id=str(obs["id"]),
                species_code=species_code,
                scientific_name=taxon_info["scientific"],
                latitude=lat,
                longitude=lng,
                observed_on=obs.get("observed_on"),
                image_url=image_url,
                is_dead=is_dead,
                is_roadkill=is_roadkill,
                quality_grade=obs.get("quality_grade", "unknown"),
                source="inaturalist",
                metadata={
                    "user": obs.get("user", {}).get("login"),
                    "place_guess": obs.get("place_guess"),
                    "photo_id": photo.get("id"),
                },
            )
            observations.append(record)

        return observations

    def download_images(
        self,
        observations: List[ObservationRecord],
        skip_existing: bool = True,
    ) -> int:
        """Download images for observations.

        Args:
            observations: List of observations to download
            skip_existing: Skip if image already exists locally

        Returns:
            Number of images successfully downloaded
        """
        try:
            import requests
            from tqdm import tqdm
        except ImportError:
            logger.error("Missing dependencies. Run: pip install requests tqdm")
            return 0

        downloaded = 0

        for obs in tqdm(observations, desc="Downloading images"):
            # Create species subdirectory
            species_dir = self.output_dir / obs.species_code
            species_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            filename = f"{obs.observation_id}.jpg"
            filepath = species_dir / filename

            # Skip if exists
            if skip_existing and filepath.exists():
                obs.local_path = str(filepath)
                downloaded += 1
                continue

            # Download image
            try:
                response = requests.get(obs.image_url, timeout=30)
                if response.status_code == 200:
                    filepath.write_bytes(response.content)
                    obs.local_path = str(filepath)
                    downloaded += 1
                else:
                    logger.warning(f"Failed to download {obs.observation_id}: HTTP {response.status_code}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.warning(f"Error downloading {obs.observation_id}: {e}")

        logger.info(f"Downloaded {downloaded}/{len(observations)} images")
        return downloaded

    def collect_species(
        self,
        species_code: str,
        max_images: int = 500,
        dead_only: bool = False,
        roadkill_only: bool = False,
    ) -> List[ObservationRecord]:
        """Convenience method to fetch and download for a species.

        Args:
            species_code: RoadEye species code
            max_images: Maximum images to download
            dead_only: Only dead specimens
            roadkill_only: Only roadkill observations

        Returns:
            List of downloaded observations
        """
        observations = self.fetch_observations(
            species_code=species_code,
            max_results=max_images,
            dead_only=dead_only,
            roadkill_only=roadkill_only,
        )

        if observations:
            self.download_images(observations)

        return observations


class ZenodoCollector:
    """Downloads roadkill images from Zenodo dataset (roadkill.at project)."""

    ZENODO_CSV_URL = "https://zenodo.org/records/5878813/files/2014-2020_Rk_Daten_Zenodo.csv?download=1"

    def __init__(self, output_dir: Path, rate_limit_delay: float = 0.5):
        """Initialise the collector.

        Args:
            output_dir: Directory to save downloaded images
            rate_limit_delay: Seconds between downloads
        """
        self.output_dir = Path(output_dir)
        self.rate_limit_delay = rate_limit_delay

    def download_csv(self) -> Path:
        """Download the Zenodo CSV file if not cached."""
        from urllib.request import urlretrieve

        csv_path = Path("/tmp/roadkill_zenodo.csv")
        if not csv_path.exists():
            logger.info("Downloading Zenodo roadkill CSV...")
            urlretrieve(self.ZENODO_CSV_URL, csv_path)
        return csv_path

    def collect_images(
        self,
        countries: Optional[List[str]] = None,
        max_images: int = 500,
    ) -> List[ObservationRecord]:
        """Collect roadkill images from Zenodo dataset.

        Only 329 records have images, so this is supplementary data.

        Args:
            countries: Filter by country names (None = all)
            max_images: Maximum images to download

        Returns:
            List of observation records
        """
        import csv

        import requests
        from tqdm import tqdm

        csv_path = self.download_csv()
        observations = []

        # Species mapping for Zenodo
        species_mapping = {
            "Macropodidae Gray, 1821": ("MACRO", "Kangaroo/Wallaby"),
            "Vombatidae Burnett, 1830": ("WOMBAT", "Wombat"),
            "Trichosurus vulpecula Kerr, 1792": ("BPOSM", "Brushtail Possum"),
            "Vulpes vulpes Linnaeus, 1758": ("FOX", "Red Fox"),
            "Felis catus Linnaeus, 1758": ("FCAT", "Feral Cat"),
            "Erinaceus Linnaeus, 1758": ("HEDGE", "Hedgehog"),
            "Lepus europaeus Pallas, 1778": ("HARE", "European Hare"),
        }

        # Parse CSV and find records with images
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")

            for row in reader:
                # Filter by country
                if countries and row.get("CNTRY_NAME") not in countries:
                    continue

                # Only records with images
                image_url = row.get("associatedMedia")
                if not image_url:
                    continue

                # Parse location
                try:
                    lat = float(row.get("decimalLatitude", 0))
                    lon = float(row.get("decimalLongitude", 0))
                except (ValueError, TypeError):
                    continue

                # Map species
                scientific_name = row.get("scientificName", "")
                species_info = species_mapping.get(scientific_name, ("OTHER", "Unknown"))

                record = ObservationRecord(
                    observation_id=row.get("occurrenceID", "").replace(":", "_"),
                    species_code=species_info[0],
                    scientific_name=scientific_name,
                    latitude=lat,
                    longitude=lon,
                    observed_on=row.get("eventDate"),
                    image_url=image_url,
                    is_dead=True,  # All Zenodo data is roadkill
                    is_roadkill=True,
                    quality_grade="verified",
                    source="zenodo",
                    metadata={
                        "country": row.get("CNTRY_NAME"),
                        "animal_class": row.get("class"),
                    },
                )
                observations.append(record)

                if len(observations) >= max_images:
                    break

        logger.info(f"Found {len(observations)} Zenodo records with images")

        # Download images
        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        for obs in tqdm(observations, desc="Downloading Zenodo images"):
            species_dir = self.output_dir / obs.species_code
            species_dir.mkdir(parents=True, exist_ok=True)

            filename = f"zenodo_{obs.observation_id}.jpg"
            filepath = species_dir / filename

            if filepath.exists():
                obs.local_path = str(filepath)
                downloaded += 1
                continue

            try:
                response = requests.get(obs.image_url, timeout=30)
                if response.status_code == 200:
                    filepath.write_bytes(response.content)
                    obs.local_path = str(filepath)
                    downloaded += 1
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.warning(f"Failed to download {obs.observation_id}: {e}")

        logger.info(f"Downloaded {downloaded}/{len(observations)} Zenodo images")
        return observations


class RoboflowCollector:
    """Downloads pre-labelled datasets from Roboflow Universe."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialise the collector.

        Args:
            api_key: Roboflow API key (can also be set via ROBOFLOW_API_KEY env var)
        """
        self.api_key = api_key

    def download_dataset(
        self,
        workspace: str,
        project: str,
        version: int,
        output_dir: Path,
        format: str = "yolov8",
    ) -> Optional[Path]:
        """Download a dataset from Roboflow Universe.

        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            output_dir: Directory to save dataset
            format: Export format (yolov8, coco, etc.)

        Returns:
            Path to downloaded dataset or None if failed
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            logger.error("roboflow not installed. Run: pip install roboflow")
            return None

        try:
            rf = Roboflow(api_key=self.api_key)
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download(format, location=str(output_dir))
            logger.info(f"Downloaded dataset to {output_dir}")
            return Path(output_dir)
        except Exception as e:
            logger.error(f"Failed to download Roboflow dataset: {e}")
            return None


class DatasetBuilder:
    """Builds YOLO-format datasets from collected images."""

    def __init__(self, output_dir: Path):
        """Initialise the builder.

        Args:
            output_dir: Directory to create YOLO dataset structure
        """
        self.output_dir = Path(output_dir)

    def create_structure(self) -> None:
        """Create YOLO dataset directory structure."""
        for split in ["train", "val"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        logger.info(f"Created dataset structure at {self.output_dir}")

    def build_from_observations(
        self,
        observations: List[ObservationRecord],
        class_names: List[str],
        train_ratio: float = 0.8,
        use_full_image_labels: bool = True,
        labels_dir: Optional[Path] = None,
    ) -> Path:
        """Build YOLO dataset from observation records.

        Args:
            observations: List of downloaded observations
            class_names: Ordered list of class names (species codes)
            train_ratio: Ratio of images for training set
            use_full_image_labels: Use full image as bounding box (no manual annotations)
            labels_dir: Directory containing pre-generated label files (from auto-labelling).
                        When provided, copies labels from here instead of generating placeholders.

        Returns:
            Path to data.yaml file
        """
        self.create_structure()

        # Filter observations with local paths
        valid_obs = [o for o in observations if o.local_path]
        if not valid_obs:
            logger.error("No valid observations with downloaded images")
            return self.output_dir / "data.yaml"

        # Create class index mapping
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Shuffle and split
        random.shuffle(valid_obs)
        split_idx = int(len(valid_obs) * train_ratio)
        train_obs = valid_obs[:split_idx]
        val_obs = valid_obs[split_idx:]

        def process_split(obs_list: List[ObservationRecord], split: str) -> None:
            for obs in obs_list:
                if obs.species_code not in class_to_idx:
                    continue

                src_path = Path(obs.local_path)
                if not src_path.exists():
                    continue

                # Copy image
                dst_img = self.output_dir / split / "images" / src_path.name
                shutil.copy(src_path, dst_img)

                # Handle label file
                label_path = self.output_dir / split / "labels" / f"{src_path.stem}.txt"

                if labels_dir:
                    # Copy pre-generated label from auto-labelling output
                    src_label = Path(labels_dir) / f"{src_path.stem}.txt"
                    if src_label.exists():
                        shutil.copy(src_label, label_path)
                        continue

                # Fall back to placeholder or empty label
                class_id = class_to_idx[obs.species_code]

                if use_full_image_labels:
                    label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                else:
                    label_content = ""

                label_path.write_text(label_content)

        process_split(train_obs, "train")
        process_split(val_obs, "val")

        # Create data.yaml
        data_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_names),
            "names": class_names,
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Dataset built: {len(train_obs)} train, {len(val_obs)} val")
        logger.info(f"data.yaml saved to {yaml_path}")

        return yaml_path

    def build_by_source(
        self,
        observations: List[ObservationRecord],
        class_names: List[str],
        train_ratio: float = 0.8,
        use_full_image_labels: bool = True,
        labels_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Build separate YOLO datasets per source, plus a combined dataset.

        Creates dataset_inaturalist/, dataset_zenodo/, dataset_combined/ directories
        as siblings of this builder's output_dir.

        Args:
            observations: List of downloaded observations
            class_names: Ordered list of class names (species codes)
            train_ratio: Ratio of images for training set
            use_full_image_labels: Use full image as bounding box
            labels_dir: Directory containing pre-generated label files

        Returns:
            Dict mapping source name to data.yaml path
        """
        from collections import Counter

        # Group observations by source
        by_source: Dict[str, List[ObservationRecord]] = {}
        for obs in observations:
            source = obs.source
            if source == "unknown" and obs.local_path:
                # Infer source from path
                if "inaturalist" in obs.local_path:
                    source = "inaturalist"
                elif "zenodo" in obs.local_path:
                    source = "zenodo"
            by_source.setdefault(source, []).append(obs)

        # Print summary table
        logger.info("\n" + "=" * 60)
        logger.info("Source Separation Summary")
        logger.info("=" * 60)

        all_species = sorted(set(o.species_code for o in observations if o.local_path))
        header = f"{'Species':<10}" + "".join(f"{s:<15}" for s in sorted(by_source.keys())) + f"{'Total':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        for species in all_species:
            row = f"{species:<10}"
            total = 0
            for source_name in sorted(by_source.keys()):
                count = sum(
                    1 for o in by_source[source_name]
                    if o.species_code == species and o.local_path
                )
                row += f"{count:<15}"
                total += count
            row += f"{total:<10}"
            logger.info(row)

        source_totals = f"{'TOTAL':<10}"
        grand_total = 0
        for source_name in sorted(by_source.keys()):
            count = sum(1 for o in by_source[source_name] if o.local_path)
            source_totals += f"{count:<15}"
            grand_total += count
        source_totals += f"{grand_total:<10}"
        logger.info("-" * len(header))
        logger.info(source_totals)
        logger.info("=" * 60)

        # Build per-source datasets
        base_dir = self.output_dir.parent
        results: Dict[str, Path] = {}

        for source_name, source_obs in by_source.items():
            source_dir = base_dir / f"dataset_{source_name}"
            # Only include classes present in this source
            source_species = sorted(set(
                o.species_code for o in source_obs
                if o.local_path and o.species_code in class_names
            ))
            if not source_species:
                logger.warning(f"No valid images for source '{source_name}', skipping")
                continue

            builder = DatasetBuilder(output_dir=source_dir)
            yaml_path = builder.build_from_observations(
                observations=source_obs,
                class_names=source_species,
                train_ratio=train_ratio,
                use_full_image_labels=use_full_image_labels,
                labels_dir=labels_dir,
            )
            results[source_name] = yaml_path
            logger.info(f"Built dataset for '{source_name}': {yaml_path}")

        # Build combined dataset
        combined_dir = base_dir / "dataset_combined"
        combined_builder = DatasetBuilder(output_dir=combined_dir)
        combined_species = sorted(set(
            o.species_code for o in observations
            if o.local_path and o.species_code in class_names
        ))
        combined_yaml = combined_builder.build_from_observations(
            observations=observations,
            class_names=combined_species,
            train_ratio=train_ratio,
            use_full_image_labels=use_full_image_labels,
            labels_dir=labels_dir,
        )
        results["combined"] = combined_yaml
        logger.info(f"Built combined dataset: {combined_yaml}")

        return results

    def save_metadata(
        self,
        observations: List[ObservationRecord],
        filename: str = "observations.json",
    ) -> Path:
        """Save observation metadata to JSON.

        Args:
            observations: List of observations
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        metadata_dir = self.output_dir.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        for obs in observations:
            # Convert observed_on to string if it's a datetime
            observed_on = obs.observed_on
            if observed_on is not None and not isinstance(observed_on, str):
                observed_on = str(observed_on)

            metadata.append({
                "observation_id": obs.observation_id,
                "species_code": obs.species_code,
                "scientific_name": obs.scientific_name,
                "latitude": obs.latitude,
                "longitude": obs.longitude,
                "observed_on": observed_on,
                "image_url": obs.image_url,
                "local_path": obs.local_path,
                "is_dead": obs.is_dead,
                "is_roadkill": obs.is_roadkill,
                "quality_grade": obs.quality_grade,
                "source": obs.source,
            })

        output_path = metadata_dir / filename
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata for {len(metadata)} observations to {output_path}")
        return output_path


def collect_training_data(
    output_dir: Path,
    species_codes: Optional[List[str]] = None,
    max_images_per_species: int = 500,
    dead_only: bool = False,
    roadkill_only: bool = False,
    train_ratio: float = 0.8,
) -> Tuple[Path, List[ObservationRecord]]:
    """Convenience function to collect and prepare training data.

    Args:
        output_dir: Base output directory
        species_codes: List of species to collect (None = all Tasmania species)
        max_images_per_species: Max images per species
        dead_only: Only dead specimens
        roadkill_only: Only roadkill observations
        train_ratio: Train/val split ratio

    Returns:
        Tuple of (data.yaml path, all observations)
    """
    output_dir = Path(output_dir)

    # Default to all Tasmania species
    if species_codes is None:
        species_codes = list(SPECIES_TAXON_MAP.keys())

    # Create collector
    raw_dir = output_dir / "raw" / "inaturalist"
    collector = INaturalistCollector(output_dir=raw_dir)

    # Collect all species
    all_observations = []
    for species_code in species_codes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Collecting {species_code}...")
        logger.info(f"{'='*50}")

        observations = collector.collect_species(
            species_code=species_code,
            max_images=max_images_per_species,
            dead_only=dead_only,
            roadkill_only=roadkill_only,
        )
        all_observations.extend(observations)

    # Build dataset
    dataset_dir = output_dir / "dataset"
    builder = DatasetBuilder(output_dir=dataset_dir)

    # Use only species that have observations
    collected_species = list(set(o.species_code for o in all_observations if o.local_path))
    # Sort by MEWC class ID order
    class_names = [s for s in species_codes if s in collected_species]

    data_yaml = builder.build_from_observations(
        observations=all_observations,
        class_names=class_names,
        train_ratio=train_ratio,
    )

    # Save metadata
    builder.save_metadata(all_observations)

    return data_yaml, all_observations
