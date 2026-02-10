"""Load training images into FiftyOne grouped by source for review.

Run with: python scripts/review_sources.py
"""

import fiftyone as fo
from pathlib import Path

RAW_DIR = Path("data/training/raw")

# Delete previous dataset if it exists
if fo.dataset_exists("roadeye-source-review"):
    fo.delete_dataset("roadeye-source-review")

dataset = fo.Dataset("roadeye-source-review", persistent=True)

samples = []
for source_dir in sorted(RAW_DIR.iterdir()):
    if not source_dir.is_dir():
        continue
    source_name = source_dir.name

    for species_dir in sorted(source_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species_code = species_dir.name

        for img_path in sorted(species_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue

            sample = fo.Sample(filepath=str(img_path.resolve()))
            sample.tags = [source_name, species_code]
            sample["source"] = source_name
            sample["species"] = species_code
            samples.append(sample)

dataset.add_samples(samples)

print(f"Loaded {len(dataset)} images")
print(f"\nBy source:")
for src in dataset.distinct("source"):
    count = len(dataset.match(fo.ViewField("source") == src))
    print(f"  {src}: {count}")

print(f"\nBy species:")
for sp in sorted(dataset.distinct("species")):
    count = len(dataset.match(fo.ViewField("species") == sp))
    print(f"  {sp}: {count}")

print("\nLaunching FiftyOne app...")
print("  Filter by 'source' field in the sidebar to compare iNaturalist vs Zenodo")
print("  Filter by 'species' to see per-class quality")
print("  Press Ctrl+C to stop\n")

session = fo.launch_app(dataset)
session.wait()
