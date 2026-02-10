"""Auto-labelling module for RoadEye training datasets.

Uses foundation models (Grounding DINO, YOLO-World) to generate
real bounding box annotations, replacing placeholder full-image labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Text prompts mapped to species codes. Multiple prompts per species
# improve recall for foundation models that may not know niche names.
SPECIES_PROMPTS: Dict[str, List[str]] = {
    # Australian species
    "DEVIL": ["tasmanian devil"],
    "FCAT": ["cat", "feral cat"],
    "PADEM": ["pademelon", "small wallaby"],
    "WALBY": ["wallaby", "kangaroo"],
    "WOMBAT": ["wombat"],
    "BPOSM": ["possum", "brushtail possum"],
    "FDEER": ["deer", "fallow deer"],
    "BANDI": ["bandicoot", "small mammal"],
    "BRONZ": ["pigeon", "bird", "common bronzewing"],
    # European species (Zenodo)
    "FOX": ["fox", "red fox"],
    "HARE": ["hare", "rabbit"],
    "HEDGE": ["hedgehog"],
    "OTHER": ["animal"],
}


def _build_prompt_to_class(
    species_classes: List[str],
) -> Tuple[str, Dict[str, str]]:
    """Build a Grounding DINO text prompt and reverse mapping.

    Returns:
        Tuple of (period-separated prompt string, {prompt_phrase: species_code})
    """
    prompt_to_code: Dict[str, str] = {}
    all_prompts: List[str] = []

    for species_code in species_classes:
        prompts = SPECIES_PROMPTS.get(species_code, [species_code.lower()])
        for p in prompts:
            prompt_to_code[p] = species_code
            all_prompts.append(p)

    # Grounding DINO expects period-separated prompts
    text_prompt = ". ".join(all_prompts) + "."
    return text_prompt, prompt_to_code


def _boxes_to_yolo(
    boxes: List[Tuple[float, float, float, float]],
    class_ids: List[int],
    confidences: List[float],
) -> str:
    """Convert bounding boxes to YOLO format string.

    Args:
        boxes: List of (x_centre, y_centre, width, height) normalised
        class_ids: List of class indices
        confidences: List of confidence scores

    Returns:
        YOLO-format label file content
    """
    lines = []
    for (cx, cy, w, h), class_id in zip(boxes, class_ids):
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


class GroundingDINOLabeller:
    """Auto-labels images using Grounding DINO via HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = 0.20,
        text_threshold: float = 0.15,
    ):
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Run: pip install transformers torch"
            )

        logger.info(f"Loading Grounding DINO model: {self.model_id}")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        logger.info("Model loaded")

    def label_single_image(
        self,
        image_path: Path,
        species_classes: List[str],
        class_to_idx: Dict[str, int],
    ) -> Dict[str, Any]:
        """Generate bounding box labels for a single image.

        Args:
            image_path: Path to image file
            species_classes: List of species codes to detect
            class_to_idx: Mapping from species code to class index

        Returns:
            Dict with keys: boxes, class_ids, confidences, labels, yolo_content
        """
        import torch
        from PIL import Image

        self._load_model()

        text_prompt, prompt_to_code = _build_prompt_to_class(species_classes)

        image = Image.open(image_path)
        w, h = image.size

        inputs = self._processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )

        boxes = []
        class_ids = []
        confidences = []
        label_names = []

        for box, score, label_text in zip(
            results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        ):
            label_lower = label_text.lower().strip()

            # Match to species code via prompt mapping
            matched_code = None
            for prompt_phrase, code in prompt_to_code.items():
                if prompt_phrase in label_lower or label_lower in prompt_phrase:
                    matched_code = code
                    break

            if matched_code is None or matched_code not in class_to_idx:
                continue

            x1, y1, x2, y2 = box.tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            boxes.append((cx, cy, bw, bh))
            class_ids.append(class_to_idx[matched_code])
            confidences.append(float(score))
            label_names.append(matched_code)

        yolo_content = _boxes_to_yolo(boxes, class_ids, confidences)

        return {
            "boxes": boxes,
            "class_ids": class_ids,
            "confidences": confidences,
            "labels": label_names,
            "yolo_content": yolo_content,
        }

    def label_directory(
        self,
        image_dir: Path,
        output_dir: Path,
        species_classes: List[str],
        existing_species_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Auto-label all images in a directory.

        Args:
            image_dir: Directory containing images (can have species subfolders)
            output_dir: Directory to write YOLO .txt label files
            species_classes: Ordered list of species codes
            existing_species_map: Optional {filename: expected_species_code} for
                                  cross-validation against folder structure

        Returns:
            Labelling report dict with stats and flagged disagreements
        """
        from tqdm import tqdm

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_to_idx = {name: idx for idx, name in enumerate(species_classes)}

        # Collect all images
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_paths.extend(Path(image_dir).rglob(ext))

        logger.info(f"Found {len(image_paths)} images to label")

        report = {
            "total_images": len(image_paths),
            "labelled": 0,
            "empty": 0,
            "disagreements": [],
            "per_class": {cls: 0 for cls in species_classes},
            "avg_confidence": 0.0,
        }
        all_confidences = []

        for img_path in tqdm(image_paths, desc="Auto-labelling"):
            try:
                result = self.label_single_image(
                    img_path, species_classes, class_to_idx
                )
            except Exception as e:
                logger.warning(f"Failed to label {img_path.name}: {e}")
                report["empty"] += 1
                continue

            # Write label file
            label_path = output_dir / f"{img_path.stem}.txt"
            label_path.write_text(result["yolo_content"])

            if result["boxes"]:
                report["labelled"] += 1
                all_confidences.extend(result["confidences"])
                for label in result["labels"]:
                    if label in report["per_class"]:
                        report["per_class"][label] += 1
            else:
                report["empty"] += 1

            # Cross-validate against expected species
            if existing_species_map and img_path.name in existing_species_map:
                expected = existing_species_map[img_path.name]
                detected = result["labels"]
                if detected and expected not in detected:
                    report["disagreements"].append({
                        "image": img_path.name,
                        "expected": expected,
                        "detected": detected,
                        "confidences": result["confidences"],
                    })

        if all_confidences:
            report["avg_confidence"] = sum(all_confidences) / len(all_confidences)

        logger.info(f"Labelling complete: {report['labelled']} labelled, "
                     f"{report['empty']} empty, "
                     f"{len(report['disagreements'])} disagreements")

        # Save report
        report_path = output_dir / "labelling_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_path}")

        return report


class YOLOWorldLabeller:
    """Auto-labels images using YOLO-World (built into Ultralytics)."""

    def __init__(
        self,
        model_name: str = "yolov8x-worldv2.pt",
        device: str = "cuda",
        conf_threshold: float = 0.20,
    ):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = None

    def _load_model(self, species_classes: List[str]) -> None:
        """Lazy-load model and set classes."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required. Run: pip install ultralytics")

        if self._model is None:
            logger.info(f"Loading YOLO-World model: {self.model_name}")
            self._model = YOLO(self.model_name)

        # Build text class list from species prompts
        # YOLO-World uses the first prompt per species as the class name
        class_texts = []
        for code in species_classes:
            prompts = SPECIES_PROMPTS.get(code, [code.lower()])
            class_texts.append(prompts[0])

        self._model.set_classes(class_texts)
        logger.info(f"Set {len(class_texts)} detection classes")

    def label_directory(
        self,
        image_dir: Path,
        output_dir: Path,
        species_classes: List[str],
        existing_species_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Auto-label all images using YOLO-World.

        Args:
            image_dir: Directory containing images
            output_dir: Directory to write YOLO .txt label files
            species_classes: Ordered list of species codes
            existing_species_map: Optional {filename: expected_species_code}

        Returns:
            Labelling report dict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model(species_classes)

        # Collect images
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_paths.extend(Path(image_dir).rglob(ext))

        logger.info(f"Found {len(image_paths)} images to label")

        # Run batch prediction
        results = self._model.predict(
            source=[str(p) for p in image_paths],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        report = {
            "total_images": len(image_paths),
            "labelled": 0,
            "empty": 0,
            "disagreements": [],
            "per_class": {cls: 0 for cls in species_classes},
            "avg_confidence": 0.0,
        }
        all_confidences = []

        for img_path, result in zip(image_paths, results):
            boxes = result.boxes
            lines = []
            detected_species = []

            for i in range(len(boxes)):
                box = boxes.xywhn[i].cpu().numpy()
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])

                if cls_id < len(species_classes):
                    cx, cy, w, h = box
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    all_confidences.append(conf)
                    species_code = species_classes[cls_id]
                    detected_species.append(species_code)
                    report["per_class"][species_code] = report["per_class"].get(species_code, 0) + 1

            label_path = output_dir / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(lines))

            if lines:
                report["labelled"] += 1
            else:
                report["empty"] += 1

            # Cross-validate
            if existing_species_map and img_path.name in existing_species_map:
                expected = existing_species_map[img_path.name]
                if detected_species and expected not in detected_species:
                    report["disagreements"].append({
                        "image": img_path.name,
                        "expected": expected,
                        "detected": detected_species,
                    })

        if all_confidences:
            report["avg_confidence"] = sum(all_confidences) / len(all_confidences)

        logger.info(f"Labelling complete: {report['labelled']} labelled, "
                     f"{report['empty']} empty, "
                     f"{len(report['disagreements'])} disagreements")

        report_path = output_dir / "labelling_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


def build_species_map_from_dirs(raw_dir: Path) -> Dict[str, str]:
    """Build a filename -> species_code mapping from the folder structure.

    Walks raw_dir looking for species subdirectories containing images.
    E.g., raw/inaturalist/DEVIL/12345.jpg -> {"12345.jpg": "DEVIL"}

    Args:
        raw_dir: Root directory containing source/species/image.jpg structure

    Returns:
        Dict mapping image filename to species code
    """
    species_map: Dict[str, str] = {}

    for source_dir in raw_dir.iterdir():
        if not source_dir.is_dir():
            continue
        for species_dir in source_dir.iterdir():
            if not species_dir.is_dir():
                continue
            species_code = species_dir.name
            for img_path in species_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    species_map[img_path.name] = species_code

    return species_map
