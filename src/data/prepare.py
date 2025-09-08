#!/usr/bin/env python3
"""
Dataset preparation utility with intelligent caching and statistics computation.
Automatically computes dataset normalization statistics and caches preprocessed data for faster training.
"""

import os
import sys
import argparse
import yaml
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import hashlib
from typing import Tuple, Dict, Any, Optional, List
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.transforms import compute_dataset_statistics


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Dataset preparation utility: always outputs COCO format in coco/ subfolder."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.coco_dir = self.data_dir / "coco"
        self.coco_dir.mkdir(exist_ok=True)

    def prepare_split(self, split: str, tile_size: int = 512) -> None:
        """Prepare a split and output COCO format in coco/ subfolder."""
        logger.info(f"Preparing {split} split in COCO format...")
        import random
        import datetime

        # Special logic for vehicle_damage_csam_proxy dataset: load from JSON, output COCO
        if self.data_dir.name == "vehicle_damage_csam_proxy":
            # Only run once for all splits
            if split != "train":
                return
            # Load both train and val JSONs
            json_files = [
                self.data_dir / "0Train_via_annos.json",
                self.data_dir / "0Val_via_annos.json",
            ]
            image_dir = self.data_dir / "image" / "image"
            all_annos = {}
            for jf in json_files:
                if jf.exists():
                    with open(jf, "r") as f:
                        all_annos.update(json.load(f))
            # Filter valid samples (with mask info and image exists)
            valid_samples = []
            for img_name, anno in all_annos.items():
                if not anno or "regions" not in anno or not anno["regions"]:
                    continue
                img_path = image_dir / img_name
                if not img_path.exists():
                    continue
                # Check if mask would be non-empty
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for region in anno["regions"]:
                    all_x = region.get("all_x")
                    all_y = region.get("all_y")
                    if (
                        all_x is None
                        or all_y is None
                        or len(all_x) < 3
                        or len(all_y) < 3
                    ):
                        continue
                    pts = np.array(list(zip(all_x, all_y)), dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                if np.sum(mask) == 0:
                    continue
                valid_samples.append(
                    (img_name, anno, np.any(mask > 0), image.shape[1], image.shape[0])
                )
            # Stratified split by mask presence
            random.seed(42)
            with_mask = [s for s in valid_samples if s[2]]
            without_mask = [s for s in valid_samples if not s[2]]

            def stratified_split(samples, val_ratio=0.15, test_ratio=0.15):
                n = len(samples)
                idxs = list(range(n))
                random.shuffle(idxs)
                n_val = int(n * val_ratio)
                n_test = int(n * test_ratio)
                val = [samples[i] for i in idxs[:n_val]]
                test = [samples[i] for i in idxs[n_val : n_val + n_test]]
                train = [samples[i] for i in idxs[n_val + n_test :]]
                return train, val, test

            train1, val1, test1 = stratified_split(with_mask)
            train2, val2, test2 = stratified_split(without_mask)
            train = train1 + train2
            val = val1 + val2
            test = test1 + test2
            random.shuffle(train)
            random.shuffle(val)
            random.shuffle(test)
            split_map = {"train": train, "val": val, "test": test}
            categories = [{"id": 1, "name": "damage", "supercategory": "defect"}]
            for split_name, split_samples in split_map.items():
                out_img_dir = self.coco_dir / split_name / "images"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                coco = {
                    "info": {
                        "description": "Vehicle Damage Segmentation",
                        "version": "1.0",
                        "date_created": str(datetime.datetime.now()),
                    },
                    "images": [],
                    "annotations": [],
                    "categories": categories,
                }
                ann_id = 1
                for img_idx, (img_name, anno, _, width, height) in enumerate(
                    tqdm(split_samples, desc=f"Writing {split_name}")
                ):
                    img_path = image_dir / img_name
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    # Save image
                    out_img = out_img_dir / img_name
                    cv2.imwrite(str(out_img), image)
                    # COCO image entry
                    coco["images"].append(
                        {
                            "id": img_idx + 1,
                            "file_name": img_name,
                            "width": width,
                            "height": height,
                        }
                    )
                    # COCO annotation(s)
                    for region in anno["regions"]:
                        all_x = region.get("all_x")
                        all_y = region.get("all_y")
                        if (
                            all_x is None
                            or all_y is None
                            or len(all_x) < 3
                            or len(all_y) < 3
                        ):
                            continue
                        segmentation = [
                            sum(
                                [[float(x), float(y)] for x, y in zip(all_x, all_y)], []
                            )
                        ]
                        x_min, x_max = min(all_x), max(all_x)
                        y_min, y_max = min(all_y), max(all_y)
                        bbox = [
                            float(x_min),
                            float(y_min),
                            float(x_max - x_min),
                            float(y_max - y_min),
                        ]
                        area = float(
                            cv2.contourArea(
                                np.array(list(zip(all_x, all_y)), dtype=np.int32)
                            )
                        )
                        coco["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": img_idx + 1,
                                "category_id": 1,
                                "segmentation": segmentation,
                                "area": area,
                                "bbox": bbox,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1
                ann_file = self.coco_dir / split_name / "annotations.json"
                ann_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ann_file, "w") as f:
                    json.dump(coco, f)
            logger.info(f"Vehicle dataset split and exported in COCO format.")
            return
        # Default logic for other datasets: convert to COCO format in coco/ subfolder
        import datetime

        categories = [{"id": 1, "name": "defect", "supercategory": "defect"}]
        image_dir = self.data_dir / split / "images"
        mask_dir = self.data_dir / split / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            logger.warning(f"Split {split} directories not found, skipping...")
            return
        out_img_dir = self.coco_dir / split / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {
                "description": f"Segmentation Dataset ({split})",
                "version": "1.0",
                "date_created": str(datetime.datetime.now()),
            },
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        ann_id = 1
        image_files = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        )
        for img_idx, img_file in enumerate(
            tqdm(image_files, desc=f"Processing {split}")
        ):
            mask_file = mask_dir / f"{img_file.stem}.png"
            if not mask_file.exists():
                mask_file = mask_dir / f"{img_file.stem}.jpg"
                if not mask_file.exists():
                    logger.warning(f"No mask found for {img_file.name}")
                    continue
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            out_img = out_img_dir / img_file.name
            cv2.imwrite(str(out_img), image)
            height, width = image.shape[:2]
            coco["images"].append(
                {
                    "id": img_idx + 1,
                    "file_name": img_file.name,
                    "width": width,
                    "height": height,
                }
            )
            # Find contours for each connected component in mask
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) < 3:
                    continue
                segmentation = [contour.flatten().astype(float).tolist()]
                x, y, w, h = cv2.boundingRect(contour)
                area = float(cv2.contourArea(contour))
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_idx + 1,
                        "category_id": 1,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        ann_file = self.coco_dir / split / "annotations.json"
        ann_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_file, "w") as f:
            json.dump(coco, f)
        logger.info(f"{split.capitalize()} split exported in COCO format.")

        # Default logic for other datasets: convert to COCO format in coco/ subfolder
        image_dir = self.data_dir / split / "images"
        mask_dir = self.data_dir / split / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            logger.warning(f"Split {split} directories not found, skipping...")
            return
        categories = [{"id": 1, "name": "defect", "supercategory": "defect"}]
        out_img_dir = self.coco_dir / split / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {
                "description": f"Segmentation Dataset ({split})",
                "version": "1.0",
                "date_created": str(datetime.datetime.now()),
            },
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        ann_id = 1
        image_files = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        )
        for img_idx, img_file in enumerate(
            tqdm(image_files, desc=f"Processing {split}")
        ):
            mask_file = mask_dir / f"{img_file.stem}.png"
            if not mask_file.exists():
                mask_file = mask_dir / f"{img_file.stem}.jpg"
                if not mask_file.exists():
                    logger.warning(f"No mask found for {img_file.name}")
                    continue
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            out_img = out_img_dir / img_file.name
            cv2.imwrite(str(out_img), image)
            height, width = image.shape[:2]
            coco["images"].append(
                {
                    "id": img_idx + 1,
                    "file_name": img_file.name,
                    "width": width,
                    "height": height,
                }
            )
            # Find contours for each connected component in mask
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) < 3:
                    continue
                segmentation = [contour.flatten().astype(float).tolist()]
                x, y, w, h = cv2.boundingRect(contour)
                area = float(cv2.contourArea(contour))
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_idx + 1,
                        "category_id": 1,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        ann_file = self.coco_dir / split / "annotations.json"
        ann_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_file, "w") as f:
            json.dump(coco, f)
        logger.info(f"{split.capitalize()} split exported in COCO format.")
        return

    def prepare_dataset(self, splits: List[str] = None) -> None:
        """Prepare all splits and output COCO format and stats in coco/ subfolder."""
        import numpy as np
        import datetime

        if splits is None:
            splits = ["train", "val", "test"]
        for split in splits:
            self.prepare_split(split)
        # Compute normalization statistics for train split images
        train_img_dir = self.coco_dir / "train" / "images"
        image_files = sorted(
            list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpg"))
        )
        if not image_files:
            logger.warning("No images found in train split for statistics.")
            return
        pixel_values = []
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sample_pixels = img.reshape(-1, 3)[::100]
            pixel_values.append(sample_pixels)
        if pixel_values:
            all_pixels = np.concatenate(pixel_values, axis=0)
            mean = np.mean(all_pixels, axis=0) / 255.0
            std = np.std(all_pixels, axis=0) / 255.0
            stats = {
                "normalization": {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "computed_from_data": True,
                    "samples_used": len(image_files),
                }
            }
            stats_file = self.coco_dir / "dataset_stats.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved normalization statistics to {stats_file}")
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info(f"COCO dataset location: {self.coco_dir}")

    def load_cached_data(self, split: str) -> Optional[Dict[str, Any]]:
        """Load cached data for a split."""
        cache_file = self.cache_dir / f"{split}_data.pkl"

        if not cache_file.exists() or not self._is_cache_valid(split):
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # Special logic for vehicle_damage_csam_proxy dataset: convert to COCO format
            if self.data_dir.name == "vehicle_damage_csam_proxy":
                import random
                from collections import defaultdict
                import datetime

                # Load both train and val JSONs
                json_files = [
                    self.data_dir / "0Train_via_annos.json",
                    self.data_dir / "0Val_via_annos.json",
                ]
                image_dir = self.data_dir / "image" / "image"
                all_annos = {}
                for jf in json_files:
                    if jf.exists():
                        with open(jf, "r") as f:
                            all_annos.update(json.load(f))
                # Filter valid samples (with mask info and image exists)
                valid_samples = []
                for img_name, anno in all_annos.items():
                    if not anno or "regions" not in anno or not anno["regions"]:
                        continue
                    img_path = image_dir / img_name
                    if not img_path.exists():
                        continue
                    # Check if mask would be non-empty
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    for region in anno["regions"]:
                        all_x = region.get("all_x")
                        all_y = region.get("all_y")
                        if (
                            all_x is None
                            or all_y is None
                            or len(all_x) < 3
                            or len(all_y) < 3
                        ):
                            continue
                        pts = np.array(list(zip(all_x, all_y)), dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 1)
                    if np.sum(mask) == 0:
                        continue
                    valid_samples.append(
                        (
                            img_name,
                            anno,
                            np.any(mask > 0),
                            image.shape[1],
                            image.shape[0],
                        )
                    )
                # Stratified split by mask presence
                random.seed(42)
                with_mask = [s for s in valid_samples if s[2]]
                without_mask = [s for s in valid_samples if not s[2]]

                def stratified_split(samples, val_ratio=0.15, test_ratio=0.15):
                    n = len(samples)
                    idxs = list(range(n))
                    random.shuffle(idxs)
                    n_val = int(n * val_ratio)
                    n_test = int(n * test_ratio)
                    val = [samples[i] for i in idxs[:n_val]]
                    test = [samples[i] for i in idxs[n_val : n_val + n_test]]
                    train = [samples[i] for i in idxs[n_val + n_test :]]
                    return train, val, test

                train1, val1, test1 = stratified_split(with_mask)
                train2, val2, test2 = stratified_split(without_mask)
                train = train1 + train2
                val = val1 + val2
                test = test1 + test2
                random.shuffle(train)
                random.shuffle(val)
                random.shuffle(test)
                split_map = {"train": train, "val": val, "test": test}
                # COCO categories (single class: damage=1)
                categories = [{"id": 1, "name": "damage", "supercategory": "defect"}]
                for split_name, split_samples in split_map.items():
                    out_img_dir = self.data_dir / split_name / "images"
                    out_img_dir.mkdir(parents=True, exist_ok=True)
                    coco = {
                        "info": {
                            "description": "Vehicle Damage Segmentation",
                            "version": "1.0",
                            "date_created": str(datetime.datetime.now()),
                        },
                        "images": [],
                        "annotations": [],
                        "categories": categories,
                    }
                    ann_id = 1
                    for img_idx, (img_name, anno, _, width, height) in enumerate(
                        tqdm(split_samples, desc=f"Writing {split_name}")
                    ):
                        img_path = image_dir / img_name
                        image = cv2.imread(str(img_path))
                        if image is None:
                            continue
                        # Save image
                        out_img = out_img_dir / img_name
                        cv2.imwrite(str(out_img), image)
                        # COCO image entry
                        coco["images"].append(
                            {
                                "id": img_idx + 1,
                                "file_name": img_name,
                                "width": width,
                                "height": height,
                            }
                        )
                        # COCO annotation(s)
                        for region in anno["regions"]:
                            all_x = region.get("all_x")
                            all_y = region.get("all_y")
                            if (
                                all_x is None
                                or all_y is None
                                or len(all_x) < 3
                                or len(all_y) < 3
                            ):
                                continue
                            segmentation = [
                                sum(
                                    [
                                        [float(x), float(y)]
                                        for x, y in zip(all_x, all_y)
                                    ],
                                    [],
                                )
                            ]
                            # Compute bbox
                            x_min, x_max = min(all_x), max(all_x)
                            y_min, y_max = min(all_y), max(all_y)
                            bbox = [
                                float(x_min),
                                float(y_min),
                                float(x_max - x_min),
                                float(y_max - y_min),
                            ]
                            area = float(
                                cv2.contourArea(
                                    np.array(list(zip(all_x, all_y)), dtype=np.int32)
                                )
                            )
                            coco["annotations"].append(
                                {
                                    "id": ann_id,
                                    "image_id": img_idx + 1,
                                    "category_id": 1,
                                    "segmentation": segmentation,
                                    "area": area,
                                    "bbox": bbox,
                                    "iscrowd": 0,
                                }
                            )
                            ann_id += 1
                    # Write COCO annotation file
                    ann_file = self.data_dir / split_name / "annotations.json"
                    with open(ann_file, "w") as f:
                        json.dump(coco, f)
                logger.info(f"Vehicle dataset split and exported in COCO format.")
                return {}
            logger.warning(f"Error loading cached data for {split}: {e}")
            return None


def update_config_with_stats(
    config_path: str, stats: Dict[str, Any], backup: bool = True
) -> bool:
    """Update configuration with computed statistics."""
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"Config file {config_path} does not exist")
        return False

    # Create backup
    if backup:
        backup_path = config_path.with_suffix(".yaml.backup")
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Update normalization stats
        if "normalization" in stats:
            norm_stats = stats["normalization"]

            # Update based on config structure
            if "datasets" in config:
                # Multi-dataset config - update all datasets
                for dataset_name in config["datasets"]:
                    if "normalization" not in config["datasets"][dataset_name]:
                        config["datasets"][dataset_name]["normalization"] = {}

                    config["datasets"][dataset_name]["normalization"].update(
                        {
                            "mean": norm_stats["mean"],
                            "std": norm_stats["std"],
                            "computed_from_data": True,
                        }
                    )
            else:
                # Single dataset config
                if "data" not in config:
                    config["data"] = {}
                if "normalization" not in config["data"]:
                    config["data"]["normalization"] = {}

                config["data"]["normalization"].update(
                    {
                        "mean": norm_stats["mean"],
                        "std": norm_stats["std"],
                        "computed_from_data": True,
                    }
                )

        # Save updated config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

        logger.info(f"Updated config: {config_path}")
        logger.info(f"Normalization - Mean: {stats['normalization']['mean']}")
        logger.info(f"Normalization - Std: {stats['normalization']['std']}")
        return True

    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset and output COCO format"
    )
    parser.add_argument("data_dir", help="Path to dataset directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return 1
    try:
        preparer = DatasetPreparer(str(data_dir))
        preparer.prepare_dataset(args.splits)
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
