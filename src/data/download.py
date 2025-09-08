#!/usr/bin/env python3
"""
Dataset downloader for public segmentation datasets.
Supports Kaggle datasets with automatic extraction and split creation.
Enhanced with KaggleHub support and grayscale conversion for C-SAM simulation.
"""

import os
import sys
import argparse
import yaml
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KaggleHubDownloader:
    """Download datasets using KaggleHub (newer, simpler API)."""

    def __init__(self, dataset_id: str, extract_path: str):
        self.dataset_id = dataset_id
        self.extract_path = Path(extract_path)
        self.extract_path.mkdir(parents=True, exist_ok=True)

    def _check_kagglehub(self) -> bool:
        """Check if KaggleHub is available."""
        try:
            import kagglehub

            return True
        except ImportError:
            logger.error("KaggleHub not found. Install with: pip install kagglehub")
            return False

    def download(self, force: bool = False) -> bool:
        """Download dataset using KaggleHub."""
        # Check if already downloaded
        if (
            not force
            and self.extract_path.exists()
            and any(self.extract_path.iterdir())
        ):
            logger.info(f"Dataset already exists at {self.extract_path}")
            return True

        if not self._check_kagglehub():
            return False

        try:
            import kagglehub

            logger.info(f"Downloading dataset via KaggleHub: {self.dataset_id}")

            # Download dataset - KaggleHub handles everything automatically
            download_path = kagglehub.dataset_download(self.dataset_id)
            logger.info(f"Dataset downloaded to: {download_path}")

            # Copy to our desired location
            download_source = Path(download_path)
            if download_source != self.extract_path:
                if self.extract_path.exists():
                    shutil.rmtree(self.extract_path)
                shutil.copytree(download_source, self.extract_path)
                logger.info(f"Dataset copied to: {self.extract_path}")

            return True

        except Exception as e:
            logger.error(f"Error downloading dataset via KaggleHub: {e}")
            return False


class KaggleDatasetDownloader:
    """Download and prepare Kaggle datasets with KaggleHub support."""

    def __init__(
        self, dataset_id: str, extract_path: str, force_grayscale: bool = False
    ):
        self.dataset_id = dataset_id
        self.extract_path = Path(extract_path)
        self.extract_path.mkdir(parents=True, exist_ok=True)
        self.force_grayscale = force_grayscale

    def _check_kagglehub_api(self) -> bool:
        """Check if KaggleHub is available."""
        try:
            import kagglehub

            return True
        except ImportError:
            logger.warning("KaggleHub not found. Falling back to regular Kaggle API.")
            return False

    def _check_kaggle_api(self) -> bool:
        """Check if Kaggle API is available and configured."""
        try:
            import kaggle

            # Test API by listing datasets (will fail if not authenticated)
            kaggle.api.dataset_list(search="test", page_size=1)
            return True
        except ImportError:
            logger.error("Kaggle package not found. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Kaggle API not configured: {e}")
            logger.error("Please set up Kaggle API credentials:")
            logger.error("1. Go to https://www.kaggle.com/settings/account")
            logger.error("2. Create API token")
            logger.error(
                "3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY"
            )
            return False

    def _download_with_kagglehub(self) -> bool:
        """Download using KaggleHub (preferred method)."""
        try:
            import kagglehub

            logger.info(f"Downloading with KaggleHub: {self.dataset_id}")

            # Download dataset
            download_path = kagglehub.dataset_download(self.dataset_id)
            download_path = Path(download_path)

            logger.info(f"Dataset downloaded to: {download_path}")

            # Move/copy files to our extract path
            if download_path != self.extract_path:
                if self.extract_path.exists():
                    shutil.rmtree(self.extract_path)

                shutil.copytree(download_path, self.extract_path)
                logger.info(f"Dataset moved to: {self.extract_path}")

            return True

        except Exception as e:
            logger.error(f"Error downloading with KaggleHub: {e}")
            return False

    def download(self, force: bool = False) -> bool:
        """Download dataset using KaggleHub or regular Kaggle API."""
        # Check if already downloaded
        if (
            not force
            and self.extract_path.exists()
            and any(self.extract_path.iterdir())
        ):
            logger.info(f"Dataset already exists at {self.extract_path}")
            return True

        # Try KaggleHub first (preferred)
        if self._check_kagglehub_api():
            success = self._download_with_kagglehub()
            if success:
                return self._post_process_dataset()

        # Fall back to regular Kaggle API
        if self._check_kaggle_api():
            success = self._download_with_kaggle_api()
            if success:
                return self._post_process_dataset()

        logger.error("Could not download dataset with either KaggleHub or Kaggle API")
        return False

    def _download_with_kaggle_api(self, force: bool = False) -> bool:
        """Download using regular Kaggle API (fallback method)."""
        if (
            not force
            and self.extract_path.exists()
            and any(self.extract_path.iterdir())
        ):
            logger.info(f"Dataset already exists at {self.extract_path}")
            return True

        if not self._check_kaggle_api():
            return False

        try:
            import kaggle

            logger.info(f"Downloading Kaggle dataset: {self.dataset_id}")

            # Create temporary download directory
            temp_dir = self.extract_path.parent / "temp_download"
            temp_dir.mkdir(exist_ok=True)

            # Download dataset
            kaggle.api.dataset_download_files(
                self.dataset_id, path=str(temp_dir), unzip=True
            )

            # Find downloaded files
            downloaded_files = list(temp_dir.rglob("*"))
            if not downloaded_files:
                logger.error("No files downloaded")
                return False

            # Move files to extract path
            for item in temp_dir.iterdir():
                dest = self.extract_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            # Cleanup temp directory
            shutil.rmtree(temp_dir)

            logger.info(f"Dataset downloaded and extracted to {self.extract_path}")
            return True

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False

    def _post_process_dataset(self) -> bool:
        """Post-process dataset including grayscale conversion if needed."""
        if not self.force_grayscale:
            return True

        try:
            import cv2
            import numpy as np

            logger.info(
                "Converting dataset to grayscale for SAM acoustic simulation..."
            )

            # Find all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            image_files = []

            for ext in image_extensions:
                image_files.extend(self.extract_path.rglob(f"*{ext}"))
                image_files.extend(self.extract_path.rglob(f"*{ext.upper()}"))

            converted_count = 0
            for img_path in image_files:
                try:
                    # Skip if already grayscale (check file size as heuristic)
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # Convert to grayscale if it's color
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Save as grayscale
                        cv2.imwrite(str(img_path), gray)
                        converted_count += 1

                except Exception as e:
                    logger.warning(f"Could not convert {img_path.name}: {e}")
                    continue

            logger.info(f"Converted {converted_count} images to grayscale")
            return True

        except ImportError:
            logger.warning("OpenCV not available - skipping grayscale conversion")
            return True
        except Exception as e:
            logger.error(f"Error during post-processing: {e}")
            return False


class DatasetSplitter:
    """Create train/val/test splits from downloaded dataset."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def convert_to_grayscale(self, data_dir: str, method: str = "luminance") -> bool:
        """
        Convert RGB images to grayscale for C-SAM simulation.

        Args:
            data_dir: Directory containing images
            method: Conversion method ('luminance', 'average', 'opencv')

        Returns:
            Success status
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("OpenCV and numpy required for grayscale conversion")
            return False

        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory {data_path} does not exist")
            return False

        # Find all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_path.rglob(ext))
            image_files.extend(data_path.rglob(ext.upper()))

        if not image_files:
            logger.warning("No image files found for grayscale conversion")
            return True

        converted_count = 0
        for img_path in image_files:
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Check if already grayscale
                if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                    continue

                # Convert based on method
                if method == "luminance":
                    # Weighted luminance conversion (standard)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif method == "average":
                    # Simple average
                    gray = np.mean(img, axis=2).astype(np.uint8)
                elif method == "opencv":
                    # OpenCV default
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    # Default to luminance
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Save as grayscale
                cv2.imwrite(str(img_path), gray)
                converted_count += 1

            except Exception as e:
                logger.warning(f"Could not convert {img_path.name}: {e}")
                continue

        logger.info(
            f"Converted {converted_count} images to grayscale using {method} method"
        )
        return True

    def create_splits(
        self, split_ratios: List[float] = [0.7, 0.15, 0.15], random_seed: int = 42
    ) -> bool:
        """
        Create train/val/test splits from flat dataset structure.

        Args:
            split_ratios: [train, val, test] ratios
            random_seed: Random seed for reproducibility

        Returns:
            Success status
        """
        import random
        import numpy as np

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Find image and mask directories
        image_dirs = []
        mask_dirs = []

        # Common naming patterns
        image_patterns = ["images", "image", "img", "train", "data"]
        mask_patterns = ["masks", "mask", "labels", "label", "gt", "groundtruth"]

        for item in self.data_dir.iterdir():
            if item.is_dir():
                item_name_lower = item.name.lower()

                if any(pattern in item_name_lower for pattern in image_patterns):
                    image_dirs.append(item)
                elif any(pattern in item_name_lower for pattern in mask_patterns):
                    mask_dirs.append(item)

        # If no clear structure, look for files directly in data_dir
        if not image_dirs:
            image_files = list(self.data_dir.glob("*.jpg")) + list(
                self.data_dir.glob("*.png")
            )
            if image_files:
                # Assume flat structure - separate images and masks by filename
                return self._create_splits_from_flat_structure(
                    split_ratios, random_seed
                )

        if not image_dirs or not mask_dirs:
            logger.error("Could not identify image and mask directories")
            logger.info(
                f"Found directories: {[d.name for d in self.data_dir.iterdir() if d.is_dir()]}"
            )
            return False

        # Use the first found directories
        image_dir = image_dirs[0]
        mask_dir = mask_dirs[0]

        logger.info(f"Using image directory: {image_dir.name}")
        logger.info(f"Using mask directory: {mask_dir.name}")

        return self._create_splits_from_directories(
            image_dir, mask_dir, split_ratios, random_seed
        )

    def _create_splits_from_flat_structure(
        self, split_ratios: List[float], random_seed: int
    ) -> bool:
        """Handle flat file structure where images and masks are mixed."""
        import random

        # Find all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
        all_files = []

        for ext in image_extensions:
            all_files.extend(self.data_dir.glob(f"*{ext}"))
            all_files.extend(self.data_dir.glob(f"*{ext.upper()}"))

        # Separate images and masks by naming convention
        images = []
        masks = []

        for file_path in all_files:
            name_lower = file_path.name.lower()
            if any(keyword in name_lower for keyword in ["mask", "label", "gt"]):
                masks.append(file_path)
            else:
                images.append(file_path)

        if not images or not masks:
            logger.error(
                f"Could not separate images and masks. Found {len(images)} images, {len(masks)} masks"
            )
            return False

        logger.info(f"Found {len(images)} images and {len(masks)} masks")

        # Match images to masks
        image_mask_pairs = []
        for img_path in images:
            # Try to find corresponding mask
            img_name = img_path.stem

            # Common mask naming patterns
            mask_candidates = [
                f"{img_name}_mask",
                f"{img_name}_label",
                f"{img_name}_gt",
                f"mask_{img_name}",
                f"label_{img_name}",
                img_name,
            ]

            mask_path = None
            for candidate in mask_candidates:
                for ext in image_extensions:
                    potential_mask = self.data_dir / f"{candidate}{ext}"
                    if potential_mask in masks:
                        mask_path = potential_mask
                        break
                if mask_path:
                    break

            if mask_path:
                image_mask_pairs.append((img_path, mask_path))

        if not image_mask_pairs:
            logger.error("Could not match images to masks")
            return False

        logger.info(f"Matched {len(image_mask_pairs)} image-mask pairs")

        # Create splits
        return self._split_and_organize_pairs(
            image_mask_pairs, split_ratios, random_seed
        )

    def _create_splits_from_directories(
        self,
        image_dir: Path,
        mask_dir: Path,
        split_ratios: List[float],
        random_seed: int,
    ) -> bool:
        """Create splits from separate image and mask directories."""
        import random

        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]:
            image_files.extend(image_dir.glob(ext))
            image_files.extend(image_dir.glob(ext.upper()))

        # Match images to masks
        image_mask_pairs = []
        for img_path in image_files:
            # Find corresponding mask
            mask_candidates = [
                mask_dir / f"{img_path.stem}.png",
                mask_dir / f"{img_path.stem}.jpg",
                mask_dir / f"{img_path.stem}_mask.png",
                mask_dir / f"{img_path.stem}_label.png",
            ]

            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path:
                image_mask_pairs.append((img_path, mask_path))

        if not image_mask_pairs:
            logger.error("Could not match images to masks")
            return False

        logger.info(f"Matched {len(image_mask_pairs)} image-mask pairs")

        return self._split_and_organize_pairs(
            image_mask_pairs, split_ratios, random_seed
        )

    def _split_and_organize_pairs(
        self,
        pairs: List[Tuple[Path, Path]],
        split_ratios: List[float],
        random_seed: int,
    ) -> bool:
        """Split pairs and organize into train/val/test structure."""
        import random

        random.shuffle(pairs)

        n_total = len(pairs)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val :]

        logger.info(
            f"Split dataset: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test"
        )

        # Create directory structure
        splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        for split_name, split_pairs in splits.items():
            if not split_pairs:
                continue

            split_dir = self.data_dir / split_name
            split_dir.mkdir(exist_ok=True)

            images_dir = split_dir / "images"
            masks_dir = split_dir / "masks"
            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)

            for img_path, mask_path in split_pairs:
                # Copy image
                dest_img = images_dir / img_path.name
                shutil.copy2(img_path, dest_img)

                # Copy mask
                dest_mask = masks_dir / mask_path.name
                shutil.copy2(mask_path, dest_mask)

        logger.info(f"Created split structure in {self.data_dir}")
        return True


def download_and_prepare_dataset(config_path: str) -> bool:
    """Download and prepare dataset based on config."""
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        download_config = config.get("download", {})
        if not download_config.get("enabled", False):
            logger.info("Dataset download disabled in config")
            return True

        data_dir = config["data"]["data_dir"]

        # Check if dataset already exists
        if Path(data_dir).exists() and any(Path(data_dir).iterdir()):
            if not download_config.get("auto_download", False):
                logger.info(f"Dataset directory {data_dir} already exists")
                return True

        # Download from KaggleHub (preferred) or Kaggle
        kagglehub_config = download_config.get("kagglehub", {})
        kaggle_config = download_config.get("kaggle", {})

        if kagglehub_config:
            # Use KaggleHub (simpler, newer API)
            dataset_id = kagglehub_config["dataset"]
            extract_path = kagglehub_config["extract_path"]

            downloader = KaggleHubDownloader(dataset_id, extract_path)
            if not downloader.download():
                return False

        elif kaggle_config:
            # Use traditional Kaggle API
            dataset_id = kaggle_config["dataset"]
            extract_path = kaggle_config["extract_path"]

            downloader = KaggleDatasetDownloader(dataset_id, extract_path)
            if not downloader.download():
                return False

        # Create splits if needed
        post_process = download_config.get("post_process", {})
        if post_process.get("create_splits", False):
            splitter = DatasetSplitter(data_dir)
            split_ratios = post_process.get("split_ratios", [0.7, 0.15, 0.15])
            random_seed = post_process.get("random_seed", 42)

            # Check if grayscale conversion is needed
            force_grayscale = post_process.get("force_grayscale", False)
            grayscale_method = post_process.get("grayscale_method", "luminance")

            if force_grayscale:
                logger.info("Converting images to grayscale for C-SAM simulation...")
                if not splitter.convert_to_grayscale(data_dir, grayscale_method):
                    logger.error("Failed to convert images to grayscale")
                    return False

            if not splitter.create_splits(split_ratios, random_seed):
                return False

        logger.info(f"Dataset successfully prepared at {data_dir}")
        return True

    except Exception as e:
        logger.error(f"Error downloading/preparing dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and prepare public datasets")
    parser.add_argument("config", help="Path to dataset configuration file")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force download even if dataset exists",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} does not exist")
        return 1

    success = download_and_prepare_dataset(args.config)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
