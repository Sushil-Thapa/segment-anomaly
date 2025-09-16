"""
Dataset implementation for wafer defect and SAM acoustic microscopy segmentation with tiling support.
Includes MAE dataset for self-supervised pretraining.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import List, Tuple, Dict, Optional, Callable, Union
import glob
from pathlib import Path
import pickle
import math
import random
from .tiling import TileGenerator, ConfigurableTileGenerator, SAMAdaptiveTileGenerator
from .transforms import (
    get_train_transform,
    get_val_transform,
    get_sam_train_transform,
    get_sam_val_transform,
    WaferSpecificTransforms,
    SAMSpecificTransforms,
)


class DynamicOversamplingDataset(Dataset):
    """
    Dataset with dynamic oversampling that adapts as model improves.

    Key features:
    - 5x initial oversampling decaying to 3x
    - Dynamic adjustment based on model performance
    - Configurable stride for different modes
    - Dataset-specific normalization support
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        tile_size: int = 512,
        train_stride: int = 384,
        val_stride: int = 256,
        transform: Optional[Callable] = None,
        initial_oversample_ratio: float = 5.0,
        target_oversample_ratio: float = 3.0,
        oversample_decay_epochs: int = 50,
        cache_tiles: bool = True,
        precompute_tiles: bool = True,
        enable_dynamic_sampling: bool = True,
        dataset_format: str = "coco",
        config: Optional[dict] = None,
    ):
        """
        Initialize dynamic oversampling dataset.

        Args:
            data_root: Root directory containing images and masks
            split: Dataset split ('train', 'val', 'test')
        """
        # Store parameters
        self.data_root = Path(data_root)
        self.split = split
        self.tile_size = tile_size
        self.transform = transform
        self.initial_oversample_ratio = initial_oversample_ratio
        self.target_oversample_ratio = target_oversample_ratio
        self.oversample_decay_epochs = oversample_decay_epochs
        self.cache_tiles = cache_tiles
        self.enable_dynamic_sampling = enable_dynamic_sampling
        self.dataset_format = dataset_format.lower() if dataset_format else "coco"
        self.config = config or {}

        # Extract grayscale settings from config
        dataset_config = self.config.get("data", {}).get("dataset", {})
        self.force_grayscale = dataset_config.get("force_grayscale", False)
        self.grayscale_method = dataset_config.get("grayscale_method", "luminance")

        # Test/Debug mode settings
        debug_config = self.config.get("debug", {})
        self.debug_mode = debug_config.get("enabled", False)
        self.debug_sample_ratio = debug_config.get(
            "sample_ratio", 0.01
        )  # 1% by default
        self.debug_max_images = debug_config.get(
            "max_images", None
        )  # Optional absolute limit

        # Current training state
        self.current_epoch = 0
        self.current_oversample_ratio = initial_oversample_ratio
        self.model_performance_history = []

        # Initialize configurable tiler
        self.tiler = ConfigurableTileGenerator(
            tile_size=tile_size,
            train_stride=train_stride,
            val_stride=val_stride,
            inference_stride=train_stride,
            use_gaussian_weights=True,
        )

        # Initialize data paths
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # For detection/YOLO/VOC

        # Load data based on format
        if self.dataset_format == "coco":
            self._load_coco()
        elif self.dataset_format == "yolo":
            self._load_yolo()
        elif self.dataset_format == "voc":
            self._load_voc()
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_format}")

        # Apply debug/test mode sampling if enabled
        if self.debug_mode:
            self._apply_debug_sampling()

        # Precompute tile information
        self.tile_info = []
        self.positive_tiles = []
        self.negative_tiles = []
        self.tile_cache = {}
        if precompute_tiles:
            self._precompute_tiles()
        self._update_sample_weights()

    def _load_coco(self):
        """Load images and masks from COCO format."""
        coco_dir = self.data_root / "coco" / self.split
        ann_path = coco_dir / "annotations.json"
        with open(ann_path, "r") as f:
            coco = json.load(f)
        imgs = {img["id"]: img for img in coco["images"]}
        anns = coco["annotations"]
        img_to_anns = {}
        for ann in anns:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)
        self.coco_imgs = imgs
        self.coco_img_to_anns = img_to_anns
        self.coco_ids = list(imgs.keys())
        self.image_paths = [
            str(coco_dir / "images" / imgs[img_id]["file_name"])
            for img_id in self.coco_ids
        ]
        self.mask_paths = [img_id for img_id in self.coco_ids]  # Use img_id as mask ref
        self.coco_ann_path = ann_path
        self.coco_dir = coco_dir

    def _load_yolo(self):
        """Load images and labels from YOLO format (segmentation or detection)."""
        yolo_dir = self.data_root / "yolo" / self.split
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        for ext in image_extensions:
            self.image_paths.extend(sorted(glob.glob(str(images_dir / ext))))
        for img_path in self.image_paths:
            img_name = Path(img_path).stem
            label_path = labels_dir / f"{img_name}.txt"
            self.labels.append(str(label_path))

    def _load_voc(self):
        """Load images and masks from Pascal VOC format."""
        voc_dir = self.data_root / "voc" / self.split
        images_dir = voc_dir / "images"
        masks_dir = voc_dir / "masks"
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        for ext in image_extensions:
            self.image_paths.extend(sorted(glob.glob(str(images_dir / ext))))
        for img_path in self.image_paths:
            img_name = Path(img_path).stem
            mask_path = masks_dir / f"{img_name}.png"
            self.mask_paths.append(str(mask_path))

    def _load_file_paths(self):
        """Load image and mask file paths."""
        if self.split == "train":
            images_dir = self.data_root / "train" / "images"
            masks_dir = self.data_root / "train" / "masks"
        elif self.split == "val":
            images_dir = self.data_root / "val" / "images"
            masks_dir = self.data_root / "val" / "masks"
        else:
            images_dir = self.data_root / "test" / "images"
            masks_dir = self.data_root / "test" / "masks"

        # Find all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif"]

        for ext in image_extensions:
            pattern = str(images_dir / ext)
            self.image_paths.extend(glob.glob(pattern))

        # Find corresponding masks
        for image_path in self.image_paths:
            image_name = Path(image_path).stem

            # Try different mask extensions
            mask_extensions = ["png", "jpg", "jpeg", "tiff", "tif"]
            mask_path = None

            for mask_ext in mask_extensions:
                candidate = masks_dir / f"{image_name}.{mask_ext}"
                if candidate.exists():
                    mask_path = str(candidate)
                    break

            if mask_path is None:
                print(f"Warning: No mask found for {image_path}")
                continue

            self.mask_paths.append(mask_path)

        print(f"Loaded {len(self.image_paths)} image-mask pairs for {self.split} split")

    def _apply_debug_sampling(self):
        """Apply debug/test mode sampling to reduce dataset size for quick testing."""
        import random

        original_size = len(self.image_paths)

        if original_size == 0:
            print("⚠️  No images loaded, skipping debug sampling")
            return

        # Calculate sample size
        if self.debug_max_images is not None:
            sample_size = min(self.debug_max_images, original_size)
        else:
            sample_size = max(1, int(original_size * self.debug_sample_ratio))

        if sample_size >= original_size:
            print(
                f"🐛 Debug mode: Using all {original_size} images (sample size >= dataset size)"
            )
            return

        # Set seed for reproducible sampling
        random.seed(42)

        # Create indices and sample
        indices = list(range(original_size))
        sampled_indices = sorted(random.sample(indices, sample_size))

        # Sample all lists consistently
        self.image_paths = [self.image_paths[i] for i in sampled_indices]
        self.mask_paths = [self.mask_paths[i] for i in sampled_indices]
        if hasattr(self, "labels") and self.labels:
            self.labels = [self.labels[i] for i in sampled_indices]

        print(
            f"🐛 DEBUG MODE: Sampled {sample_size} images ({sample_size/original_size*100:.1f}%) from {original_size} total"
        )
        print(
            f"📊 Sample ratio: {self.debug_sample_ratio*100:.1f}% | Max images: {self.debug_max_images}"
        )
        print(
            f"⚡ This will significantly speed up training for testing pipeline issues!"
        )

    def _precompute_tiles(self):
        """Precompute all tiles with defect information."""
        total_images = len(self.image_paths)
        print(f"🔄 Precomputing tiles for {total_images} images...")
        mode = "train" if self.split == "train" else "val"
        for img_idx, image_path in enumerate(self.image_paths):
            # Show progress every 100 images with progress bar
            if (img_idx + 1) % 100 == 0 or img_idx == 0:
                progress_pct = ((img_idx + 1) / total_images) * 100
                bar_length = 30
                filled_length = int(bar_length * (img_idx + 1) // total_images)
                bar = "█" * filled_length + "-" * (bar_length - filled_length)
                print(
                    f"\r📊 Processing |{bar}| {progress_pct:5.1f}% [{img_idx + 1:4d}/{total_images:4d}]",
                    end="",
                    flush=True,
                )
                print(f"  Processed {img_idx + 1}/{len(self.image_paths)} images...")

            if self.dataset_format == "coco":
                img_id = self.coco_ids[img_idx]
                img_info = self.coco_imgs[img_id]
                height, width = img_info["height"], img_info["width"]
                mask = np.zeros((height, width), dtype=np.uint8)
                anns = self.coco_img_to_anns.get(img_id, [])
                for ann in anns:
                    for seg in ann["segmentation"]:
                        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                        pts = np.round(pts).astype(np.int32)
                        cv2.fillPoly(mask, [pts], color=ann["category_id"])
            elif self.dataset_format == "yolo":
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                mask = np.zeros((height, width), dtype=np.uint8)
                label_path = self.labels[img_idx]
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            cls = int(parts[0])
                            if len(parts) > 5:
                                pts = np.array(
                                    [float(x) for x in parts[1:]], dtype=np.float32
                                ).reshape(-1, 2)
                                pts[:, 0] *= width
                                pts[:, 1] *= height
                                pts = np.round(pts).astype(np.int32)
                                cv2.fillPoly(mask, [pts], color=cls)
                            else:
                                x_c, y_c, w, h = map(float, parts[1:5])
                                x_c *= width
                                y_c *= height
                                w *= width
                                h *= height
                                x1 = int(x_c - w / 2)
                                y1 = int(y_c - h / 2)
                                x2 = int(x_c + w / 2)
                                y2 = int(y_c + h / 2)
                                cv2.rectangle(
                                    mask, (x1, y1), (x2, y2), color=cls, thickness=-1
                                )
            elif self.dataset_format == "voc":
                mask_path = self.mask_paths[img_idx]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                height, width = mask.shape
            else:
                raise ValueError(f"Unsupported dataset format: {self.dataset_format}")

            tile_coords = self.tiler.get_tile_indices(height, width, mode=mode)
            for tile_coord in tile_coords:
                start_y, end_y, start_x, end_x = tile_coord
                mask_tile = mask[start_y:end_y, start_x:end_x]
                if mask_tile.shape != (self.tile_size, self.tile_size):
                    mask_tile = cv2.resize(
                        mask_tile,
                        (self.tile_size, self.tile_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                has_defect = bool(np.any(mask_tile > 0))  # Convert to Python bool
                defect_density = np.sum(mask_tile > 0) / (
                    self.tile_size * self.tile_size
                )
                tile_info = {
                    "image_idx": img_idx,
                    "coords": tile_coord,
                    "has_defect": has_defect,
                    "defect_density": defect_density,
                }
                tile_idx = len(self.tile_info)
                self.tile_info.append(tile_info)
                if has_defect:
                    self.positive_tiles.append(tile_idx)
                else:
                    self.negative_tiles.append(tile_idx)
        print(
            f"\n✅ Precomputed {len(self.tile_info)} tiles: "
            f"{len(self.positive_tiles)} positive, {len(self.negative_tiles)} negative"
        )
        if len(self.tile_info) > 0:
            print(
                f"📈 Positive ratio: {len(self.positive_tiles) / len(self.tile_info):.4f}"
            )
        else:
            print("❌ No tiles found.")

    def update_epoch(self, epoch: int, performance_metrics: Optional[Dict] = None):
        """
        Update current epoch and adapt sampling strategy.

        Args:
            epoch: Current training epoch
            performance_metrics: Dict containing model performance metrics
        """
        self.current_epoch = epoch

        # Store performance history for dynamic adaptation
        if performance_metrics is not None:
            self.model_performance_history.append(performance_metrics)

        # Update oversample ratio with decay
        if self.oversample_decay_epochs > 0:
            decay_factor = min(1.0, epoch / self.oversample_decay_epochs)
            self.current_oversample_ratio = (
                self.initial_oversample_ratio
                - decay_factor
                * (self.initial_oversample_ratio - self.target_oversample_ratio)
            )

        # Dynamic adaptation based on recent performance
        if self.enable_dynamic_sampling and len(self.model_performance_history) > 5:
            self._adapt_sampling_strategy()

        # Update sample weights
        self._update_sample_weights()

        print(
            f"Epoch {epoch}: Updated oversample ratio to {self.current_oversample_ratio:.2f}"
        )

    def _adapt_sampling_strategy(self):
        """Adapt sampling strategy based on model performance."""
        if len(self.model_performance_history) < 5:
            return

        # Get recent performance metrics
        recent_metrics = self.model_performance_history[-5:]

        # Check if false positive rate is high (adjust as needed)
        if "precision" in recent_metrics[-1]:
            recent_precision = [m.get("precision", 0.5) for m in recent_metrics]
            avg_precision = np.mean(recent_precision)

            # If precision is low (high FP rate), increase negative sampling
            if avg_precision < 0.7:
                self.current_oversample_ratio = min(
                    self.current_oversample_ratio * 1.1, 8.0
                )
                print(
                    f"Low precision detected ({avg_precision:.3f}), increasing oversample ratio"
                )
            elif avg_precision > 0.9:
                # High precision, can reduce oversampling
                self.current_oversample_ratio = max(
                    self.current_oversample_ratio * 0.95, self.target_oversample_ratio
                )
                print(
                    f"High precision detected ({avg_precision:.3f}), reducing oversample ratio"
                )

    def _update_sample_weights(self):
        """Update sample weights based on current oversampling strategy."""
        if self.split != "train" or len(self.positive_tiles) == 0:
            self.sample_weights = None
            return

        # Calculate weights for balanced sampling
        num_positive = len(self.positive_tiles)
        num_negative = len(self.negative_tiles)
        total_samples = len(self.tile_info)

        # Target samples per class based on oversampling ratio
        target_positive_samples = int(
            total_samples / (1 + self.current_oversample_ratio)
        )
        target_negative_samples = int(
            target_positive_samples * self.current_oversample_ratio
        )

        # Calculate sample weights
        weights = np.zeros(total_samples)

        # Weight positive samples
        pos_weight = target_positive_samples / num_positive if num_positive > 0 else 0
        for idx in self.positive_tiles:
            weights[idx] = pos_weight

        # Weight negative samples
        neg_weight = target_negative_samples / num_negative if num_negative > 0 else 0
        for idx in self.negative_tiles:
            weights[idx] = neg_weight

        self.sample_weights = torch.from_numpy(weights).float()

        print(
            f"Updated sample weights: pos_weight={pos_weight:.4f}, neg_weight={neg_weight:.4f}"
        )

    def get_weighted_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get weighted random sampler for dynamic oversampling."""
        if self.sample_weights is None:
            return None

        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.tile_info)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tile sample, loading images/masks/labels according to dataset_format.
        """
        if self.cache_tiles and idx in self.tile_cache:
            return self.tile_cache[idx]

        tile_info = self.tile_info[idx]
        image_idx = tile_info["image_idx"]
        coords = tile_info["coords"]

        if self.dataset_format == "coco":
            # Load image
            img_id = self.coco_ids[image_idx]
            img_info = self.coco_imgs[img_id]
            img_path = self.coco_dir / "images" / img_info["file_name"]
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale if required
            if self.force_grayscale:
                if self.grayscale_method == "luminance":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    # Keep as single channel since model expects 1 channel
                elif self.grayscale_method == "average":
                    image = np.mean(image, axis=2).astype(np.uint8)

            height, width = img_info["height"], img_info["width"]
            # Build mask from polygons
            mask = np.zeros((height, width), dtype=np.uint8)
            anns = self.coco_img_to_anns.get(img_id, [])
            for ann in anns:
                for seg in ann["segmentation"]:
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=ann["category_id"])
        elif self.dataset_format == "yolo":
            image_path = self.image_paths[image_idx]
            label_path = self.labels[image_idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale if required
            if self.force_grayscale:
                if self.grayscale_method == "luminance":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    # Keep as single channel since model expects 1 channel
                elif self.grayscale_method == "average":
                    image = np.mean(image, axis=2).astype(np.uint8)

            # Build mask from YOLO segmentation/detection txt
            if len(image.shape) == 2:  # Grayscale
                height, width = image.shape
            else:  # RGB
                height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue  # Not enough info
                        cls = int(parts[0])
                        # If segmentation: x1 y1 x2 y2 ...
                        if len(parts) > 5:
                            pts = np.array(
                                [float(x) for x in parts[1:]], dtype=np.float32
                            ).reshape(-1, 2)
                            pts[:, 0] *= width
                            pts[:, 1] *= height
                            pts = np.round(pts).astype(np.int32)
                            cv2.fillPoly(mask, [pts], color=cls)
                        else:
                            # Detection: x_center y_center w h (normalized)
                            x_c, y_c, w, h = map(float, parts[1:5])
                            x_c *= width
                            y_c *= height
                            w *= width
                            h *= height
                            x1 = int(x_c - w / 2)
                            y1 = int(y_c - h / 2)
                            x2 = int(x_c + w / 2)
                            y2 = int(y_c + h / 2)
                            cv2.rectangle(
                                mask, (x1, y1), (x2, y2), color=cls, thickness=-1
                            )
        elif self.dataset_format == "voc":
            image_path = self.image_paths[image_idx]
            mask_path = self.mask_paths[image_idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale if required
            if self.force_grayscale:
                if self.grayscale_method == "luminance":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif self.grayscale_method == "average":
                    image = np.mean(image, axis=2).astype(np.uint8)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_format}")

        # Extract tile
        start_y, end_y, start_x, end_x = coords
        image_tile = image[start_y:end_y, start_x:end_x]
        mask_tile = mask[start_y:end_y, start_x:end_x]

        # Resize if needed (for edge tiles)
        if image_tile.shape[:2] != (self.tile_size, self.tile_size):
            image_tile = cv2.resize(image_tile, (self.tile_size, self.tile_size))
            mask_tile = cv2.resize(
                mask_tile,
                (self.tile_size, self.tile_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image_tile, mask=mask_tile)
            image_tile = transformed["image"]
            mask_tile = transformed["mask"]
        else:
            # Handle single channel vs multi-channel images
            if len(image_tile.shape) == 2:  # Grayscale
                image_tile = (
                    torch.from_numpy(image_tile).float().unsqueeze(0) / 255.0
                )  # Add channel dimension
            else:  # RGB
                image_tile = (
                    torch.from_numpy(image_tile.transpose(2, 0, 1)).float() / 255.0
                )
            mask_tile = torch.from_numpy(mask_tile).long()

        sample = {
            "image": image_tile,
            "mask": mask_tile,
            "has_defect": bool(
                tile_info["has_defect"]
            ),  # Convert numpy.bool_ to Python bool
            "defect_density": tile_info["defect_density"],
            "image_idx": image_idx,
            "coords": coords,
        }

        if self.cache_tiles:
            self.tile_cache[idx] = sample

        return sample


# Backward compatibility alias
class WaferTileDataset(DynamicOversamplingDataset):
    """Backward compatibility alias for DynamicOversamplingDataset."""

    pass


class SAMAcousticDataset(DynamicOversamplingDataset):
    """
    Dataset for SAM acoustic microscopy defect segmentation.

    Specialized features:
    - Adaptive tiling with 50-75% overlap
    - Grayscale image support with RGB conversion
    - Speckle and acoustic noise simulation
    - Defect-aware tile positioning
    - Copy-paste augmentation with acoustic defect synthesis
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        tile_size: int = 512,
        overlap_range: Tuple[float, float] = (0.5, 0.75),
        transform: Optional[Callable] = None,
        precompute_tiles: bool = False,
        cache_dir: Optional[str] = None,
        initial_positive_ratio: float = 0.75,
        final_positive_ratio: float = 0.5,
        ratio_decay_epochs: int = 50,
        hard_negative_mining: bool = True,
        defect_synthesis: bool = True,
        defect_image_dir: Optional[str] = None,
    ):
        """
        Initialize SAM acoustic microscopy dataset.

        Args:
            data_root: Root directory containing images and masks
            split: Dataset split ('train', 'val', 'test')
            tile_size: Size of tiles to extract
            overlap_range: Range of overlap percentages for adaptive tiling
            transform: Transform to apply to samples
            precompute_tiles: Whether to precompute all tiles
            cache_dir: Directory to cache computed tiles
            initial_positive_ratio: Initial oversampling ratio for positive samples
            final_positive_ratio: Final oversampling ratio for positive samples
            ratio_decay_epochs: Number of epochs over which to decay ratio
            hard_negative_mining: Whether to use hard negative mining
            defect_synthesis: Whether to enable defect synthesis
            defect_image_dir: Directory containing defect images for copy-paste
        """
        # Initialize base class with dynamic oversampling
        super().__init__(
            data_root=data_root,
            split=split,
            tile_size=tile_size,
            stride=int(tile_size * (1 - max(overlap_range))),  # Conservative stride
            transform=transform,
            precompute_tiles=precompute_tiles,
            cache_dir=cache_dir,
            initial_positive_ratio=initial_positive_ratio,
            final_positive_ratio=final_positive_ratio,
            ratio_decay_epochs=ratio_decay_epochs,
            hard_negative_mining=hard_negative_mining,
        )

        self.overlap_range = overlap_range
        self.defect_synthesis = defect_synthesis
        self.defect_image_dir = defect_image_dir

        # Initialize SAM-specific components
        self.sam_tiler = SAMAdaptiveTileGenerator(
            tile_size=tile_size, overlap_range=overlap_range, defect_avoidance=True
        )

        # Load defect images for synthesis if available
        self.defect_images = []
        if defect_synthesis and defect_image_dir and os.path.exists(defect_image_dir):
            self._load_defect_images()

        # Initialize SAM-specific transforms if none provided
        if transform is None:
            self._setup_sam_transforms()

    def _load_defect_images(self):
        """Load defect images for copy-paste synthesis."""
        defect_files = glob.glob(
            os.path.join(self.defect_image_dir, "*.png")
        ) + glob.glob(os.path.join(self.defect_image_dir, "*.jpg"))

        for defect_file in defect_files[:50]:  # Limit to 50 defect images
            try:
                img = cv2.imread(defect_file)
                if img is not None:
                    # Convert to grayscale and then back to RGB for consistency
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    self.defect_images.append(rgb_img)
            except Exception as e:
                print(f"Warning: Could not load defect image {defect_file}: {e}")

        print(f"Loaded {len(self.defect_images)} defect images for synthesis")

    def _setup_sam_transforms(self):
        """Setup SAM-specific transforms."""
        sam_factory = SAMSpecificTransforms(
            data_dir=self.data_root, defect_images=self.defect_images
        )

        if self.split == "train":
            self.transform = sam_factory.get_train_transform(self.tile_size)
        else:
            self.transform = sam_factory.get_val_transform(self.tile_size)

    def load_image_and_mask(
        self, image_path: str, mask_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and mask with SAM-specific preprocessing.

        Args:
            image_path: Path to image file
            mask_path: Path to mask file

        Returns:
            Tuple of (image, mask) arrays
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale if it's not already (SAM typically uses grayscale)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # For consistency with transform pipeline, we'll let GrayscaleToRGB handle conversion
        image = gray

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Create empty mask if not found
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Ensure mask is binary
        mask = (mask > 128).astype(np.uint8)

        return image, mask

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for SAM dataset."""
        if hasattr(self, "tiles") and self.tiles:
            defect_count = sum(
                1 for tile in self.tiles if tile.get("has_defect", False)
            )
            background_count = len(self.tiles) - defect_count
        else:
            # Fallback to base class method
            base_dist = super().get_class_distribution()
            return {
                "background": base_dist.get("negative", 0),
                "acoustic_defect": base_dist.get("positive", 0),
            }

        return {"background": background_count, "acoustic_defect": defect_count}

    def update_sampling_strategy(
        self, epoch: int, performance_metrics: Optional[Dict] = None
    ):
        """
        Update sampling strategy for SAM with acoustic-specific adaptations.

        Args:
            epoch: Current training epoch
            performance_metrics: Optional performance metrics for adaptive adjustment
        """
        # Call parent method for base functionality
        super().update_sampling_strategy(epoch, performance_metrics)

        # SAM-specific adjustments based on acoustic defect detection performance
        if performance_metrics and "precision" in performance_metrics:
            precision = performance_metrics["precision"]

            # If precision is low, increase positive sampling to find more defects
            if precision < 0.7:
                adjustment_factor = 1.2
            # If precision is very high, we can be more selective
            elif precision > 0.9:
                adjustment_factor = 0.8
            else:
                adjustment_factor = 1.0

            self.current_positive_ratio = min(
                0.8, self.current_positive_ratio * adjustment_factor
            )


def create_dataloaders(
    config: dict, num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = get_train_transform(tile_size=config["data"]["tile_size"])
    val_transform = get_val_transform(tile_size=config["data"]["tile_size"])

    # Create datasets
    train_dataset = WaferTileDataset(
        data_root=config["data"]["root"],
        split="train",
        tile_size=config["data"]["tile_size"],
        stride=config["data"]["stride"],
        transform=train_transform,
        oversample_ratio=config["data"]["oversample_ratio"],
    )

    val_dataset = WaferTileDataset(
        data_root=config["data"]["root"],
        split="val",
        tile_size=config["data"]["tile_size"],
        stride=config["data"]["stride"],
        transform=val_transform,
        oversample_ratio=1.0,  # No oversampling for validation
    )

    test_dataset = WaferTileDataset(
        data_root=config["data"]["root"],
        split="test",
        tile_size=config["data"]["tile_size"],
        stride=config["data"]["stride"],
        transform=val_transform,
        oversample_ratio=1.0,  # No oversampling for test
    )

    # Create samplers
    train_sampler = train_dataset.get_sampler()

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"],
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def test_dataset():
    """Test dataset functionality."""
    # Create dummy data structure
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure
        for split in ["train", "val", "test"]:
            (temp_path / "images" / split).mkdir(parents=True)
            (temp_path / "masks" / split).mkdir(parents=True)

            # Create dummy images and masks
            for i in range(3):
                # Create dummy image
                image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
                cv2.imwrite(str(temp_path / "images" / split / f"image_{i}.png"), image)

                # Create dummy mask
                mask = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8) * 255
                cv2.imwrite(str(temp_path / "masks" / split / f"image_{i}.png"), mask)

        # Test dataset
        dataset = WaferTileDataset(
            data_root=str(temp_path),
            split="train",
            tile_size=512,
            stride=256,
            precompute_tiles=True,
        )

        print(f"Dataset length: {len(dataset)}")
        print(f"Class distribution: {dataset.get_class_distribution()}")

        # Test sample loading
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")

        return True


class MAEPretrainingDataset(Dataset):
    """
    Dataset for MAE self-supervised pretraining on unlabeled images.

    Key features:
    - Loads unlabeled images without masks
    - Supports millions of images for SSL pretraining
    - Minimal augmentation (just normalization + light intensity jitter)
    - Efficient loading with caching support
    - Compatible with existing dataset structure
    """

    def __init__(
        self,
        data_root: str,
        image_dirs: List[str] = None,
        tile_size: int = 384,  # Match Swin input size
        transform: Optional[Callable] = None,
        cache_tiles: bool = False,  # Usually too many for caching
        max_images: Optional[int] = None,
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".tif"],
        in_channels: int = 3,
        debug_mode: bool = False,
        debug_sample_ratio: float = 0.001,  # 0.1% for debug
        **kwargs,
    ):
        """
        Initialize MAE pretraining dataset.

        Args:
            data_root: Root directory containing unlabeled images
            image_dirs: List of subdirectories to search
            tile_size: Size of image tiles for training
            transform: Image transformations to apply
            cache_tiles: Whether to cache processed tiles
            max_images: Maximum number of images to use
            extensions: Valid image file extensions
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            debug_mode: Enable debug mode with small sample
            debug_sample_ratio: Ratio of full dataset to use in debug mode
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.tile_size = tile_size
        self.in_channels = in_channels
        self.debug_mode = debug_mode
        self.cache_tiles = cache_tiles

        # Discover all images
        self.image_paths = self._discover_images(image_dirs, extensions, max_images)

        # Apply debug sampling if needed
        if debug_mode:
            original_count = len(self.image_paths)
            sample_count = max(1, int(original_count * debug_sample_ratio))
            self.image_paths = random.sample(self.image_paths, sample_count)
            print(f"Debug mode: Using {len(self.image_paths)}/{original_count} images")

        # Set up transforms
        if transform is None:
            from .transforms import get_mae_transform

            self.transform = get_mae_transform(tile_size, in_channels)
        else:
            self.transform = transform

        # Optional tile cache
        self.tile_cache = {} if cache_tiles else None

        print(f"MAE dataset initialized with {len(self.image_paths)} images")
        print(f"Sample image paths: {self.image_paths[:3]}")

    def _discover_images(
        self,
        image_dirs: Optional[List[str]],
        extensions: List[str],
        max_images: Optional[int],
    ) -> List[Path]:
        """Discover all image files in the data directories."""
        image_paths = []

        # Determine search directories
        if image_dirs is None:
            # Search common directory patterns
            search_dirs = ["images", "image", "data", "train", "val", "unlabeled"]
            search_dirs = [d for d in search_dirs if (self.data_root / d).exists()]
            if not search_dirs:
                search_dirs = ["."]  # Search root directory
        else:
            search_dirs = image_dirs

        # Search for images
        for dir_name in search_dirs:
            dir_path = self.data_root / dir_name

            if not dir_path.exists():
                print(f"Warning: Directory {dir_path} does not exist")
                continue

            # Search for images with all extensions
            for ext in extensions:
                patterns = [
                    f"**/*{ext}",
                    f"**/*{ext.upper()}",
                ]

                for pattern in patterns:
                    found_paths = list(dir_path.glob(pattern))
                    image_paths.extend(found_paths)

        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))

        # Apply max_images limit
        if max_images is not None and len(image_paths) > max_images:
            image_paths = image_paths[:max_images]

        if not image_paths:
            raise ValueError(
                f"No images found in {self.data_root} with directories {search_dirs}"
            )

        return image_paths

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample for MAE pretraining.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'image' tensor
        """
        # Check cache first
        if self.tile_cache is not None and idx in self.tile_cache:
            return self.tile_cache[idx]

        # Load image
        image_path = self.image_paths[idx]

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Handle grayscale conversion if needed
            if self.in_channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=-1)

            # Resize to target size
            if image.shape[:2] != (self.tile_size, self.tile_size):
                image = cv2.resize(
                    image,
                    (self.tile_size, self.tile_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Apply transforms
            if self.transform:
                # Ensure image is uint8
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)

                # Apply albumentations transform
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                # Default normalization
                image = image.astype(np.float32) / 255.0

                # Convert to torch tensor [C, H, W]
                if len(image.shape) == 3:
                    image = torch.from_numpy(image).permute(2, 0, 1)
                else:
                    image = torch.from_numpy(image).unsqueeze(0)

            sample = {"image": image}

            # Cache if enabled
            if self.tile_cache is not None:
                self.tile_cache[idx] = sample

            return sample

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(self.in_channels, self.tile_size, self.tile_size)
            return {"image": dummy_image}

    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample."""
        image_path = self.image_paths[idx]

        # Try to load image to get dimensions
        try:
            image = cv2.imread(str(image_path))
            if image is not None:
                h, w = image.shape[:2]
            else:
                h, w = 0, 0
        except Exception:
            h, w = 0, 0

        return {
            "index": idx,
            "image_path": str(image_path),
            "original_size": (h, w),
            "tile_size": (self.tile_size, self.tile_size),
        }

    def clear_cache(self):
        """Clear the tile cache to free memory."""
        if self.tile_cache is not None:
            self.tile_cache.clear()
            print("Tile cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        cache_size = 0
        if self.tile_cache is not None:
            cache_size = len(self.tile_cache)

        return {
            "dataset_size": len(self.image_paths),
            "cached_samples": cache_size,
            "cache_ratio": (
                cache_size / len(self.image_paths) if len(self.image_paths) > 0 else 0.0
            ),
        }


def create_mae_dataset(config: dict, debug_mode: bool = False) -> MAEPretrainingDataset:
    """
    Create MAE dataset from configuration.

    Args:
        config: Configuration dictionary
        debug_mode: Enable debug mode for testing

    Returns:
        MAEPretrainingDataset instance
    """
    data_config = config.get("data", {})
    mae_config = data_config.get("mae", {})

    # Get data root
    data_root = mae_config.get("data_root", "data/mae_pretraining")
    if not os.path.exists(data_root):
        # Try alternative locations
        alternative_roots = [
            "data/unlabeled",
            "data/train",
            "data",
            os.path.join(data_config.get("data_root", "data"), "images"),
        ]

        for alt_root in alternative_roots:
            if os.path.exists(alt_root):
                data_root = alt_root
                break
        else:
            raise ValueError(
                f"MAE data root not found. Tried: {[data_root] + alternative_roots}"
            )

    # Create dataset
    dataset = MAEPretrainingDataset(
        data_root=data_root,
        image_dirs=mae_config.get("image_dirs", None),
        tile_size=mae_config.get("tile_size", 384),
        max_images=mae_config.get("max_images", None),
        in_channels=config.get("model", {}).get("in_channels", 3),
        debug_mode=debug_mode,
        debug_sample_ratio=mae_config.get("debug_sample_ratio", 0.001),
        cache_tiles=mae_config.get("cache_tiles", False),
    )

    return dataset


def test_mae_dataset():
    """Test MAE dataset functionality."""
    # Create a dummy data directory structure for testing
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy images
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir)

        # Create some dummy images
        for i in range(5):
            dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(images_dir, f"image_{i:03d}.jpg"), dummy_image)

        # Test dataset creation
        dataset = MAEPretrainingDataset(
            data_root=temp_dir,
            tile_size=384,
            debug_mode=True,
            debug_sample_ratio=1.0,  # Use all samples in test
        )

        print(f"MAE dataset size: {len(dataset)}")

        # Test sample loading
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Image dtype: {sample['image'].dtype}")
        print(
            f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]"
        )

        # Test memory usage
        memory_usage = dataset.get_memory_usage()
        print(f"Memory usage: {memory_usage}")

        # Test sample info
        info = dataset.get_sample_info(0)
        print(f"Sample info: {info}")

        return True


if __name__ == "__main__":
    test_dataset()
    test_mae_dataset()
    print("Dataset tests passed!")
