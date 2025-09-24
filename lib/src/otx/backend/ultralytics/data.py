"""Ultralytics dataloading integration for OTX.

This module defines an `OTXUltralyticsDetectionDataset` that emits batches compatible
with Ultralytics YOLO v8 detection trainers, and a corresponding data module that
provides infinite dataloaders for train/val/test.
"""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import torch
from datumaro import Dataset as DmDataset
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms.v2 import Normalize
from ultralytics.data.build import InfiniteDataLoader

from otx.config.data import TileConfig
from otx.data.dataset import OTXDetectionDataset
from otx.data.dataset.base import OTXDataset
from otx.data.dataset.tile import OTXTileDatasetFactory
from otx.data.entity import OTXDataItem
from otx.data.factory import TransformLibFactory
from otx.data.module import OTXDataModule
from otx.data.utils import adapt_input_size_to_dataset, adapt_tile_config, get_adaptive_num_workers
from otx.data.utils.pre_filtering import pre_filtering
from otx.types.device import DeviceType
from otx.types.image import ImageColorChannel
from otx.types.label import LabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from otx.config.data import SubsetConfig


class OTXUltralyticsDetectionDataset(OTXDetectionDataset):
    """Ultralytics Detection Dataset."""

    def collate_fn(self, batch: list[OTXDataItem]) -> dict[str, Any]:
        """Collate a list of samples into Ultralytics-compatible batch dict."""
        images = torch.stack([item.image for item in batch])
        bboxes = []
        classes = []
        batch_indices = []
        ori_shapes = [item.img_info.ori_shape for item in batch]
        ratio_pads = [((1.0, 1.0), (0.0, 0.0)) for _ in batch]

        for i, item in enumerate(batch):
            n = item.bboxes.shape[0]

            # Always create batch indices and tensors, even for empty annotations
            batch_indices.append(torch.full((n,), i, dtype=torch.long))

            if n == 0:
                # For images with no annotations, create empty tensors with proper shape
                b_norm_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
                cls = torch.zeros((0,), dtype=torch.long)
            else:
                if isinstance(item.bboxes, tv_tensors.BoundingBoxes):
                    # Convert to top-left xywh then to normalized center-based xywh
                    fmt = getattr(item.bboxes, "format", "XYWH")
                    in_fmt = fmt.lower() if isinstance(fmt, str) else "xywh"
                    b_tlwh = (
                        box_convert(item.bboxes, in_fmt=in_fmt, out_fmt="xywh")
                        if item.bboxes.numel() > 0
                        else torch.zeros((0, 4))
                    )
                else:
                    b_tlwh = item.bboxes if item.bboxes is not None else torch.zeros((0, 4), dtype=torch.float32)

                if b_tlwh.numel() > 0:
                    tl_x = b_tlwh[:, 0]
                    tl_y = b_tlwh[:, 1]
                    w = b_tlwh[:, 2]
                    h = b_tlwh[:, 3]
                    _, height, width = item.image.shape  # CHW
                    cx = tl_x + w / 2.0
                    cy = tl_y + h / 2.0
                    b_norm_cxcywh = torch.stack((cx / width, cy / height, w / width, h / height), dim=1).to(
                        torch.float32
                    )
                else:
                    b_norm_cxcywh = torch.zeros((0, 4), dtype=torch.float32)

                cls = (
                    item.label.to(torch.long).view(-1)
                    if item.label is not None and item.label.numel() > 0
                    else torch.zeros((0,), dtype=torch.long)
                )

            bboxes.append(b_norm_cxcywh)  # (n,4) normalized cxcywh
            classes.append(cls)  # (n,) long
        bboxes = torch.cat(bboxes, dim=0) if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        classes = torch.cat(classes, dim=0) if len(classes) > 0 else torch.zeros((0,), dtype=torch.long)
        batch_idx = torch.cat(batch_indices, dim=0) if len(batch_indices) > 0 else torch.zeros((0,), dtype=torch.long)

        # Generate file names for each image in the batch
        im_files = [f"image_{i}.jpg" for i in range(len(batch))]

        return {
            "img": images,  # Don't convert to half precision here - let the validator handle it
            "bboxes": bboxes,
            "cls": classes,
            "batch_idx": batch_idx,
            "im_file": im_files,
            "ori_shape": ori_shapes,
            "ratio_pad": ratio_pads,
        }


class OTXUltralyticsDataModule(OTXDataModule):
    """OTX DataModule wrapper producing Ultralytics-ready dataloaders."""

    def __init__(
        self,
        task: OTXTaskType,
        data_format: str,
        data_root: str,
        train_subset: SubsetConfig,
        val_subset: SubsetConfig,
        test_subset: SubsetConfig,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        include_polygons: bool = False,
        ignore_index: int = 255,
        unannotated_items_ratio: float = 0.0,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
        input_size: tuple[int, int] | str = "auto",
        input_size_multiplier: int = 1,
    ) -> None:
        """Constructor."""
        super().__init__(
            task=task,
            data_format=data_format,
            data_root=data_root,
            train_subset=train_subset,
            val_subset=val_subset,
            test_subset=test_subset,
            tile_config=tile_config,
            image_color_channel=image_color_channel,
            include_polygons=include_polygons,
            ignore_index=ignore_index,
            unannotated_items_ratio=unannotated_items_ratio,
            auto_num_workers=auto_num_workers,
            device=device,
            input_size=input_size,
            input_size_multiplier=input_size_multiplier,
        )

        self.subsets: dict[str, OTXDataset] = {}
        self.save_hyperparameters(ignore=["input_size"])

        dataset = DmDataset.import_from(self.data_root, format=self.data_format)
        if self.task != OTXTaskType.H_LABEL_CLS and not (
            self.task == OTXTaskType.KEYPOINT_DETECTION and self.data_format == "arrow"
        ):
            dataset = pre_filtering(
                dataset,
                self.data_format,
                self.unannotated_items_ratio,
                self.task,
                ignore_index=self.ignore_index if self.task == "SEMANTIC_SEGMENTATION" else None,
            )
        if isinstance(input_size, str) and input_size == "auto":
            input_size = adapt_input_size_to_dataset(
                dataset,
                self.task,
                input_size_multiplier,
            )
        elif not isinstance(input_size, (tuple, list)):
            msg = f"input_size should be tuple/list of ints or 'auto', but got {input_size}"
            raise ValueError(msg)

        for subset_cfg in [train_subset, val_subset, test_subset]:
            if subset_cfg.input_size is None:
                subset_cfg.input_size = input_size  # type: ignore[assignment]

        # get mean and std from Normalize transform
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        if train_subset.transforms is not None:
            for transform in train_subset.transforms:
                if isinstance(transform, dict) and "Normalize" in transform.get("class_path", ""):
                    # CLI case with jsonargparse
                    mean = transform["init_args"].get("mean", (0.0, 0.0, 0.0))
                    std = transform["init_args"].get("std", (1.0, 1.0, 1.0))
                    break

                if isinstance(transform, Normalize):
                    # torchvision.transforms case
                    mean = transform.mean
                    std = transform.std
                    break

        self.input_mean = mean
        self.input_std = std
        self.input_size = input_size

        if self.tile_config.enable_tiler and self.tile_config.enable_adaptive_tiling:
            adapt_tile_config(self.tile_config, dataset=dataset, task=self.task)

        config_mapping = {
            self.train_subset.subset_name: self.train_subset,
            self.val_subset.subset_name: self.val_subset,
            self.test_subset.subset_name: self.test_subset,
        }

        if self.auto_num_workers:
            if self.device not in [DeviceType.gpu, DeviceType.auto]:
                log.warning(
                    "Only GPU device type support auto_num_workers. "
                    f"Current deveice type is {self.device!s}. auto_num_workers is skipped.",
                )
            elif (num_workers := get_adaptive_num_workers()) is not None:
                for subset_name, subset_config in config_mapping.items():
                    log.info(
                        f"num_workers of {subset_name} subset is changed : "
                        f"{subset_config.num_workers} -> {num_workers}",
                    )
                    subset_config.num_workers = num_workers

        label_infos: list[LabelInfo] = []

        for name, dm_subset in dataset.subsets().items():
            if name not in config_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            transforms = TransformLibFactory.generate(config_mapping[name])
            common_kwargs = {
                "dm_subset": dm_subset.as_dataset(),
                "transforms": transforms,
                "data_format": self.data_format,
                "image_color_channel": self.image_color_channel,
                "to_tv_image": config_mapping[name].to_tv_image,
            }

            dataset = OTXUltralyticsDetectionDataset(**common_kwargs)

            if self.tile_config.enable_tiler:
                dataset = OTXTileDatasetFactory.create(
                    task=self.task,
                    dataset=dataset,
                    tile_config=self.tile_config,
                )
            self.subsets[name] = dataset
            label_infos += [self.subsets[name].label_info]
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

        if self._is_meta_info_valid(label_infos) is False:
            msg = "All data meta infos of subsets should be the same."
            raise ValueError(msg)

        self.label_info = next(iter(label_infos))

    def train_dataloader(self) -> DataLoader:
        """Build infinite train dataloader for Ultralytics integration."""
        config = self.train_subset
        dataset = self._get_dataset(config.subset_name)

        return InfiniteDataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            collate_fn=dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Build infinite val dataloader for Ultralytics integration."""
        config = self.val_subset
        dataset = self._get_dataset(config.subset_name)

        return InfiniteDataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            collate_fn=dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Build infinite test dataloader for Ultralytics integration."""
        config = self.test_subset
        dataset = self._get_dataset(config.subset_name)

        return InfiniteDataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            collate_fn=dataset.collate_fn,
        )
