"""Ultralytics dataloading integration for OTX.

This module defines an `OTXUltralyticsDetectionDataset` that emits batches compatible
with Ultralytics YOLO v8 detection trainers, and a corresponding data module that
provides infinite dataloaders for train/val/test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torchvision import tv_tensors
from torchvision.ops import box_convert
from ultralytics.data.build import InfiniteDataLoader

from otx.data.dataset import OTXDetectionDataset
from otx.data.entity import OTXDataItem
from otx.data.module import OTXDataModule

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


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
