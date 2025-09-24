from __future__ import annotations

import logging
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torchvision.transforms.v2 import (
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.utils import DEFAULT_CFG, ops

from otx.backend.ultralytics import OTXUltralyticsDataModule
from otx.config.data import SubsetConfig
from otx.data.entity.base import ImageInfo
from otx.data.entity.torch.torch import OTXPredBatch
from otx.engine import Engine
from otx.types import OTXTaskType, PathLike
from otx.types.export import OTXExportFormatType

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger
    from torch.utils.data import DataLoader
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.utils import IterableSimpleNamespace

    from otx.data.dataset.base import OTXDataset
    from otx.data.module import OTXDataModule
    from otx.types.types import DATA, METRICS, MODEL

log = logging.getLogger(__name__)


class _OTXUltralyticsDetectionTrainer(DetectionTrainer):
    # unfortunately, we need to set the datamodule here as model accepts the class rather than instance.
    datamodule: OTXDataModule
    progress_bar_callback: Callback | None

    def __init__(
        self, cfg: IterableSimpleNamespace = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ):
        # add save_dir to cfg otherwise it skips the key
        cfg.save_dir = cfg.get("save_dir", overrides.get("save_dir"))
        super().__init__(cfg, overrides, _callbacks)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train") -> DataLoader:
        if mode == "train":
            return self.datamodule.train_dataloader()
        if mode == "val":
            return self.datamodule.val_dataloader()
        if mode == "test":
            return self.datamodule.test_dataloader()
        msg = f"Invalid mode: {mode}"
        raise ValueError(msg)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return _OTXUltralyticsDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, OTXDataset]:
        train_subset = (
            self.datamodule.subsets.get("train")
            if "train" in self.datamodule.subsets
            else self.datamodule.subsets["TRAINING"]
        )
        val_subset = (
            self.datamodule.subsets.get("val")
            if "val" in self.datamodule.subsets
            else self.datamodule.subsets["VALIDATION"]
        )
        test_subset = (
            self.datamodule.subsets.get("test")
            if "test" in self.datamodule.subsets
            else self.datamodule.subsets["TESTING"]
        )
        return {
            "train": train_subset,
            "val": val_subset,
            "test": test_subset,
            "nc": self.datamodule.label_info.num_classes,
            "channels": 3,
            "names": self.datamodule.label_info.label_names,
        }

    def run_callbacks(self, event: str) -> None:
        """Pseudo process the callbacks."""
        super().run_callbacks(event)
        if event == "on_train_epoch_start" and self.progress_bar_callback is not None:
            self.progress_bar_callback.progress_updater.update_progress(progress=self.epoch / self.epochs)

    def plot_training_labels(self) -> None:
        """Skip method."""

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess the batch for training."""
        # We already add transforms on top
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch


class _OTXUltralyticsDetectionValidator(DetectionValidator):
    datamodule: OTXDataModule

    def __init__(
        self,
        dataloader: InfiniteDataLoader | None = None,
        save_dir: Path | None = None,
        args: IterableSimpleNamespace | None = None,
        _callbacks: dict[str, list] | None = None,
    ):
        args.save_dir = str(save_dir) if save_dir is not None else args.get("save_dir", None)
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect"

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train") -> DataLoader:
        if mode == "train":
            return self.datamodule.train_dataloader()
        if mode == "val":
            return self.datamodule.val_dataloader()
        if mode == "test":
            return self.datamodule.test_dataloader()
        msg = f"Invalid mode: {mode}"
        raise ValueError(msg)

    def get_dataset(self) -> dict[str, OTXDataset]:
        train_subset = (
            self.datamodule.subsets.get("train")
            if "train" in self.datamodule.subsets
            else self.datamodule.subsets["TRAINING"]
        )
        val_subset = (
            self.datamodule.subsets.get("val")
            if "val" in self.datamodule.subsets
            else self.datamodule.subsets["VALIDATION"]
        )
        test_subset = (
            self.datamodule.subsets.get("test")
            if "test" in self.datamodule.subsets
            else self.datamodule.subsets["TESTING"]
        )
        return {
            "train": train_subset,
            "val": val_subset,
            "test": test_subset,
            "nc": self.datamodule.label_info.num_classes,
            "channels": 3,
            "names": self.datamodule.label_info.label_names,
        }

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess the batch for validation."""
        # We already add transforms on top
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation.

        Override to handle 0-dimensional tensor issue.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx]

        # Handle the case where we get a 0-dimensional tensor
        if cls.dim() == 0:
            # If it's a 0-d tensor, it means there are no annotations for this image
            cls = torch.zeros((0,), dtype=cls.dtype, device=cls.device)

        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si] if batch["im_file"] is not None else f"image_{si}.jpg",
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the predictions in JSON format."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json_list = list((Path(self.datamodule.data_root) / "annotations").glob("*_val.json"))
        if len(anno_json_list) == 0:
            return {}
        anno_json = anno_json_list[0].resolve()
        return self.coco_evaluate(stats, pred_json, anno_json)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        file_image_id = int(stem) if stem.isnumeric() else stem
        image_id = pbatch.get("image_id", file_image_id)
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )


class _OTXUltralyticsDetectionPredictor(DetectionPredictor):
    datamodule: OTXDataModule

    def __init__(
        self, cfg: IterableSimpleNamespace = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ):
        # add save_dir to cfg otherwise it skips the key
        cfg.save_dir = cfg.get("save_dir", overrides.get("save_dir"))
        super().__init__(cfg, overrides, _callbacks)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train") -> DataLoader:
        if mode == "train":
            return self.datamodule.train_dataloader()
        if mode == "val":
            return self.datamodule.val_dataloader()
        if mode == "test":
            return self.datamodule.test_dataloader()
        msg = f"Invalid mode: {mode}"
        raise ValueError(msg)

    def get_dataset(self) -> dict[str, OTXDataset]:
        train_subset = (
            self.datamodule.subsets.get("train")
            if "train" in self.datamodule.subsets
            else self.datamodule.subsets["TRAINING"]
        )
        val_subset = (
            self.datamodule.subsets.get("val")
            if "val" in self.datamodule.subsets
            else self.datamodule.subsets["VALIDATION"]
        )
        test_subset = (
            self.datamodule.subsets.get("test")
            if "test" in self.datamodule.subsets
            else self.datamodule.subsets["TESTING"]
        )
        return {
            "train": train_subset,
            "val": val_subset,
            "test": test_subset,
            "nc": self.datamodule.label_info.num_classes,
            "channels": 3,
            "names": self.datamodule.label_info.label_names,
        }


class UltralyticsEngine(Engine):
    """Ultralytics Engine.

    Args:
        model (YOLO): The model to use. Defaults to "yolov8n.pt". Refer to https://docs.ultralytics.com/models/ for more information.
        task (TaskType): The task to use. Defaults to TaskType.DETECT.
        work_dir (PathLike): The working directory to use. Defaults to "./otx-workspace".

    """

    def __init__(
        self,
        model: YOLO,
        data: OTXDataModule,
        task: OTXTaskType = OTXTaskType.DETECTION,
        work_dir: PathLike = "./otx-workspace",
    ):
        self._model = model
        if task == OTXTaskType.DETECTION:
            self.task = "detect"
        else:
            msg = f"Unexpected task type: {task}"
            raise ValueError(msg)
        self._work_dir = Path(work_dir)
        self.metrics_logger: Logger | None = None
        # this part is hacky
        _OTXUltralyticsDetectionTrainer.datamodule = data
        _OTXUltralyticsDetectionValidator.datamodule = data
        _OTXUltralyticsDetectionPredictor.datamodule = data

    def train(
        self,
        max_epochs: int = 100,
        checkpoint: PathLike | None = None,
        callbacks: list[Callback] | None = None,
        logger: list[Logger] | Logger | None = None,
        **kwargs,
    ) -> METRICS:
        """Train entrypoint.

        Note: The name and data type follows the Geti convention

        Args:
            max_epochs (int): The maximum number of epochs to train.
            checkpoint (PathLike): The checkpoint to load.
            callbacks (list[Callbacks]): The callbacks to use.
            logger (list[Logger]): The logger to use.
            **kwargs: Additional kwargs to pass to the model.train method.
        """
        if checkpoint is not None:
            self.model.load(checkpoint)
        # TODO: sort out progress logger and metric callback
        # Deal with the Progress callback passed by Geti
        _OTXUltralyticsDetectionTrainer.progress_bar_callback = None

        if callbacks is not None:
            for callback in callbacks:
                if hasattr(callback, "progress_updater"):
                    _OTXUltralyticsDetectionTrainer.progress_bar_callback = callback
                    break

        _loggers = logger if isinstance(logger, list) else [logger]
        for _logger in _loggers:
            if hasattr(_logger, "metrics"):
                self.metrics_logger = _logger
                break

        log.info(f"Following kwargs are unprocessed: {kwargs}")

        if _OTXUltralyticsDetectionTrainer.progress_bar_callback is not None:
            _OTXUltralyticsDetectionTrainer.progress_bar_callback.progress_updater.update_progress(progress=0.0)

        self.model.train(
            trainer=_OTXUltralyticsDetectionTrainer,
            epochs=max_epochs,
            save_dir=self._work_dir,
        )

        if _OTXUltralyticsDetectionTrainer.progress_bar_callback is not None:
            _OTXUltralyticsDetectionTrainer.progress_bar_callback.progress_updater.update_progress(progress=1.0)

        if self.metrics_logger is not None:
            for key, value in self.model.metrics:
                if isinstance(value, float):
                    self.metrics_logger.log_metrics(key, value)

    def test(self, data: DATA, **kwargs) -> METRICS:
        log.warning("Ultralytics Engine does not suppport testing. Ignoring...")
        return {}

    def predict(
        self,
        data: OTXUltralyticsDataModule | PathLike | list[np.array] | None = None,
        checkpoint: PathLike | None = None,
    ) -> list[OTXPredBatch]:
        """Predict on the model."""
        if checkpoint is not None:
            self.model = checkpoint

        datamodule: None | OTXUltralyticsDataModule = None
        if data is None:
            datamodule = self.datamodule
        elif isinstance(data, list) and isinstance(data[0], np.ndarray):
            _data = data
        elif isinstance(data, (str, PathLike)):
            test_transforms = [
                Resize(size=(640, 640)),
                ToDtype(torch.float32),
                Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
                ToImage(),
            ]
            datamodule = OTXUltralyticsDataModule(
                data_root=data,
                data_format="coco_instances",
                task=self.task,
                train_subset=SubsetConfig(
                    batch_size=32,
                    subset_name="train",
                    transforms=test_transforms,
                ),
                val_subset=SubsetConfig(
                    batch_size=32,
                    subset_name="val",
                    transforms=test_transforms,
                ),
                test_subset=SubsetConfig(
                    batch_size=32,
                    subset_name="test",
                    transforms=test_transforms,
                ),
            )
        elif isinstance(data, OTXUltralyticsDataModule):
            datamodule = data
        else:
            msg = "The input data should be either a datamodule, valid path to data root or a list of numpy arrays."
            raise TypeError(msg)

        if datamodule is None:
            predictions = self.model.predict(
                _data, save_dir=self._work_dir, predictor=_OTXUltralyticsDetectionPredictor
            )
        else:
            predictions = []
            for data_batch in datamodule.predict_dataloader():
                for item in data_batch["img"]:
                    predictions.extend(
                        self.model.predict(
                            item.unsqueeze(0), save_dir=self._work_dir, predictor=_OTXUltralyticsDetectionPredictor
                        )
                    )
        converted_predictions = []

        for idx, prediction in enumerate(predictions):
            converted_predictions.append(
                OTXPredBatch(
                    batch_size=1,
                    images=torch.tensor(prediction.orig_img).permute(2, 0, 1).unsqueeze(0),
                    bboxes=[prediction.boxes.xyxy],
                    scores=[prediction.boxes.conf],
                    labels=[prediction.boxes.cls.to(torch.long)],
                    imgs_info=ImageInfo(
                        img_idx=idx, img_shape=prediction.orig_img.shape, ori_shape=prediction.orig_shape
                    ),
                )
            )
        return converted_predictions

    def export(
        self,
        checkpoint: PathLike | None = None,
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        **kwargs,
    ) -> MODEL:
        if checkpoint is not None:
            self.model = checkpoint
        log.info(f"Following kwargs are unprocessed: {kwargs}")
        return self.model.export(format=export_format.value.lower())

    def track(self, source: str | Path | int | list | tuple | list | np.ndarray | torch.Tensor) -> METRICS: ...

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        return bool(isinstance(model, YOLO) and isinstance(data, OTXUltralyticsDataModule))

    @property
    def work_dir(self) -> PathLike:
        return self._work_dir

    @property
    def model(self) -> MODEL:
        if self._model is None:
            msg = "Model is not set"
            raise RuntimeError(msg)
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module | Path | str) -> None:
        if isinstance(model, torch.nn.Module):
            self._model = model
            return

        model_path = Path(model)  # Path or str
        if (
            self._model is not None
            and isinstance(self._model, torch.nn.Module)
            and model_path.resolve().suffix in [".pt", "pth"]
        ):
            log.info("loading torch checkpoint")
            self._model.load(model_path)
        elif model_path.is_dir() or model_path.resolve().suffix in [".xml", ".bin"]:
            model_path = model_path.parent if model_path.is_file() else model_path
            log.info("loading openvino model")
            self._model = YOLO(model_path, task=self.task)
        else:
            msg = "Model must be a torch.nn.Module, Path, or str"
            raise ValueError(msg)

    @property
    def checkpoint(self) -> PathLike | None:
        if self.model.ckpt_path is None:
            msg = "Checkpoint is not set"
            raise RuntimeError(msg)
        return self.model.ckpt_path

    @property
    def datamodule(self) -> DATA:
        return _OTXUltralyticsDetectionTrainer.datamodule
