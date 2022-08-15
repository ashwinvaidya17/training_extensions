# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, Dict, Optional, Union, Iterable
import warnings

import cv2
import numpy as np

from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue
from openvino.model_zoo.model_api.adapters.model_adapter import ModelAdapter
from ote.api.utils.argument_checks import check_input_parameters_type
from ote.api.utils.segmentation_utils import create_hard_prediction_from_soft_prediction


@check_input_parameters_type()
def get_actmap(
    features: Union[np.ndarray, Iterable, int, float], output_res: Union[tuple, list]
):
    am = cv2.resize(features, output_res)
    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
    am = cv2.cvtColor(am, cv2.COLOR_BGR2RGB)
    return am


class BlurSegmentation(SegmentationModel):
    __model__ = "blur_segmentation"

    @check_input_parameters_type()
    def __init__(
        self,
        model_adapter: ModelAdapter,
        configuration: Optional[dict] = None,
        preload: bool = False,
    ):
        super().__init__(model_adapter, configuration, preload)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                "soft_threshold": NumericalValue(default_value=0.5, min=0.0, max=1.0),
                "blur_strength": NumericalValue(
                    value_type=int, default_value=1, min=0, max=25
                ),
            }
        )

        return parameters

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        pass

    def _get_outputs(self):
        layer_name = "output"
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) == 3:
            self.out_channels = 0
        elif len(layer_shape) == 4:
            self.out_channels = layer_shape[1]
        else:
            raise Exception(
                "Unexpected output layer shape {}. Only 4D and 3D output layers are supported".format(
                    layer_shape
                )
            )

        return layer_name

    @check_input_parameters_type()
    def postprocess(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        predictions = outputs[self.output_blob_name].squeeze()
        soft_prediction = np.transpose(predictions, axes=(1, 2, 0))

        hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=self.soft_threshold,
            blur_strength=self.blur_strength,
        )
        hard_prediction = cv2.resize(
            hard_prediction,
            metadata["original_shape"][1::-1],
            0,
            0,
            interpolation=cv2.INTER_NEAREST,
        )

        if "feature_vector" not in outputs or "saliency_map" not in outputs:
            warnings.warn(
                "Could not find Feature Vector and Saliency Map in OpenVINO output. "
                "Please rerun OpenVINO export or retrain the model."
            )
            metadata["saliency_map"] = None
            metadata["feature_vector"] = None
        else:
            metadata["saliency_map"] = get_actmap(
                outputs["saliency_map"][0],
                (metadata["original_shape"][1], metadata["original_shape"][0]),
            )
            metadata["feature_vector"] = outputs["feature_vector"]

        return hard_prediction
