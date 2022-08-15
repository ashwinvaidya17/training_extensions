# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from ote.api.configuration.default_model_parameters import DefaultModelParameters
from ote.api.tests.constants.ote_api_components import OteApiComponent
from ote.api.tests.constants.requirements import Requirements


@pytest.mark.components(OteApiComponent.OTE_API)
class TestModelConfiguration:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_configuration(self):
        mc = DefaultModelParameters()
        assert hasattr(mc, "learning_parameters")

        epoch_default = mc.learning_parameters.get_metadata("epochs")["default_value"]
        batch_size_default = mc.learning_parameters.get_metadata("batch_size")[
            "default_value"
        ]

        assert mc.learning_parameters.epochs == epoch_default
        assert mc.learning_parameters.batch_size == batch_size_default

        mc.learning_parameters.epochs = epoch_default + 5
        mc.learning_parameters.batch_size = batch_size_default + 4

        assert mc.learning_parameters.batch_size == batch_size_default + 4
        assert mc.learning_parameters.epochs == epoch_default + 5
