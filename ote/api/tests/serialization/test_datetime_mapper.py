#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from datetime import datetime

import pytest
from ote.api.serialization.datetime_mapper import DatetimeMapper
from ote.api.tests.constants.ote_api_components import OteApiComponent
from ote.api.tests.constants.requirements import Requirements
from ote.api.utils.time_utils import now


@pytest.mark.components(OteApiComponent.OTE_API)
class TestDatetimeMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialization_deserialization(self):
        """
        This test serializes datetime, deserializes serialized datetime and compares with original one.
        """

        original_time = now()
        serialized_time = DatetimeMapper.forward(original_time)
        assert serialized_time == original_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        deserialized_time = DatetimeMapper.backward(serialized_time)
        assert original_time == deserialized_time

        deserialized_time = DatetimeMapper.backward(None)
        assert isinstance(deserialized_time, datetime)
