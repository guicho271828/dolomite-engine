# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import EnergyConfig
from .layer import EnergyBlock


class EnergyPreTrainedModel(PreTrainedModelMixin):
    config_class = EnergyConfig
    layer_class = EnergyBlock
    _no_split_modules_override = None

    @property
    def _no_split_modules(self):
        if self._no_split_modules_override is not None:
            return self._no_split_modules_override
        if getattr(self.config, "shared_backbone", False):
            return []
        return ["EnergyBlock"]

    @_no_split_modules.setter
    def _no_split_modules(self, value):
        self._no_split_modules_override = value


class EnergyModel(EnergyPreTrainedModel, BaseModelMixin): ...
