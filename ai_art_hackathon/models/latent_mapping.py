"""Defines a template mode.

To use this, change the key from ``"template"`` to whatever your project
is called. Next, just override the ``forward`` model to whatever signature
your task expects, and you're good to go!
"""

from dataclasses import dataclass

from omegaconf import MISSING
import ml.api as ml
from torch import Tensor, nn


@dataclass
class LatentMappingModelConfig(ml.BaseModelConfig):
    pass


@ml.register_model("latent_mapping", LatentMappingModelConfig)
class LatentMappingModel(ml.BaseModel[LatentMappingModelConfig]):
    def __init__(self, config: LatentMappingModelConfig) -> None:
        super().__init__(config)

        self.age_pred = nn.Linear(512, 77)
        self.country_pred = nn.Linear(512, 176)
        self.sex_pred = nn.Linear(512, 2)

    def forward(self, x: Tensor):
        return self.age_pred(x), self.country_pred(x), self.sex_pred(x)
