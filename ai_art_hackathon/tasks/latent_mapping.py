"""Defines a simple supervised learning template task.

This task is meant to be used as a template for creating new tasks. Just
change the key from ``template`` to whatever you want to name your task, and
implement the following methods:

- :meth:`run_model`
- :meth:`compute_loss`
- :meth:`get_dataset`
"""

from dataclasses import dataclass
from ai_art_hackathon.tasks.datasets.latent_dataset import LatentMappingDataset
import ml.api as ml
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataset import Dataset


@dataclass
class LatentMappingTaskConfig(ml.SupervisedLearningTaskConfig):
    pass


# These types are defined here so that they can be used consistently
# throughout the task and only changed in one location.
Model = ml.BaseModel
Batch = tuple[Tensor, Tensor]
Output = Tensor
Loss = Tensor


@ml.register_task("latent_mapping", LatentMappingTaskConfig)
class LatentMappingTask(ml.SupervisedLearningTask[LatentMappingTaskConfig, Model, Batch, Output, Loss]):
    def run_model(self, model: Model, batch: Batch, state: ml.State) -> Output:
        audio_emb, _, _, _, _ = batch
        return model(audio_emb)

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        (_, age, _, country, sex), (p_age, p_country, p_sex) = batch, output
        return {
            "age": F.cross_entropy(p_age, age.squeeze(1).long()),
            "country": F.cross_entropy(p_country, country.squeeze(1).long()),
            "sex": F.cross_entropy(p_sex, sex.squeeze(1).long()),
        }

    def get_dataset(self, phase: ml.Phase) -> Dataset:
        return LatentMappingDataset()
