"""Latent mapping dataset."""

import csv

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from pathlib import Path
import json


def get_mapping(l):
    unique_values = list(sorted(list(set(l))))
    return {k: i for i, k in enumerate(unique_values)}


class LatentMappingDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.root = "/treasure/big-fast/data/datasets/kaggle"
        csv_path = f'{self.root}/speakers_all.csv'
        self.files = []
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                age, _, birthplace, filename, _, sex, _, country, missing, *_ = row
                if not country:
                    continue
                age = str(round(float(age)))
                if missing == "TRUE":
                    continue
                audio_emb_path = Path(f"{self.root}/audio_embeddings/{filename}.pt")
                if not audio_emb_path.exists():
                    continue
                if sex == "famale":
                    sex = "female"
                self.files.append((age, birthplace, sex, country, audio_emb_path))

        # Maps all properties to integers
        self.age_mapping = get_mapping([f[0] for f in self.files])
        self.birthplace_mapping = get_mapping([f[1] for f in self.files])
        self.sex_mapping = get_mapping([f[2] for f in self.files])
        self.country_mapping = get_mapping([f[3] for f in self.files])

    def __getitem__(self, index: int) -> tuple[Tensor, int, int, int, int]:
        age, birthplace, sex, country, audio_emb_path = self.files[index]
        age_id = self.age_mapping[age]
        birthplace_id = self.birthplace_mapping[birthplace]
        country_id = self.country_mapping[country]
        sex_id = self.sex_mapping[sex]
        return torch.load(audio_emb_path, map_location='cpu'), age_id, birthplace_id, country_id, sex_id

    def __len__(self) -> int:
        return len(self.files)


def main() -> None:
    ds = LatentMappingDataset()

    print("unique ages:", len(ds.age_mapping))
    print("unique birthplaces:", len(ds.birthplace_mapping))
    print("unique countries:", len(ds.country_mapping))
    print("unqiue sexes:", len(ds.sex_mapping))

    with open("ages.json", "w") as f:
        json.dump(ds.age_mapping, f, indent=2)

    with open("birthplaces.json", "w") as f:
        json.dump(ds.birthplace_mapping, f, indent=2)

    with open("countries.json", "w") as f:
        json.dump(ds.country_mapping, f, indent=2)

    with open("sexes.json", "w") as f:
        json.dump(ds.sex_mapping, f, indent=2)

    for i in range(len(ds)):
        audio_emb, _, _, _, _ = ds[i]
        print(audio_emb.shape)
        if i >= 3:
            break


if __name__ == "__main__":
    # python -m ai_art_hackathon.tasks.datasets.latent_dataset
    main()
