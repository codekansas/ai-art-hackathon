import tqdm
import torch
from pathlib import Path
from pyannote.audio import Model
from pyannote.audio import Inference

model = Model.from_pretrained("pyannote/embedding",  use_auth_token="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")
inference = Inference(model, window="whole")

root_dir = '/treasure/big-fast/data/datasets/kaggle/recordings/recordings'
files = list(Path(root_dir).glob('*.mp3'))

for file in tqdm.tqdm(files):
    embedding = inference(file)
    torch.save(embedding, f'/treasure/big-fast/data/datasets/kaggle/audio_embeddings/{file.stem}.pt')
