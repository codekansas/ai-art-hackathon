import tqdm
import torch
from pathlib import Path
from pyannote.audio import Model
from pyannote.audio import Inference
from diffusers import DiffusionPipeline
import tqdm
import json
from torch import nn

mapping_ckpt_path = "ckpt.pt"
mapping_ckpt = torch.load(mapping_ckpt_path)

device = "cuda"

def get_model(prefix):
    weight, bias = mapping_ckpt[f'{prefix}_pred.weight'], mapping_ckpt[f'{prefix}_pred.bias']
    model = nn.Linear(weight.shape[1], weight.shape[0])
    model.load_state_dict({'weight': weight, 'bias': bias})
    model.to(device)
    return model

age_model = get_model("age")
country_model = get_model("country")
sex_model = get_model("sex")

def get_mapping(prefix):
    with open(f"{prefix}.json", "r") as f:
        rev_mapping = json.load(f)
        return {k: v for v, k in rev_mapping.items()}

age_mapping = get_mapping("ages")
country_mapping = get_mapping("countries")
sex_mapping = get_mapping("sexes")

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to(device)

audio_model = Model.from_pretrained("pyannote/embedding",  use_auth_token="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")
audio_model.to(device)
inference = Inference(audio_model, window="whole")

root_dir = '/treasure/big-fast/data/datasets/kaggle/recordings/recordings'
# files = list(Path(root_dir).glob('*.mp3'))
# files = [Path("/home/ben/Github/hackathon/AudioCLIP/british.flac")]
files = [Path("/home/ben/Github/hackathon/AudioCLIP/pcs3.flac")]

for file in tqdm.tqdm(files):
    embedding = inference(file)
    embedding = torch.from_numpy(embedding).to(device).unsqueeze(0)
    age = age_mapping[int(age_model(embedding).argmax(-1).item())]
    country = country_mapping[int(country_model(embedding).argmax(-1).item())]
    sex = sex_mapping[int(sex_model(embedding).argmax(-1).item())]
    prompt = f"a picture of an {age}-year-old {sex} from {country}"
    print("prompt:", prompt)
    image = pipe(prompt=prompt).images[0]
    image.save(f'/treasure/big-fast/data/datasets/kaggle/sd_images/{file.stem}.png')
