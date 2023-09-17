from diffusers import DiffusionPipeline
import torch
import tqdm
import csv

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

# prompt = "An anime avatar of a Danish person"

csv_path = '/treasure/big-fast/data/datasets/kaggle/speakers_all.csv'
with open(csv_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for prefix in ["anime ", ""]:
        for row in tqdm.tqdm(csv_reader):
            age, age_onset, birthplace, filename, native_language, sex, speakerid, country, file_missing, *_ = row
            if file_missing == "TRUE":
                continue
            # audio_path = f"/treasure/big-fast/data/datasets/kaggle/recordings/recordings/{filename}.mp3"
            prompt = f"a {prefix}picture of an {age}-year-old {sex} from {birthplace}"

            images = pipe(prompt=prompt).images[0]
            break
            # latent = pipe(prompt=prompt)
            # torch.save(latent, f'/treasure/big-fast/data/datasets/kaggle/sd_embeddings/{prefix}{filename}.pt')

# Saves the image.
# images.save("image.png")
