# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from config import MODEL_IDS, PRELOAD_MODEL

import torch
if torch.cuda.is_available():
  print("CUDA is available download")
  torch.cuda.set_device(0)
  print("Using CUDA device download: " + torch.cuda.get_device_name())
  torch.multiprocessing.set_start_method("spawn", force=True) 
  print("Using torch.multiprocessing on download: " + torch.multiprocessing.get_start_method())

from loadModel import loadModel


def download_model():
  # do a dry run of loading the huggingface model, which will download weights at build time
  # For local dev & preview deploys, download all the models (terrible for serverless deploys)
  if PRELOAD_MODEL == "ALL":
    for MODEL in MODEL_IDS:
      loadModel(model_id=MODEL, load=False)
  else:
      loadModel(model_id=PRELOAD_MODEL, load=False)


if __name__ == "__main__":
  download_model()
  exit(0)
