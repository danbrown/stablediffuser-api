import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from config import HF_AUTH_TOKEN

def loadModel(model_id: str, mode="prompt", load=True):
    # torch.multiprocessing.set_start_method("spawn", force=True)
    
    print(("Loading" if load else "Downloading") + " model: " + model_id)

    model = None

    if mode == "prompt":
      model = StableDiffusionPipeline.from_pretrained(
          model_id,
          revision="fp16",
          torch_dtype=torch.float16,
          use_auth_token=HF_AUTH_TOKEN,
      )
    elif mode == "img2img":
      model = StableDiffusionImg2ImgPipeline.from_pretrained(
          model_id,
          revision="fp16",
          torch_dtype=torch.float16,
          use_auth_token=HF_AUTH_TOKEN,
      )

    device = torch.device("cuda:0")
    print("Loading model to device: " + str(device))
    # return model if load else None
    return model.to(device) if load else None
