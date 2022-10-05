# This file configure the variables used in the other scripts
# Like tokens, paths, modules, etc.

import os

# Api port and host
API_HOST = '0.0.0.0'
API_PORT = 3000

# Base path of this project
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# Path to the directory where the patched files are located
PATCHED_PATH = os.path.join(BASE_PATH, 'patches')

# Path to the modules
MODULES_PATH = os.path.join(BASE_PATH, 'modules')

# Path to scripts
SCRIPTS_PATH = os.path.join(BASE_PATH, 'scripts')

# Token to authenticate in huggingface
HF_AUTH_TOKEN = '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'

# Models to load on diffusers
MODEL_IDS = ["hakurei/waifu-diffusion", "CompVis/stable-diffusion-v1-4", "doohickey/trinart-waifu-diffusion-50-50"]

# "CompVis/stable-diffusion-v1-4", "hakurei/waifu-diffusion"
PRELOAD_MODEL = "ALL"

# Upscalers modules to load
UPSCALERS = ["None","GFPGAN","Enhanced Real-ESRGAN", "GFPGAN + Enhanced ESRGAN", "CodeFormer", "CodeFormer + Enhanced ESRGAN", "IMG2IMG"]

# Schedulers modules to load
SCHEDULERS = ["default", "pndm", "k-lms", "ddim", 'ddim clip sampled']

# Pipelines modules to load
PIPELINES = ["StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline"]