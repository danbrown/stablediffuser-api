# This file set up utility functions

# Get a remote file and save it to disk (using wget)
def wgeto(url, outputdir): # PAGODA
  import sys, subprocess
  res = None
  try:
    res = subprocess.run(['wget', '-q', '--show-progress', url, '-O', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  except OSError as e:
    raise e
  finally:
    if res and res.strip() != '':
      print(res)


# get the pip modules directory
def get_pip_modules_dir():
  import site
  return site.getsitepackages()[0]

# manage imports for each scrpit
def manage_imports(requester):
  if requester == 'patch_nsfw':
    from shutil import copyfile
    from os import remove
    return copyfile, remove

  elif requester == 'general_diffusion_run':
    import torch, sys
    from random import randint
    return torch, randint, sys

  elif requester == "prompt_run":
    pass

  elif requester == "img2img_run":
    pass
  
  elif requester == "inpainter_run":
    pass
  
  elif requester == 'clean_env':
    import gc, torch
    return gc, torch

  elif requester == "PROMPT":
    try:
      from diffusers import StableDiffusionPipeline
    except ModuleNotFoundError:
      Manager.Diffusion.install_diffusers()
      from diffusers import StableDiffusionPipeline
    return StableDiffusionPipeline
  
  elif requester == "IMG2IMG":
    try:
      from diffusers import StableDiffusionImg2ImgPipeline
    except ModuleNotFoundError:
      Manager.Diffusion.install_diffusers()
      from diffusers import StableDiffusionImg2ImgPipeline
    return StableDiffusionImg2ImgPipeline

  elif requester == "Inpainting":
    try:
      from diffusers import StableDiffusionInpaintPipeline
    except ModuleNotFoundError:
      Manager.Diffusion.install_diffusers()
      from diffusers import StableDiffusionInpaintPipeline
    return StableDiffusionInpaintPipeline

  elif requester == "CLIP GUIDED PROMPT":
    try: 
      from CLIP_GUIDED import CLIPGuidedStableDiffusion
    except ModuleNotFoundError:
      Manager.Diffusion.install_diffusers()
      from CLIP_GUIDED import CLIPGuidedStableDiffusion
    return CLIPGuidedStableDiffusion

  elif requester == 'diffuser_install':
    import subprocess, os
    from os import remove # PAGODA
    from shutil import copyfile # PAGODA
    return subprocess, os, copyfile, remove # PAGODA

  elif requester == 'default' or requester == 'pndm':
    from diffusers.schedulers import PNDMScheduler
    return PNDMScheduler
  
  elif requester == 'k-lms':
    from diffusers.schedulers import LMSDiscreteScheduler
    return LMSDiscreteScheduler

  elif requester == 'ddim':
    from diffusers.schedulers import DDIMScheduler
    return DDIMScheduler
