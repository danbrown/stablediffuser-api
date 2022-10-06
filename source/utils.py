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
      install_diffusers()
      from diffusers import StableDiffusionPipeline
    return StableDiffusionPipeline
  
  elif requester == "IMG2IMG":
    try:
      from diffusers import StableDiffusionImg2ImgPipeline
    except ModuleNotFoundError:
      install_diffusers()
      from diffusers import StableDiffusionImg2ImgPipeline
    return StableDiffusionImg2ImgPipeline

  elif requester == "Inpainting":
    try:
      from diffusers import StableDiffusionInpaintPipeline
    except ModuleNotFoundError:
      install_diffusers()
      from diffusers import StableDiffusionInpaintPipeline
    return StableDiffusionInpaintPipeline

  elif requester == "CLIP GUIDED PROMPT":
    try: 
      from CLIP_GUIDED import CLIPGuidedStableDiffusion
    except ModuleNotFoundError:
      install_diffusers()
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

# get hugginface credentials
def creds():
  return 'x90', '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'

# install a given model
def install_model(self, username, token, model_id):
  subprocess, os = manage_imports('diffuser_install')
  print('Installing Model')
  print(subprocess.run(['git','lfs','install'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  # This will take a while
  print(subprocess.run(['git','lfs','clone',f'https://{username}:{token}@huggingface.co/{model_id}'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

# patch the nsfw filter
def patch_nsfw(ENABLE_NSFW_FILTER):
  copyfile, remove = manage_imports('patch_nsfw')
  remove('/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py')
  if ENABLE_NSFW_FILTER:
    copyfile(f'/workspace/api/safety_checker.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py')
  else:
    copyfile(f'/workspace/api/safety_checker_patched.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py')

# install diffusers
def install_diffusers(self, settings):
  subprocess, os, copyfile, remove = manage_imports('diffuser_install') # PAGODA
  
  print('Installing FastApi') # PAGODA
  print(subprocess.run(['pip','install','fastapi', 'nest-asyncio', 'uvicorn', 'pydantic'], stdout=subprocess.PIPE).stdout.decode('utf-8')) # PAGODA
  # Colab.clear() # PAGODA

  print('Installing Transformers')
  print(subprocess.run(['pip','install','transformers'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  # Colab.clear()
  print('Installing Diffusers')
  if settings['diffusers_version'] == 'latest':
    print(subprocess.run(['pip','install','-U',f'git+https://github.com/huggingface/diffusers.git'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  else:
    print(subprocess.run(['pip','install','-U',f'git+https://github.com/huggingface/diffusers.git@{settings["diffusers_version"]}'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  # Colab.clear()
  print('Creating Directories')
  print(subprocess.run(['mkdir','diffusers_output'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  # Colab.clear()
  print('Installing Dependencies')
  print(subprocess.run(['pip','install','pytorch-pretrained-bert','spacy','ftfy','scipy'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  # Colab.clear()
  print('Populating Spacy')
  print(subprocess.run(['python','-m','space','download','en'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
  # Colab.clear()
  print('Logging into Huggingface')
  username, token = self.creds()
  subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], stdout=subprocess.DEVNULL)
  # left_of_pipe = subprocess.Popen(["echo", token], stdout=subprocess.PIPE)
  # right_of_pipe = subprocess.run(['huggingface-cli', 'login'], stdin=left_of_pipe.stdout, stdout=subprocess.PIPE).stdout.decode('utf-8')
  # Diffusion.install_model(username, token, settings["model_id"])
  # Colab.clear()
  if not os.path.exists('/workspace/api/safety_checker_patched.py'):
    print('Creating Patches')
    print(subprocess.run(['cp','/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py','/workspace/api/safety_checker.py'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['cp','/workspace/api/safety_checker.py','/workspace/api/safety_checker_patched.py'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    with open(f'/workspace/api/safety_checker_patched.py','r') as unpatched_file:
      patch = unpatched_file.read().replace('for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):','#for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):').replace('if has_nsfw_concept:','# if has_nsfw_concept:').replace('images[idx] = np.zeros(images[idx].shape)  # black image', '# images[idx] = np.zeros(images[idx].shape)  # black image').replace("Potential NSFW content was detected in one or more images. A black image will be returned instead.","Potential NSFW content was detected in one or more images. It's patched out, no actions were taken.").replace(" Try again with a different prompt and/or seed.","")
    with open(f'/workspace/api/safety_checker_patched.py','w') as file:
      file.write(patch)
  with open('/workspace/api/CLIP_GUIDED.py', 'w') as file:
    file.write('''
import inspect
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import AutoencoderKL, DiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from torchvision import transforms
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTextModel, CLIPTokenizer


class MakeCutouts(nn.Module):
def __init__(self, cut_size, cut_power=1.0):
    super().__init__()

    self.cut_size = cut_size
    self.cut_power = cut_power

def forward(self, pixel_values, num_cutouts):
    sideY, sideX = pixel_values.shape[2:4]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, self.cut_size)
    cutouts = []
    for _ in range(num_cutouts):
        size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
        offsetx = torch.randint(0, sideX - size + 1, ())
        offsety = torch.randint(0, sideY - size + 1, ())
        cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
        cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
    return torch.cat(cutouts)


def spherical_dist_loss(x, y):
x = F.normalize(x, dim=-1)
y = F.normalize(y, dim=-1)
return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def set_requires_grad(model, value):
for param in model.parameters():
    param.requires_grad = value


class CLIPGuidedStableDiffusion(DiffusionPipeline):
"""CLIP guided stable diffusion based on the amazing repo by @crowsonkb and @Jack000
- https://github.com/Jack000/glid-3-xl
- https://github.dev/crowsonkb/k-diffusion
"""

def __init__(
    self,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    clip_model: CLIPModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    scheduler: Union[PNDMScheduler, LMSDiscreteScheduler],
    feature_extractor: CLIPFeatureExtractor,
):
    super().__init__()
    scheduler = scheduler.set_format("pt")
    self.register_modules(
        vae=vae,
        text_encoder=text_encoder,
        clip_model=clip_model,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )

    self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    self.make_cutouts = MakeCutouts(feature_extractor.size)

    set_requires_grad(self.text_encoder, False)
    set_requires_grad(self.clip_model, False)

def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
    if slice_size == "auto":
        # half the attention head size is usually a good trade-off between
        # speed and memory
        slice_size = self.unet.config.attention_head_dim // 2
    self.unet.set_attention_slice(slice_size)

def disable_attention_slicing(self):
    self.enable_attention_slicing(None)

def freeze_vae(self):
    set_requires_grad(self.vae, False)

def unfreeze_vae(self):
    set_requires_grad(self.vae, True)

def freeze_unet(self):
    set_requires_grad(self.unet, False)

def unfreeze_unet(self):
    set_requires_grad(self.unet, True)

@torch.enable_grad()
def cond_fn(
    self,
    latents,
    timestep,
    index,
    text_embeddings,
    noise_pred_original,
    text_embeddings_clip,
    clip_guidance_scale,
    num_cutouts,
    use_cutouts=True,
):
    latents = latents.detach().requires_grad_()

    if isinstance(self.scheduler, LMSDiscreteScheduler):
        sigma = self.scheduler.sigmas[index]
        # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
        latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
    else:
        latent_model_input = latents

    # predict the noise residual
    noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    if isinstance(self.scheduler, PNDMScheduler):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        fac = torch.sqrt(beta_prod_t)
        sample = pred_original_sample
    elif isinstance(self.scheduler, LMSDiscreteScheduler):
        sigma = self.scheduler.sigmas[index]
        sample = latents - sigma * noise_pred
    else:
        raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

    sample = 1 / 0.18215 * sample
    image = self.vae.decode(sample).sample
    image = (image / 2 + 0.5).clamp(0, 1)

    if use_cutouts:
        image = self.make_cutouts(image, num_cutouts)
    else:
        image = transforms.Resize(self.feature_extractor.size)(image)
    image = self.normalize(image)

    image_embeddings_clip = self.clip_model.get_image_features(image).float()
    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

    if use_cutouts:
        dists = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip)
        dists = dists.view([num_cutouts, sample.shape[0], -1])
        loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
    else:
        loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean() * clip_guidance_scale

    grads = -torch.autograd.grad(loss, latents)[0]

    if isinstance(self.scheduler, LMSDiscreteScheduler):
        latents = latents.detach() + grads * (sigma**2)
        noise_pred = noise_pred_original
    else:
        noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
    return noise_pred, latents

@torch.no_grad()
def __call__(
    self,
    prompt: Union[str, List[str]],
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    clip_guidance_scale: Optional[float] = 100,
    clip_prompt: Optional[Union[str, List[str]]] = None,
    num_cutouts: Optional[int] = 4,
    use_cutouts: Optional[bool] = True,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
):
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # get prompt text embeddings
    text_input = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

    if clip_guidance_scale > 0:
        if clip_prompt is not None:
            clip_text_input = self.tokenizer(
                clip_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
        else:
            clip_text_input = text_input.input_ids.to(self.device)
        text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
        text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # get the initial random noise unless the user supplied it

    # Unlike in other pipelines, latents need to be generated in the target device
    # for 1-to-1 results reproducibility with the CompVis implementation.
    # However this currently doesn't work in `mps`.
    latents_device = "cpu" if self.device.type == "mps" else self.device
    latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
    if latents is None:
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=latents_device,
        )
    else:
        if latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
    latents = latents.to(self.device)

    # set timesteps
    accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1

    self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
    if isinstance(self.scheduler, LMSDiscreteScheduler):
        latents = latents * self.scheduler.sigmas[0]

    for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[i]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform classifier free guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # perform clip guidance
        if clip_guidance_scale > 0:
            text_embeddings_for_guidance = (
                text_embeddings.chunk(2)[0] if do_classifier_free_guidance else text_embeddings
            )
            noise_pred, latents = self.cond_fn(
                latents,
                t,
                i,
                text_embeddings_for_guidance,
                noise_pred,
                text_embeddings_clip,
                clip_guidance_scale,
                num_cutouts,
                use_cutouts,
            )

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = self.scheduler.step(noise_pred, i, latents).prev_sample
        else:
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    if not return_dict:
        return (image, None)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    ''')
  patch_nsfw(settings['enable_nsfw_filter'])
  # Colab.clear()

  # Patch Stable Diffusion Pipelines
  print('Patching Diffusers') # PAGODA
  remove('/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py') # PAGODA
  wgeto('https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/replacements/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py', '/workspace/api/pipeline_stable_diffusion.py') # PAGODA
  copyfile(f'/workspace/api/pipeline_stable_diffusion.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py') # PAGODA

  remove('/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py') # PAGODA
  wgeto('https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/replacements/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py', '/workspace/api/pipeline_stable_diffusion_img2img.py') # PAGODA
  copyfile(f'/workspace/api/pipeline_stable_diffusion_img2img.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py') # PAGODA

  remove('/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py') # PAGODA
  wgeto('https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/replacements/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py', '/workspace/api/pipeline_stable_diffusion_inpaint.py') # PAGODA
  copyfile(f'/workspace/api/pipeline_stable_diffusion_inpaint.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py') # PAGODA
  # Colab.clear() # PAGODA

