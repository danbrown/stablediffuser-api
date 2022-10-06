import os

from .diffusion import Diffusion
from .upscalers import Upscalers
from .cleaner import Cleaner
from .cache import Cache
from .scheduler import Scheduler
from ..utils import manage_imports, hugginface_credentials
from ..config import BASE_PATH
class Runner:
  def run(pipe, settings):
    collected_results = [] # PAGODA

    def sharpen_mage(image, samples=1):
      from PIL import ImageFilter
      im = image
      for i in range(samples):
          im = im.filter(ImageFilter.SHARPEN)
      return im

    import time
    torch, precision_scope, randint, sys = Runner.get_general_imports(settings)
    with torch.no_grad():
      with precision_scope("cuda"):
        if settings['seed'] == 0:
          settings['seed'] = randint(0,sys.maxsize)
        generator = torch.Generator("cuda").manual_seed(settings['seed'])
        counter = 1
        clean_counter = 0
        running = True
   
        outdir = f'{BASE_PATH}/diffusers_output'
        if not os.path.exists(outdir):
          os.makedirs(outdir)

        epoch_time = int(time.time())
        if settings["save_prompt_details"]:
          with open(f'{outdir}/{epoch_time}_prompt.json', 'w') as file:
            import json
            file.write(json.dumps(settings, indent=2))
        if settings['mode'] == "IMG2IMG":
          init_image = Diffusion.img2img_init(settings)
        elif settings['mode'] == 'Inpainting':
          init_image, mask_image = Diffusion.inpaint_init(settings)
        while running:

          collected_current = {} # PAGODA

          Cleaner.clean_env()
          if settings["mode"] == "PROMPT":
            if settings['prompt_type'] == 'TEXT':
              image = Runner.text_prompt(pipe=pipe, settings=settings, torch=torch, generator=generator)
            else:
              for prompt in settings["file_prompt"]:
                pass
                # TODO
          elif settings["mode"] == "IMG2IMG":
            image = Runner.img_to_img(pipe=pipe, settings=settings, torch=torch, generator=generator, init_image=init_image)
          elif settings["mode"] == "Inpainting":
            image = Runner.inpainting(pipe=pipe, settings=settings, torch=torch, generator=generator, init_image=init_image, mask_image=mask_image)
          elif settings["mode"] == "CLIP GUIDED PROMPT":
            Cleaner.clean_env()
            image = Runner.clip_guided_prompt(pipe=pipe, settings=settings, torch=torch, generator=generator)
            Cleaner.clean_env()
          if settings['image_upscaler'] in ['None','IMG2IMG'] or not settings["delete_originals"]:
            image.save(f'{outdir}/{epoch_time}_seed_{settings["seed"]}_original.png')
            epoch_time = int(time.time())
          clean_counter += 1
          if settings["clean_iters"] <= clean_counter:
            # Colab.clear()
            clean_counter = 0
          print(f'Image {counter}. SEED: {settings["seed"]}')
          # display(image)
          # Collect results
          if not settings["delete_originals"]: # PAGODA
            collected_current = { # PAGODA
              "image": image, # PAGODA
              "seed": settings["seed"], # PAGODA
              "index": counter, # PAGODA
              "epoch_time": epoch_time # PAGODA
            } # PAGODA

          print('Enhancing and Upscaling')
          if settings['image_upscaler'] != 'None':
            image = Upscalers.check_upscalers(settings,image)
            if settings['image_upscaler'] == 'IMG2IMG':
              image = image.resize((settings['width']*settings["upscale_amount"], settings['height']*settings["upscale_amount"]))
            if settings['sharpen_amount'] > 0:
              image = sharpen_mage(image, settings['sharpen_amount'])
              if not settings["bulky_skip"]:
                # display(image)
                # Collect results
                collected_current = { # PAGODA
                  "image": image, # PAGODA
                  "seed": settings["seed"], # PAGODA
                  "index": counter, # PAGODA
                  "epoch_time": epoch_time # PAGODA
                } # PAGODA
              image.save(f'{outdir}/{epoch_time}_seed_{settings["seed"]}_upscaled_{settings["upscale_amount"]}_sharpened_{settings["sharpen_amount"]}.png')
              epoch_time = int(time.time())
            else:
              if not settings["bulky_skip"]:
                # display(image)
                # Collect results
                collected_current = { # PAGODA
                  "image": image, # PAGODA
                  "seed": settings["seed"], # PAGODA
                  "index": counter, # PAGODA
                  "epoch_time": epoch_time # PAGODA
                } # PAGODA
              image.save(f'{outdir}/{epoch_time}_seed_{settings["seed"]}_upscaled_{settings["upscale_amount"]}.png')
              epoch_time = int(time.time())
          if settings["img2img_postprocess"] or settings['image_upscaler'] == 'IMG2IMG':
            if settings['image_upscaler'] == 'IMG2IMG':
              image = image.resize((settings['width']*settings["upscale_amount"], settings['height']*settings["upscale_amount"]))
            image = Runner.img2img_postprocess(pipe=pipe, settings=settings, image=image, generator=generator)
            if not settings["bulky_skip"]:
              # display(image)
              # Collect results
              collected_current = { # PAGODA
                "image": image, # PAGODA
                "seed": settings["seed"], # PAGODA
                "index": counter, # PAGODA
                "epoch_time": epoch_time # PAGODA
              } # PAGODA
            image.save(f'{outdir}/{epoch_time}_seed_{settings["seed"]}_upscaled_{settings["upscale_amount"]}_img2imgenhanced.png')
          if not settings['keep_seed']:
            settings['seed'] += 1
            generator = torch.Generator("cuda").manual_seed(settings['seed'])

          if not settings['run_forever']:
            if counter >= settings['num_iters']:
              running = False
              
          collected_results.append(collected_current) # PAGODA
          counter += 1
    return collected_results # PAGODA    

  def img2img_postprocess(self, pipe, settings, image, generator):
    import torch
    print("running img2img postprocessing. Switching to img2img pipe")
    pipe = None
    Cleaner.clean_env()
    pipe_library = manage_imports("IMG2IMG")
    import os, subprocess, torch
    username, token = hugginface_credentials()
    subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], stdout=subprocess.DEVNULL)
    # left_of_pipe = subprocess.Popen(["echo", token], stdout=subprocess.PIPE)
    # right_of_pipe = subprocess.run(['huggingface-cli', 'login'], stdin=left_of_pipe.stdout, stdout=subprocess.PIPE).stdout.decode('utf-8')
    if settings["low_vram_patch"]:
      try:
        local_pipe = pipe_library.from_pretrained(settings['model_id'], revision="fp16", torch_dtype=torch.float16, use_auth_token=token).to("cuda")
      except OSError:
        local_pipe = pipe_library.from_pretrained(settings['model_id'], use_auth_token=token).to("cuda")
    else:
      local_pipe = pipe_library.from_pretrained(settings['model_id'], use_auth_token=token).to("cuda")
    if settings["vram_over_speed"]:
      local_pipe.enable_attention_slicing()
      Cache.Pipe.optimize_attention(local_pipe.unet)
      
    pipe = local_pipe
    local_pipe = None
    del local_pipe
    Cleaner.clean_env()
    scheduler = manage_imports(settings["scheduler"])
    if settings["vram_over_speed"]:
      pipe.enable_attention_slicing()
      Cache.Pipe.optimize_attention(pipe.unet)
    if settings["scheduler"] == 'default' or settings["scheduler"] == 'pndm':
      pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True)
    elif settings["scheduler"] == 'k-lms':
      pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    elif settings["scheduler"] == 'ddim':
      pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    img2img_settings = {
        "text_prompt":settings['img2img']["text_prompt"],
        'init_strength':settings['img2img']["init_strength"],
        'scheduler':settings['scheduler'],
        'scale':settings['scale'],
        'steps':settings['img2img']["steps"]
    }
    image = Runner.img_to_img(pipe=pipe, settings=img2img_settings, torch=torch, generator=generator, image=image)
    pipe = None
    del pipe
    print('Switching back to old pipe and then displaying the image')
    Cleaner.clean_env()
    Cache.Pipe.make(settings)
    Scheduler.make(settings)
    return image

  def get_general_imports(settings):
    torch, randint, sys = manage_imports('general_diffusion_run')
    if settings['precision'] == 'autocast':
      return torch, torch.autocast, randint, sys
    else:
      from contextlib import nullcontext
      return torch, nullcontext, randint, sys

  def text_prompt(pipe, settings, torch, generator):
    if settings['scheduler'] == 'ddim':
      image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], width=settings['width'], height=settings['height'], guidance_scale=settings['scale'], eta=settings["ddim_eta"], generator=generator)
    else:
      image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], width=settings['width'], height=settings['height'], guidance_scale=settings['scale'], generator=generator)
    return image["sample"][0]
    

  def file_prompt(pipe, settings, torch, generator):
    pass

  def img_to_img(pipe, settings, torch, generator, init_image):
    if settings['scheduler'] == 'ddim':
      image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, strength=settings['init_strength'], eta=settings["ddim_eta"], guidance_scale=settings['scale'], generator=generator)["sample"][0]
    else:
      image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, strength=settings['init_strength'], guidance_scale=settings['scale'], generator=generator)["sample"][0]
    return image

  def inpainting(pipe, settings, torch, generator, init_image, mask_image):
    if settings['scheduler'] == 'ddim':
      image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, mask_image=mask_image, strength=settings["inpaint_strength"], eta=settings["ddim_eta"], guidance_scale=settings['scale'], generator=generator)["sample"][0]
    else:
      image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, mask_image=mask_image, strength=settings["inpaint_strength"], guidance_scale=settings['scale'], generator=generator)["sample"][0]
    return image

  def clip_guided_prompt(pipe, settings, torch, generator):
    if settings["unfreeze_unet"] == "True":
      pipe.unfreeze_unet()
    else:
      pipe.freeze_unet()

    if settings["unfreeze_vae"] == "True":
      pipe.unfreeze_vae()
    else:
      pipe.freeze_vae()
    use_cutouts = False
    if settings["clip_cutouts"] >= 1:
      use_cutouts = True
    Cleaner.clean_env()
    image = pipe(
        settings["text_prompt"],
        clip_prompt=settings["clip_prompt"] if settings["clip_prompt"].strip() != "" else None,
        negative_prompt=settings['negative_text_prompt'], 
        num_inference_steps=settings["steps"],
        guidance_scale=settings["scale"], 
        clip_guidance_scale=settings["clip_guidance_scale"],
        num_cutouts=settings["clip_cutouts"],
        use_cutouts=use_cutouts == "True",
        generator=generator,
        width=settings["width"],
        height=settings["height"]
    ).images[0]
    return image
