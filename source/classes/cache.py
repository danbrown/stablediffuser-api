from .diffusion import Diffusion
from .cleaner import Cleaner
from .scheduler import Scheduler
from ..utils import manage_imports

class Cache:
  class Pipe:
    def __init__(self, settings):
      global pipe
      global pipetype
      try:
        if pipetype != settings["mode"] or pipe is None:
          self.make(settings)
      except NameError:
        self.make(settings)
      Scheduler.make(settings)
      self.pipe = pipe
      self.pipetype = settings_pipetype

    def forward(self, x, context=None, mask=None):

      import math
      from torch import einsum
      try:
        from einops import rearrange
      except ModuleNotFoundError:
        import subprocess
        print(subprocess.run(['pip','install','einops'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
        # !pip install einops
        from einops import rearrange
      import types
      from diffusers.models.attention import CrossAttention
      import torch
      batch_size, sequence_length, dim = x.shape

      h = self.heads

      q = self.to_q(x)
      context = context if context is not None else x
      k = self.to_k(context)
      v = self.to_v(context)
      del context, x

      q = self.reshape_heads_to_batch_dim(q)
      k = self.reshape_heads_to_batch_dim(k)
      v = self.reshape_heads_to_batch_dim(v)

      r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

      stats = torch.cuda.memory_stats(q.device)
      mem_total = torch.cuda.get_device_properties(0).total_memory
      mem_active = stats['active_bytes.all.current']
      mem_free = mem_total - mem_active

      mem_required = q.shape[0] * q.shape[1] * k.shape[1] * 4 * 2.5
      steps = 1

      if mem_required > mem_free:
          steps = 2**(math.ceil(math.log(mem_required / mem_free, 2)))

      slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
      for i in range(0, q.shape[1], slice_size):
          end = i + slice_size
          s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)
          s1 *= self.scale

          s2 = s1.softmax(dim=-1)
          del s1

          r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
          del s2

      del q, k, v

      r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
      del r1

      return self.to_out(r2)

    def optimize_attention(model):
        import types
        from diffusers.models.attention import CrossAttention
        for module in model.modules():
            if isinstance(module, CrossAttention):
                module.forward = types.MethodType(Cache.Pipe.forward, module)

    def make(settings):
      global pipe 
      global pipetype
      pipe = None
      Cleaner.clean_env()
      pipetype = settings['mode']
      pipe_library = manage_imports(pipetype)
      import os, subprocess, torch
      username, token = Diffusion.creds()
      subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], stdout=subprocess.DEVNULL)
      # left_of_pipe = subprocess.Popen(["echo", token], stdout=subprocess.PIPE)
      # right_of_pipe = subprocess.run(['huggingface-cli', 'login'], stdin=left_of_pipe.stdout, stdout=subprocess.PIPE).stdout.decode('utf-8')
      print('Making Pipe')
      if settings['mode'] == "CLIP GUIDED PROMPT":
        import torch
        from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
        from PIL import Image
        from transformers import CLIPFeatureExtractor, CLIPModel
        import CLIP_GUIDED
        def create_clip_guided_pipeline(
            model_id="CompVis/stable-diffusion-v1-4", clip_model_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", scheduler="plms", low_vram_patch=True
        ):
            if low_vram_patch:
              pipeline = StableDiffusionPipeline.from_pretrained(
                  model_id,
                  torch_dtype=torch.float16,
                  revision="fp16",
                  use_auth_token=token,
              )
              clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)
              feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id, torch_dtype=torch.float16)
            else:
              pipeline = StableDiffusionPipeline.from_pretrained(
                  model_id,
                  use_auth_token=token,
              )
              clip_model = CLIPModel.from_pretrained(clip_model_id)
              feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)

            if settings["scheduler"] == 'ddim clip sampled':
              schedulers = manage_imports('ddim')
            else:
              schedulers = manage_imports(settings["scheduler"])
            if settings["scheduler"] == 'default' or settings["scheduler"] == 'pndm':
              scheduler = schedulers(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               num_train_timesteps=1000, skip_prk_steps=True)
            elif settings["scheduler"] == 'k-lms':
              scheduler = schedulers(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                                      num_train_timesteps=1000)
            elif settings["scheduler"] == 'ddim':
              scheduler = schedulers(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            elif settings["scheduler"] == 'ddim clip sampled':
              scheduler = schedulers(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=True, set_alpha_to_one=False)
            

            guided_pipeline = CLIP_GUIDED.CLIPGuidedStableDiffusion(
                unet=pipeline.unet,
                vae=pipeline.vae,
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                scheduler=scheduler,
                clip_model=clip_model,
                feature_extractor=feature_extractor,
            )

            return guided_pipeline
        
        local_pipe = create_clip_guided_pipeline(settings["model_id"], settings["clip_model_id"], settings['scheduler'], settings["low_vram_patch"])
        if settings["vram_over_speed"]:
          local_pipe.enable_attention_slicing()
          Cache.Pipe.optimize_attention(local_pipe.unet)
        local_pipe = local_pipe.to("cuda")
      else:
        if settings["low_vram_patch"]:
          try:
            local_pipe = pipe_library.from_pretrained(settings['model_id'], revision="fp16", torch_dtype=torch.float16, use_auth_token=token).to("cuda")
          except OSError:
            local_pipe = pipe_library.from_pretrained(settings['model_id'], use_auth_token=token).to("cuda")
          if settings["mode"] == "PROMPT":
            del local_pipe.vae.encoder
        else:
          local_pipe = pipe_library.from_pretrained(settings['model_id'], use_auth_token=token).to("cuda")
        if settings["vram_over_speed"]:
          local_pipe.enable_attention_slicing()
          Cache.Pipe.optimize_attention(local_pipe.unet)
      pipe = local_pipe
      local_pipe = None
      Cleaner.clean_env()
