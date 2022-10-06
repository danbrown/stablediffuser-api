from pydantic import BaseModel
import base64
from io import BytesIO

from classes.cleaner import Cleaner
from classes.manager import Manager
from classes.runner import Runner

# class for a generation request
class GenerationRequest(BaseModel):
  mode: str = "PROMPT" # ["PROMPT","IMG2IMG","Inpainting","PROMPT FILE"]
  seed: int = 0 
  scheduler: str = "default" # ["default", "pndm", "k-lms", "ddim]
  precision: str = "autocast" # ["full","autocast"]
  width: int = 512
  height: int = 512
  scale: float = 13.8
  prompt_type: str = "TEXT" # ["TEXT","FILE"]
  text_prompt: str = "A robot on fire"
  negative_text_prompt: str = ""
  prompt_file: str = ""
  init_strength: float = 0.5
  init_image: str = ""
  inpaint_image: str = ""
  mask_image: str = ""
  inpaint_strength: float = 0.5
  steps: int = 20
  image_upscaler: str = "CodeFormer + Enhanced ESRGAN" # ["None","GFPGAN","Enhanced Real-ESRGAN", "GFPGAN + Enhanced ESRGAN", "CodeFormer", "CodeFormer + Enhanced ESRGAN"]
  upscale_amount: int = 2
  codeformer_fidelity: float = 0.8
  sharpen_amount: int = 1
  model_id: str = "CompVis/stable-diffusion-v1-4" # ["CompVis/stable-diffusion-v1-4", "CompVis/stable-diffusion-v1-3","CompVis/stable-diffusion-v1-2","CompVis/stable-diffusion-v1-1",'hakurei/waifu-diffusion']
  diffusers_version: str = "latest"
  clean_iters: int = 3
  bulky_skip: bool = False
  keep_seed: bool = False
  num_iters: int = 1
  run_forever: bool = False
  save_prompt_details: bool = True
  use_drive_for_pics: bool = False
  drive_pic_dir: str = "AI_PICS"
  delete_originals: bool = True
  low_vram_patch: bool = True
  vram_over_speed: bool = True
  enable_nsfw_filter: bool = False
  variations_batch: bool = False
  variations_batch_increase: float = 0.05
  variations_batch_init: float = 0.25
  variations_batch_max: float = 1.0

# Function to resume generation
def generate_to_base64(parameters):
    global last_model
    Cleaner.clean_env()
    manager = Manager()
    # manager.eval_settings()

    # replace settings with request available values
    current_settings = manager.colab.settings
    current_settings.update(parameters)

    manager.eval_settings(current_settings)

    # make images
    last_model = current_settings["model_id"]
    collected = []
    
    if current_settings['variations_batch'] and current_settings['mode'] == 'IMG2IMG':
      batch_increase = current_settings['variations_batch_increase']
      batch_max = current_settings['variations_batch_max']
      batch_value = current_settings['variations_batch_init']
      while batch_value <= batch_max:
        current_settings["init_strength"] = batch_value
        if not batch_value == current_settings["variations_batch_init"]:
          current_settings["save_prompt_details"] = False
        collected_round = Runner.run(current_settings) # array of PIL images
        collected.extend(collected_round)
        print(f"Batch {batch_value} done")
        batch_value += batch_increase
      print("Done with batch")
    else:
      collected = Runner.run(current_settings) # array of PIL images
    
    results_base64 = []
    print('collected')
    print(collected)
    for result in collected:
      buffered = BytesIO()
      result["image"].save(buffered, format="JPEG")
      # img_str = buffered.getvalue()
      img_str = base64.b64encode(buffered.getvalue())
      result["image"].close()
      # result["image"] = img_str
      result["image"] = 'data:image/jpeg;base64,' + img_str.decode('utf-8')
      results_base64.append(result)
    return results_base64