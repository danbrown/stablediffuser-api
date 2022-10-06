from io import BytesIO
import base64

from .colab import Colab
from ..utils import wgeto, manage_imports, patch_nsfw

class Diffusion:
  def img2img_init(settings):
    def preprocess(image):
      import numpy as np
      import torch
      import PIL.Image
      w, h = image.size
      w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
      image = image.resize((w, h), resample=PIL.Image.LANCZOS)
      image = np.array(image).astype(np.float32) / 255.0
      image = image[None].transpose(0, 3, 1, 2)
      image = torch.from_numpy(image)
      return 2.*image - 1.
    global pipe
    import PIL.Image
    import requests
    if 'http' in settings['init_image']:
      response = requests.get(settings['init_image'])
      init_image = PIL.Image.open(BytesIO(response.content))
    elif 'data:image' in settings['init_image']:
      init_image = PIL.Image.open(BytesIO(base64.b64decode(settings['init_image'].split(',')[1])))
    elif settings['init_image'] is None or settings['init_image'] == "":
      init_image = Colab.Images.Painter.img2img(settings['width'], settings['height'])
    else:
      #it's a file
      init_image = PIL.Image.open(settings['init_image'])
    print("Init Image (automatically resized to match user input)")
    init_image = init_image.resize((settings['width'], settings['height']))
    # display(init_image)
    init_image = preprocess(init_image.convert("RGB"))
    return init_image
  
  def inpaint_init(settings):
    import PIL.Image
    def download(location):
      from io import BytesIO
      import requests
      import PIL.Image
      response = requests.get(location)
      return PIL.Image.open(BytesIO(response.content))
    if 'http' in settings["inpaint_image"]:
      init_image = download(settings["inpaint_image"])
    else:
      init_image = PIL.Image.open(settings["inpaint_image"])
    init_image = init_image.resize((settings['width'], settings['height']))
    if 'http' in settings["mask_image"]:
      mask_image = download(settings["mask_image"])
    elif settings["mask_image"]:
      mask_image = PIL.Image.open(settings["mask_image"])
    else:
      init_image.save("init.jpg")
      mask_image = Colab.Images.Painter.inpaint(settings['width'], settings['height'])
    mask_image.resize((settings['width'], settings['height']))
    return init_image, mask_image

