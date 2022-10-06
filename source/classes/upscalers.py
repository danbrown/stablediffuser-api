from .cleaner import Cleaner
from ..config import MODULES_PATH

class Upscalers:
  def check_upscalers(settings,image):
    Cleaner.clean_env()
    if settings['image_upscaler'] == 'GFPGAN':
      image = Upscalers.gfpgan(settings, image)
    elif settings['image_upscaler'] == 'Enhanced Real-ESRGAN':
      image = Upscalers.esrgan(settings, image)
    elif settings['image_upscaler'] == 'GFPGAN + Enhanced ESRGAN':
      image = Upscalers.gfpgan_esrgan(settings, image)
    elif settings['image_upscaler'] == 'CodeFormer':
      image = Upscalers.codeformer(settings, image)
    elif settings['image_upscaler'] == 'CodeFormer + Enhanced ESRGAN':
      image = Upscalers.codeformer_esrgan(settings, image)
    Cleaner.clean_env()
    return image

  def gfpgan(settings, image):
    import os, subprocess
    if not os.path.exists(f'{MODULES_PATH}/GFPGAN/'):
      Upscalers.Install.gfpgan()
    os.chdir(f'{MODULES_PATH}/GFPGAN/')
    print(subprocess.run(['mkdir', f'{MODULES_PATH}/GFPGAN/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['mkdir', f'{MODULES_PATH}/GFPGAN/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    image.save(f'{MODULES_PATH}/GFPGAN/temp/temp.png')
    print(subprocess.run(['python',f'{MODULES_PATH}/GFPGAN/inference_gfpgan.py','-i', f'{MODULES_PATH}/GFPGAN/temp/temp.png','-o',f'{MODULES_PATH}/GFPGAN/results/', '-w', f'{settings["codeformer_fidelity"]}','-s',f'{settings["upscale_amount"]}', '--bg_upsampler', 'realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    import PIL.Image
    image = PIL.Image.open(f'{MODULES_PATH}/GFPGAN/results/restored_imgs/temp.png')
    print(subprocess.run(['rm', '-rf', f'{MODULES_PATH}/GFPGAN/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['rm', '-rf', f'{MODULES_PATH}/GFPGAN/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    os.chdir(f'{MODULES_PATH}/')
    return image

  def esrgan(settings, image):
    def closest_value(input_list, input_value):
      difference = lambda input_list : abs(input_list - input_value)
      res = min(input_list, key=difference)
      return res
    import os, subprocess
    if int(settings["upscale_amount"]) not in [1,2,4,8]:
      nearest_value = closest_value([2,4,8],settings["upscale_amount"])
      settings["upscale_amount"] = nearest_value
      print(f'For Real-ESRGAN upscaling only 2, 4, and 8 are supported. Choosing the nearest Value: {nearest_value}')
    if not os.path.exists(f'{MODULES_PATH}/realesrgan'):
      # create a folder for the model
      print(subprocess.run(['wget',f'https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x2.pth','-O',f'{MODULES_PATH}/realesrgan/weights/RealESRGAN_x2.pth'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x2.pth -O {MODULES_PATH}/realesrgan/weights/RealESRGAN_x2.pth
      print(subprocess.run(['wget',f'https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x4.pth','-O',f'{MODULES_PATH}/realesrgan/weights/RealESRGAN_x4.pth'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x4.pth -O {MODULES_PATH}/realesrgan/weights/RealESRGAN_x4.pth
      print(subprocess.run(['wget',f'https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x8.pth','-O',f'{MODULES_PATH}/realesrgan/weights/RealESRGAN_x8.pth'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x8.pth -O {MODULES_PATH}/realesrgan/weights/RealESRGAN_x8.pth
    
    # transform the realesrgan into a python module
    import os, subprocess
    print(subprocess.run(['touch', f'{MODULES_PATH}/realesrgan/__init__.py'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    # print(subprocess.run(['touch', f'{MODULES_PATH}/__init__.py'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    
    # import the realesrgan module
    from ...modules.realesrgan import RealESRGAN

    import torch
    model = RealESRGAN(torch.device('cuda'), scale = settings["upscale_amount"])
    model.load_weights(f'{MODULES_PATH}/realesrgan/weights/RealESRGAN_x{settings["upscale_amount"]}.pth')
    import numpy as np
    image = model.predict(np.array(image))
    os.chdir(f'{MODULES_PATH}/')
    model = None
    Cleaner.clean_env()
    return image

  def codeformer(settings, image):
    import os, subprocess
    if not os.path.exists(f'{MODULES_PATH}/CodeFormer/'):
      Upscalers.Install.codeformer()
    print(subprocess.run(['mkdir', f'{MODULES_PATH}/CodeFormer/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['mkdir', f'{MODULES_PATH}/CodeFormer/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    image.save(f'{MODULES_PATH}/CodeFormer/temp/temp.png')
    os.chdir(f'{MODULES_PATH}/CodeFormer/')
    print(subprocess.run(['python',f'inference_codeformer.py','--w', f'{settings["codeformer_fidelity"]}','--test_path',f'{MODULES_PATH}/CodeFormer/temp','--upscale',f'{settings["upscale_amount"]}', '--bg_upsampler', 'realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    import PIL.Image
    image = PIL.Image.open(f'{MODULES_PATH}/CodeFormer/results/temp_{settings["codeformer_fidelity"]}/final_results/temp.png')
    print(subprocess.run(['rm', '-rf', f'{MODULES_PATH}/CodeFormer/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['rm', '-rf', f'{MODULES_PATH}/CodeFormer/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    os.chdir(f'{MODULES_PATH}/')
    return image

  def gfpgan_esrgan(settings, image):
    orig_upscale = settings['upscale_amount']
    settings['upscale_amount'] = 1
    image = Upscalers.gfpgan(settings, image)
    settings['upscale_amount'] = orig_upscale
    image = Upscalers.esrgan(settings, image)
    return image

  def codeformer_esrgan(settings, image):
    orig_upscale = settings['upscale_amount']
    settings['upscale_amount'] = 1
    image = Upscalers.codeformer(settings, image)
    settings['upscale_amount'] = orig_upscale
    image = Upscalers.esrgan(settings, image)
    return image
    

  class Install:
    def gfpgan():
      import subprocess
      print(subprocess.run(['git','clone','https://github.com/TencentARC/GFPGAN.git'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','basicsr'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','facexlib'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      import os
      os.chdir(f'{MODULES_PATH}/GFPGAN')
      print(subprocess.run(['pip','install', '-r', 'requirements.txt'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['python','setup.py', 'develop'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['wget','https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', '-P', f'{MODULES_PATH}/GFPGAN/experiments/pretrained_models'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir(f'{MODULES_PATH}/')

    def esrgan():
      import subprocess, os
      print(subprocess.run(['git','clone','https://github.com/sberbank-ai/Real-ESRGAN', f'{MODULES_PATH}/realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir(f'{MODULES_PATH}/realesrgan')
      print(subprocess.run(['git','reset', '--hard','2a5afd04a0e43956d1640db00d3a528ca5972fd2'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','-r',f'{MODULES_PATH}/realesrgan/requirements.txt'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir(f'{MODULES_PATH}/')
      
    def codeformer():
      import subprocess
      import os
      print(subprocess.run(['git','clone','https://github.com/sczhou/CodeFormer.git'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','-r',f'{MODULES_PATH}/CodeFormer/requirements.txt'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir(f'{MODULES_PATH}/CodeFormer/')
      print(subprocess.run(['python','basicsr/setup.py','develop'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['python','scripts/download_pretrained_models.py','facelib'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['python','scripts/download_pretrained_models.py','CodeFormer'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir(f'{MODULES_PATH}/')
