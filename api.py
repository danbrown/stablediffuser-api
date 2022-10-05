#@title Render Images

from symbol import parameters


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

class Cleaner:
  def clean_env():
    gc, torch = Manager.manage_imports('clean_env')
    gc.collect()
    torch.cuda.empty_cache()

class Colab:
  def __init__(self):
    self.settings = self.UserSettings.set_settings()

  def clear():
    from IPython.display import clear_output; clear_output()

  def manage_drive(drive_pic_dir):
    exists, mount, makedirs = Manager.manage_imports('manage_drive')
    if not exists('/workspace/api/drive'):
      mount('/workspace/api/drive')
    if not exists(f'/workspace/api/drive/MyDrive/{drive_pic_dir}'):
      makedirs(f'/workspace/api/drive/MyDrive/{drive_pic_dir}')

  class Images:
    def resize_image():
      pass

    def suggest_resolution():
      pass
      
    class Painter:
      def inpaint(width, height):
        import requests
        from io import BytesIO
        def draw(filename='drawing.png', color="white", w=256, h=256, line_width=50,loop=False, init_img="init.jpg"):
          filename="init.jpg"
          import google
          from IPython.display import HTML
          from base64 import b64decode
          import os
          import shutil
          import uuid
          COLAB_HTML_ROOT = "/usr/local/share/jupyter/nbextensions/google.colab/"

          def moveToExt(filename:str) -> str:
            if not os.path.exists(filename):
              print("Image file not found")
              return None
            
            target = os.path.basename(filename)
            target = os.path.join(COLAB_HTML_ROOT, str(uuid.uuid4()) + target)
            
            shutil.copyfile(filename,target)
            print("moved to ext")
            return target
          real_filename = os.path.realpath(filename)
          html_filename = real_filename
          html_real_filename = html_filename
          if os.path.exists(real_filename):
            html_real_filename = moveToExt(real_filename)
            html_filename = html_real_filename.replace("/usr/local/share/jupyter","")
            

          canvas_html = f"""
        <canvas width={w} height={h}></canvas>

        <div class="slidecontainer">
        <label for="lineWidth" id="lineWidthLabel">{line_width}px</label>
          <input type="range" min="1" max="100" value="1" class="slider" id="lineWidth">
        </div>

        <div>
          <button id="loadImage">Reload from disk</button>
          <button id="reset">Reset</button>
          <button id="save">Save</button>
        </div>
        <script>

        function loadImage(url) {{
        return new Promise(r => {{ let i = new Image(); i.onload = (() => r(i)); i.src = url; }});
        }}
          
          
          var canvas = document.querySelector('canvas')
          var ctx = canvas.getContext('2d')
          ctx.lineWidth = {line_width};
          
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = "{color}";


          var slider = document.getElementById("lineWidth");
          slider.oninput = function() {{
            ctx.lineWidth = this.value;
            lineWidthLabel.innerHTML = `${{this.value}}px`
          }}


          function updateStroke(event){{
              ctx.strokeStyle = event.target.value
          }}
          function updateBG(event){{
              ctx.fillStyle = event.target.value
          }}
          
          
          var clear_button = document.querySelector('#reset')
          var reload_img_button = document.querySelector('#loadImage')
          
          var button = document.querySelector('#save')

          var mouse = {{x: 0, y: 0}}
          canvas.addEventListener('mousemove', function(e) {{
            mouse.x = e.pageX - this.offsetLeft
            mouse.y = e.pageY - this.offsetTop
          }})
          canvas.onmousedown = ()=>{{
            ctx.beginPath()
            ctx.moveTo(mouse.x, mouse.y)
            canvas.addEventListener('mousemove', onPaint)
          }}
          canvas.onmouseup = ()=>{{
            canvas.removeEventListener('mousemove', onPaint)
          }}
          var onPaint = ()=>{{
            ctx.lineTo(mouse.x, mouse.y)
            ctx.stroke()
          }}
          reload_img_button.onclick = async ()=>{{
            console.log("Reloading Image {html_filename}")
            let img = await loadImage('{html_filename}'); 
            console.log("Loaded image")
            ctx.drawImage(img, 0, 0);

          }}
          reload_img_button.click()
        
          clear_button.onclick = ()=>{{
              console.log('Clearing Screen')
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.fillRect(0, 0, canvas.width, canvas.height);
            }}
            canvas.addEventListener('load', function() {{
            console.log('All assets are loaded')
          }})
          var data = new Promise(resolve=>{{
            button.onclick = ()=>{{

              var c = ctx
              var imageData = ctx.getImageData(0,0, {w}, {h});
              var pixel = imageData.data;
              var r=0, g=1, b=2,a=3;
            for (var p = 0; p<pixel.length; p+=4)
            {{
              if (
                  pixel[p+r] != 255 &&
                  pixel[p+g] != 255 &&
                  pixel[p+b] != 255) 
              {{pixel[p+r] =0; pixel[p+g]=0; pixel[p+b]=0}}
            }}

            c.putImageData(imageData,0,0);
              resolve(canvas.toDataURL('image/png'))
            }}
            
          }})
          
          
        </script>
        """
          print(HTML)
          # display(HTML(canvas_html))
          print("Evaluating JS")
          
          data = google.colab.output.eval_js("data")
          if data:
            print("Saving Sketch")  
            binary = b64decode(data.split(',')[1])
            # filename = html_real_filename if loop else filename
            with open("init_mask.png", 'wb') as f:
              f.write(binary)
            #return len(binary)



        draw(filename = "init_mask.png", w=width, h=height)
        import PIL.Image
        return PIL.Image.open('init_mask.png')
      def img2img(width, height):
        import os
        os.chdir('/workspace/api/')
        def draw(filename='drawing.png', color="black", bg_color="transparent",w=256, h=256, line_width=1,loop=False):
          import google
          from IPython.display import HTML
          from base64 import b64decode
          import os
          import shutil
          import uuid
          COLAB_HTML_ROOT = "/usr/local/share/jupyter/nbextensions/google.colab/"
          def moveToExt(filename:str) -> str:
            if not os.path.exists(filename):
              print("Image file not found")
              return None
            
            target = os.path.basename(filename)
            target = os.path.join(COLAB_HTML_ROOT, str(uuid.uuid4()) + target)
            
            shutil.copyfile(filename,target)
            print("moved to ext")
            return target
          real_filename = os.path.realpath(filename)
          html_filename = real_filename
          html_real_filename = html_filename
          if os.path.exists(real_filename):
            html_real_filename = moveToExt(real_filename)
            html_filename = html_real_filename.replace("/usr/local/share/jupyter","")
            

          canvas_html = f"""
        <canvas width={w} height={h}></canvas>
        <div>
          <label for="strokeColor">Stroke</label>
          <input type="color" value="{color}" id="strokeColor">
        
          <label for="bgColor">Background</label>
          <input type="color" value="{bg_color}" id="bgColor">
        </div>
        <div class="slidecontainer">
        <label for="lineWidth" id="lineWidthLabel">{line_width}px</label>
          <input type="range" min="1" max="35" value="1" class="slider" id="lineWidth">
        </div>

        <div>
          <button id="loadImage">Reload from disk</button>
          <button id="reset">Reset</button>
          <button id="save">Save</button>
          <button id="exit">Exit</button>
        </div>
        <script>

        function loadImage(url) {{
        return new Promise(r => {{ let i = new Image(); i.onload = (() => r(i)); i.src = url; }});
      }}
          
          
          var canvas = document.querySelector('canvas')
          var ctx = canvas.getContext('2d')
          ctx.lineWidth = {line_width}
          ctx.fillStyle = "{bg_color}";
          
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = "{color}";

          var strokeColor = document.querySelector('#strokeColor')
          var bgColor = document.querySelector('#bgColor')

          var slider = document.getElementById("lineWidth");
          slider.oninput = function() {{
            ctx.lineWidth = this.value;
            lineWidthLabel.innerHTML = `${{this.value}}px`
          }}

          function updateStroke(event){{
              ctx.strokeStyle = event.target.value
          }}
          function updateBG(event){{
              ctx.fillStyle = event.target.value
          }}
          
          bgColor.addEventListener("change", updateBG, false);
          strokeColor.addEventListener("change", updateStroke, false);
          
          var clear_button = document.querySelector('#reset')
          var reload_img_button = document.querySelector('#loadImage')
          var button = document.querySelector('#save')
          var exit_button = document.querySelector('#exit')

          var mouse = {{x: 0, y: 0}}
          canvas.addEventListener('mousemove', function(e) {{
            mouse.x = e.pageX - this.offsetLeft
            mouse.y = e.pageY - this.offsetTop
          }})
          canvas.onmousedown = ()=>{{
            ctx.beginPath()
            ctx.moveTo(mouse.x, mouse.y)
            canvas.addEventListener('mousemove', onPaint)
          }}
          canvas.onmouseup = ()=>{{
            canvas.removeEventListener('mousemove', onPaint)
          }}
          var onPaint = ()=>{{
            ctx.lineTo(mouse.x, mouse.y)
            ctx.stroke()
          }}
          reload_img_button.onclick = async ()=>{{
            console.log("Reloading Image {html_filename}")
            let img = await loadImage('{html_filename}'); 
            console.log("Loaded image")
            ctx.drawImage(img, 0, 0);

          }}
          
          clear_button.onclick = ()=>{{
              console.log('Clearing Screen')
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.fillRect(0, 0, canvas.width, canvas.height);
            }}
            canvas.addEventListener('load', function() {{
            console.log('All assets are loaded')
          }})
          var data = new Promise(resolve=>{{
            button.onclick = ()=>{{
              resolve(canvas.toDataURL('image/png'))
            }}
            exit_button.onclick = ()=>{{
            resolve()
          }}
            
          }})
          
          // window.onload = async ()=>{{
          //   console.log("loaded")
          //   let img = await loadImage('{html_filename}');  
          //   ctx.drawImage(img, 0, 0);
          // }}
          
          
        </script>
        """
          print(HTML)
          # display(HTML(canvas_html))
          print("Evaluating JS")
          
          data = google.colab.output.eval_js("data")
          if data:
            print("Saving Sketch")  
            binary = b64decode(data.split(',')[1])
            # filename = html_real_filename if loop else filename
            with open(filename, 'wb') as f:
              f.write(binary)
            #return len(binary)
        
        draw(filename = "custom_image.png", w=width, h=height, bg_color="blue", line_width=10)
        import PIL.Image
        return PIL.Image.open("/workspace/api/custom_image.png")

  class UserSettings:

    def set_settings():
      MODE = "PROMPT" #@param ["PROMPT", "CLIP GUIDED PROMPT", "IMG2IMG","Inpainting","PROMPT FILE"]
      #@markdown `MODE` Select what mode you want to use <br>

      #@markdown ---
      settings = {"mode":MODE}
      #@markdown GENERAL SETTINGS

      WIDTH = 512 #@param {type:"slider", min:256, max:4096, step:64}
      HEIGHT = 512 #@param {type:"slider", min:256, max:4096, step:64}
      SCALE = 13.8 #@param {type:"slider", min:0, max:25, step:0.1}
      SEED = 0 #@param {type:'integer'}
      settings["seed"] = SEED 
      SCHEDULER = 'default' #@param ["default", "pndm", "k-lms", "ddim", 'ddim clip sampled']
      settings["scheduler"] = SCHEDULER 

      if SCHEDULER == 'ddim':
        DDIM_ETA = 0.72 #@param {type:"slider", min:0, max:1, step:0.01} 
        settings["ddim_eta"] = DDIM_ETA 
      PRECISION = "autocast" #@param ["full","autocast"]
      settings["precision"] = PRECISION
      settings["width"] = WIDTH
      settings["height"] = HEIGHT
      settings["scale"] = SCALE
      IMG2IMG_POSTPROCESS = False #@param {type:'boolean'}
      #@markdown `IMG2IMG_POSTPROCESS`: Postprocess the image with img2img after it's done. Will take the img2img settings to do the postprocessing, so make sure to change those if you set this. It can add a lot of detail to the final image after upscaling (sometimes it's hit or miss) although it is very slow since it needs to switch pipes<br>CAUTION: Very error-prone (will fix that down the road). If this runs into an error you need to clean the environment by clicking on "Runtime" and then on "Restart Environment"
      settings["img2img_postprocess"] = IMG2IMG_POSTPROCESS
      
      #@markdown ---

      #@markdown UPSCALING SETTINGS

      IMAGE_UPSCALER = "GFPGAN + Enhanced ESRGAN" #@param ["None","GFPGAN","Enhanced Real-ESRGAN", "GFPGAN + Enhanced ESRGAN", "CodeFormer", "CodeFormer + Enhanced ESRGAN", "IMG2IMG"]
      settings["image_upscaler"] = IMAGE_UPSCALER 

      UPSCALE_AMOUNT = 2 #@param {type:"raw"}
      settings["upscale_amount"] = UPSCALE_AMOUNT 

      FIDELITY = 0.8 #@param {type:"slider", min:0, max:1, step:0.01}
      settings["codeformer_fidelity"] = FIDELITY

      SHARPEN_AMOUNT = 1 #@param{type:'slider', min:0, max:3, step:1}
      settings["sharpen_amount"] = SHARPEN_AMOUNT

      
      #@markdown ---

      #@markdown MODE: PROMPT FILE SETTINGS
      if MODE == "PROMPT FILE":
        FILE_LOCATION = "/workspace/api/diffusers_output/1663720628_prompt.json" #@param {type:"string"}
        settings['prompt_file'] = FILE_LOCATION

      #@markdown ---

      if MODE == "PROMPT":
        #@markdown MODE: PROMPT SETTINGS
        # PROMPT_TYPE = "TEXT" #@param ["TEXT","FILE"]
        # TODO
        settings["prompt_type"] = "TEXT"

        if settings["prompt_type"] == "TEXT":
          TEXT_PROMPT = "A young woman wearing a hat, greg rutkowski, artgerm, trending on artstation, cinematic animation still, by lois van baarle, ilya kuvshinov, metahuman" #@param {type:"string"}
          settings["text_prompt"] = TEXT_PROMPT
        # elif PROMPT_TYPE == "FILE":
        #   FILE_PROMPT = "/workspace/api/prompt_file.txt" #@param {type:"string"}
        #   prompts = []
        #   with open(FILE_PROMPT, 'r') as file:
        #     for line in file.readlines():
        #       prompts.append(line)
        #   settings["file_prompt"] = prompts
        # TODO

        PROMPT_STEPS = 200 #@param {type:"slider", min:5, max:500, step:5} 
        settings["steps"] = PROMPT_STEPS 

      #@markdown ---

      elif MODE == "CLIP GUIDED PROMPT":
        
        #@markdown MODE: CLIP GUIDED PROMPT (still buggy and finicky, but works with autocast & low_vram_patch)<br>
        #@markdown The VRAM tends to stick if this errors out. When that happens click on "Runtime" and then "Restart Runtime". It's very finicky, some settings work better than others.
        CLIP_MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" #@param ["laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K", "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"] {allow-input: true}
        settings["clip_model_id"] = CLIP_MODEL_ID
        CLIP_MODE_TEXT_PROMPT = "A young woman wearing a hat, greg rutkowski, artgerm, trending on artstation, cinematic animation still, by lois van baarle, ilya kuvshinov, metahuman" #@param {type:"string"}
        settings["text_prompt"] = CLIP_MODE_TEXT_PROMPT
        CLIP_GUIDANCE_PROMPT = "" #@param {type:"string"}
        settings["clip_prompt"] = CLIP_GUIDANCE_PROMPT
        CLIP_MODE_STEPS = 200 #@param {type:"integer"}
        settings["steps"] = CLIP_MODE_STEPS
        CLIP_MODE_SCALE = 13.7 #@param {type:"raw"}
        settings["scale"] = CLIP_MODE_SCALE
        CLIP_GUIDANCE_SCALE = 100 #@param {type:"raw"}
        settings["clip_guidance_scale"] = CLIP_GUIDANCE_SCALE
        CLIP_MODE_NUM_CUTOUTS = 64 #@param {type:"raw"}
        settings["clip_cutouts"] = CLIP_MODE_NUM_CUTOUTS
        CLIP_UNFREEZE_UNET = True #@param {type:"boolean"}
        settings["unfreeze_unet"] = CLIP_UNFREEZE_UNET
        CLIP_UNFREEZE_VAE = True #@param {type:"boolean"}
        settings["unfreeze_vae"] = CLIP_UNFREEZE_VAE

    
      elif MODE == "Inpainting":
        #@markdown ---
        
        #@markdown MODE: Inpainting SETTINGS
        INPAINT_PROMPT = "A cat sitting on a bench" #@param {type:"string"}
        settings["text_prompt"] = INPAINT_PROMPT
        
        INPAINT_IMAGE = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" #@param {type:'string'}
        settings["inpaint_image"] = INPAINT_IMAGE
        
        MASK_IMAGE = "" #@param {type:'string'}
        settings["mask_image"] = MASK_IMAGE
        
        INPAINT_STRENGTH = 0.5 #@param {type:"slider", min:0, max:1, step:0.01} 
        settings["inpaint_strength"] = INPAINT_STRENGTH

        INPAINT_MODE_STEPS = 200 #@param {type:"slider", min:5, max:500, step:5} 
        settings["steps"] = INPAINT_MODE_STEPS


      if MODE == "IMG2IMG" or settings["img2img_postprocess"] or settings['image_upscaler'] == 'IMG2IMG':
        #@markdown ---
        
        #@markdown MODE: IMG2IMG SETTINGS
        IMG_PROMPT = "A young woman wearing a hat, greg rutkowski, artgerm, trending on artstation, cinematic animation still, by lois van baarle, ilya kuvshinov, metahuman" #@param {type:"string"}
        
        INIT_IMAGE = "https://raw.githubusercontent.com/dblunk88/txt2imghd/master/character_with_hat.jpg" #@param {type: 'string'}
        settings["init_image"] = INIT_IMAGE
        
        INIT_STRENGTH = 0.6 #@param{type:"slider", min:0.01, max:1, step:0.01}
        IMG2IMG_MODE_STEPS=40 #@param {type:"slider", min:5, max:500, step:5} 
        
        if settings["img2img_postprocess"] or settings['image_upscaler'] == 'IMG2IMG':
          if settings['mode'] == "IMG2IMG":
            settings["text_prompt"] = IMG_PROMPT
            settings["init_strength"] = INIT_STRENGTH
            settings["steps"] = IMG2IMG_MODE_STEPS
          settings['img2img'] = {}
          settings['img2img']["text_prompt"] = IMG_PROMPT
          settings['img2img']["init_strength"] = INIT_STRENGTH
          settings['img2img']["steps"] = IMG2IMG_MODE_STEPS 
        else:
          settings["text_prompt"] = IMG_PROMPT
          settings["init_strength"] = INIT_STRENGTH
          settings["steps"] = IMG2IMG_MODE_STEPS 


      #@markdown ---
      

      #@markdown Version Settings
      #https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
      MODEL_ID = "CompVis/stable-diffusion-v1-4" #@param ["CompVis/stable-diffusion-v1-4", "CompVis/stable-diffusion-v1-3","CompVis/stable-diffusion-v1-2","CompVis/stable-diffusion-v1-1",'hakurei/waifu-diffusion', 'ayan4m1/trinart_diffusers_v2', 'doohickey/trinart-waifu-diffusion-50-50', 'lambdalabs/sd-pokemon-diffusers', 'anton-l/ddpm-ema-pokemon-64'] {allow-input:true}
      settings["model_id"] = MODEL_ID

      DIFFUSERS_VERSION = '91db81894b44798649b6cf54be085c205e146805' #@param ["latest", "91db81894b44798649b6cf54be085c205e146805", "f3937bc8f3667772c9f1428b66f0c44b6087b04d"]
      settings["diffusers_version"] = DIFFUSERS_VERSION
      
      #@markdown ---
      
      #@markdown ADVANCED SETTINGS
      CLEAN_PREVIEW_AFTER_ITERS = 19 #@param {type:"slider", min:1, max:100, step:1} 
      settings["clean_iters"] = CLEAN_PREVIEW_AFTER_ITERS

      SKIP_BULKY_PREVIEWS = False #@param {type:'boolean'}
      settings["bulky_skip"] = SKIP_BULKY_PREVIEWS

      KEEP_SEED = False #@param {type:'boolean'}
      settings["keep_seed"] = KEEP_SEED

      NUM_ITERS = 4 #@param {type:"slider", min:1, max:100, step:1} 
      settings["num_iters"] = NUM_ITERS

      RUN_FOREVER = False #@param {type:"boolean"}
      settings["run_forever"] = RUN_FOREVER

      SAVE_PROMPT_DETAILS = True #@param {type:"boolean"}
      settings["save_prompt_details"] = SAVE_PROMPT_DETAILS

      USE_DRIVE_FOR_PICS = False #@param {type:"boolean"}
      settings["use_drive_for_pics"] = USE_DRIVE_FOR_PICS

      

      if USE_DRIVE_FOR_PICS:
        DRIVE_PIC_DIR = "AI_PICS" #@param {type:"string"}
        settings["drive_pic_dir"] = DRIVE_PIC_DIR

      DELETE_ORIGINALS = True #@param{type:'boolean'}
      settings["delete_originals"] = DELETE_ORIGINALS

      LOW_VRAM_PATCH = True #@param {type:"boolean"}
      settings["low_vram_patch"] = LOW_VRAM_PATCH

      VRAM_OVER_SPEED = True #@param {type:"boolean"}
      settings["vram_over_speed"] = VRAM_OVER_SPEED

      ENABLE_NSFW_FILTER = False #@param {type:"boolean"}
      settings["enable_nsfw_filter"] = ENABLE_NSFW_FILTER

  

      return settings

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
    if not os.path.exists('/workspace/api/GFPGAN/'):
      Upscalers.Install.gfpgan()
    os.chdir('/workspace/api/GFPGAN/')
    print(subprocess.run(['mkdir', f'/workspace/api/GFPGAN/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['mkdir', f'/workspace/api/GFPGAN/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    image.save('/workspace/api/GFPGAN/temp/temp.png')
    print(subprocess.run(['python',f'/workspace/api/GFPGAN/inference_gfpgan.py','-i', '/workspace/api/GFPGAN/temp/temp.png','-o','/workspace/api/GFPGAN/results/', '-w', f'{settings["codeformer_fidelity"]}','-s',f'{settings["upscale_amount"]}', '--bg_upsampler', 'realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    import PIL.Image
    image = PIL.Image.open(f'/workspace/api/GFPGAN/results/restored_imgs/temp.png')
    print(subprocess.run(['rm', '-rf', f'/workspace/api/GFPGAN/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['rm', '-rf', f'/workspace/api/GFPGAN/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    os.chdir('/workspace/api/')
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
    if not os.path.exists('/workspace/api/realesrgan'):
      # create a folder for the model
      print(subprocess.run(['wget',f'https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x2.pth','-O',f'/workspace/api/realesrgan/weights/RealESRGAN_x2.pth'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x2.pth -O /workspace/api/realesrgan/weights/RealESRGAN_x2.pth
      print(subprocess.run(['wget',f'https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x4.pth','-O',f'/workspace/api/realesrgan/weights/RealESRGAN_x4.pth'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x4.pth -O /workspace/api/realesrgan/weights/RealESRGAN_x4.pth
      print(subprocess.run(['wget',f'https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x8.pth','-O',f'/workspace/api/realesrgan/weights/RealESRGAN_x8.pth'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # wget https://huggingface.co/datasets/db88/Enhanced_ESRGAN/resolve/main/RealESRGAN_x8.pth -O /workspace/api/realesrgan/weights/RealESRGAN_x8.pth
    
    # transform the realesrgan into a python module
    import os, subprocess
    print(subprocess.run(['touch', f'/workspace/api/realesrgan/__init__.py'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    # print(subprocess.run(['touch', f'/workspace/api/__init__.py'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    
    # import the realesrgan module
    from realesrgan.realesrgan import RealESRGAN

    import torch
    model = RealESRGAN(torch.device('cuda'), scale = settings["upscale_amount"])
    model.load_weights(f'/workspace/api/realesrgan/weights/RealESRGAN_x{settings["upscale_amount"]}.pth')
    import numpy as np
    image = model.predict(np.array(image))
    os.chdir('/workspace/api/')
    model = None
    Cleaner.clean_env()
    return image

  def codeformer(settings, image):
    import os, subprocess
    if not os.path.exists('/workspace/api/CodeFormer/'):
      Upscalers.Install.codeformer()
    print(subprocess.run(['mkdir', f'/workspace/api/CodeFormer/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['mkdir', f'/workspace/api/CodeFormer/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    image.save('/workspace/api/CodeFormer/temp/temp.png')
    os.chdir('/workspace/api/CodeFormer/')
    print(subprocess.run(['python',f'inference_codeformer.py','--w', f'{settings["codeformer_fidelity"]}','--test_path','/workspace/api/CodeFormer/temp','--upscale',f'{settings["upscale_amount"]}', '--bg_upsampler', 'realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    import PIL.Image
    image = PIL.Image.open(f'/workspace/api/CodeFormer/results/temp_{settings["codeformer_fidelity"]}/final_results/temp.png')
    print(subprocess.run(['rm', '-rf', f'/workspace/api/CodeFormer/results/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    print(subprocess.run(['rm', '-rf', f'/workspace/api/CodeFormer/temp/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    os.chdir('/workspace/api/')
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
      os.chdir('/workspace/api/GFPGAN')
      print(subprocess.run(['pip','install', '-r', 'requirements.txt'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['python','setup.py', 'develop'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['wget','https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', '-P', '/workspace/api/GFPGAN/experiments/pretrained_models'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir('/workspace/api/')

    def esrgan():
      import subprocess, os
      print(subprocess.run(['git','clone','https://github.com/sberbank-ai/Real-ESRGAN', '/workspace/api/realesrgan'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir('/workspace/api/realesrgan')
      print(subprocess.run(['git','reset', '--hard','2a5afd04a0e43956d1640db00d3a528ca5972fd2'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','-r','/workspace/api/realesrgan/requirements.txt'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir('/workspace/api/')
      
    def codeformer():
      import subprocess
      import os
      print(subprocess.run(['git','clone','https://github.com/sczhou/CodeFormer.git'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['pip','install','-r','/workspace/api/CodeFormer/requirements.txt'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir('/workspace/api/CodeFormer/')
      print(subprocess.run(['python','basicsr/setup.py','develop'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['python','scripts/download_pretrained_models.py','facelib'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      print(subprocess.run(['python','scripts/download_pretrained_models.py','CodeFormer'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      os.chdir('/workspace/api/')

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
      Manager.Diffusion.Scheduler.make(settings)
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
      pipe_library = Manager.manage_imports(pipetype)
      import os, subprocess, torch
      username, token = Manager.Diffusion.creds()
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
              schedulers = Manager.manage_imports('ddim')
            else:
              schedulers = Manager.manage_imports(settings["scheduler"])
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

  

class Manager:
  def __init__(self):
    self.colab = Colab()
    self.cache = Cache()
    from IPython.display import Javascript
    # display(Javascript("google.colab.output.resizeIframeToContent()"))

  def manage_imports(requester):
    if requester == 'manage_drive':
      from os.path import exists
      from os import makedirs
      from google.colab import drive
      return exists, drive.mount, makedirs

    elif requester == 'patch_nsfw':
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



  def eval_settings(self, settings = False):
    if not settings:
      settings = self.colab.settings
    if settings['mode'] == "PROMPT FILE":
      with open(settings['prompt_file'],'r') as file:
        import json
        self.colab.settings = json.loads(file.read())
        settings = self.colab.settings
    import json
    print(json.dumps(settings, indent=2))
    global pipetype
    global pipe
    global last_model
    try:
      if pipetype != settings['mode'] or pipe is None or settings['model_id'] != last_model:
        Cache.Pipe.make(settings)
    except NameError:
      Cache.Pipe.make(settings)
    if settings["use_drive_for_pics"]:
      Colab.manage_drive(settings['drive_pic_dir'])
    

  class Diffusion:
    def patch_nsfw(ENABLE_NSFW_FILTER):
      copyfile, remove = Manager.manage_imports('patch_nsfw')
      remove('/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py')
      if ENABLE_NSFW_FILTER:
        copyfile(f'/workspace/api/safety_checker.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py')
      else:
        copyfile(f'/workspace/api/safety_checker_patched.py', '/venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py')

    def install_diffusers():
      subprocess, os, copyfile, remove = Manager.manage_imports('diffuser_install') # PAGODA
      settings = Colab.UserSettings.set_settings()
      
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
      username, token = Manager.Diffusion.creds()
      subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], stdout=subprocess.DEVNULL)
      # left_of_pipe = subprocess.Popen(["echo", token], stdout=subprocess.PIPE)
      # right_of_pipe = subprocess.run(['huggingface-cli', 'login'], stdin=left_of_pipe.stdout, stdout=subprocess.PIPE).stdout.decode('utf-8')
      # Manager.Diffusion.install_model(username, token, settings["model_id"])
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
      Manager.Diffusion.patch_nsfw(settings['enable_nsfw_filter'])
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


    def install_model(username, token, model_id):
      subprocess, os = Manager.manage_imports('diffuser_install')
      print('Installing Model')
      print(subprocess.run(['git','lfs','install'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
      # This will take a while
      print(subprocess.run(['git','lfs','clone',f'https://{username}:{token}@huggingface.co/{model_id}'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

    def creds():
      return 'x90', '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'

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
        from io import BytesIO
        response = requests.get(settings['init_image'])
        init_image = PIL.Image.open(BytesIO(response.content))
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

    class Scheduler:
      def make(settings):
        scheduler = Manager.manage_imports(settings["scheduler"])
        global pipe
        if settings["scheduler"] == 'default' or settings["scheduler"] == 'pndm':
          pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True)
        elif settings["scheduler"] == 'k-lms':
          pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        elif settings["scheduler"] == 'ddim':
          pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    class Runner:

      def run(settings):
        collected_results = [] # PAGODA


        def sharpen_mage(image, samples=1):
          from PIL import ImageFilter
          im = image
          for i in range(samples):
              im = im.filter(ImageFilter.SHARPEN)
          return im
        import time
        torch, precision_scope, randint, sys = Manager.Diffusion.Runner.get_general_imports(settings)
        with torch.no_grad():
          with precision_scope("cuda"):
            if settings['seed'] == 0:
              settings['seed'] = randint(0,sys.maxsize)
            generator = torch.Generator("cuda").manual_seed(settings['seed'])
            counter = 1
            clean_counter = 0
            running = True
            if settings["use_drive_for_pics"]:
              outdir = f'/workspace/api/drive/MyDrive/{settings["drive_pic_dir"]}'
            else:
              outdir = '/workspace/api/diffusers_output'
            epoch_time = int(time.time())
            if settings["save_prompt_details"]:
              with open(f'{outdir}/{epoch_time}_prompt.json', 'w') as file:
                import json
                file.write(json.dumps(settings, indent=2))
            if settings['mode'] == "IMG2IMG":
              init_image = Manager.Diffusion.img2img_init(settings)
            elif settings['mode'] == 'Inpainting':
              init_image, mask_image = Manager.Diffusion.inpaint_init(settings)
            while running:

              collected_current = {} # PAGODA

              Cleaner.clean_env()
              if settings["mode"] == "PROMPT":
                if settings['prompt_type'] == 'TEXT':
                  image = Manager.Diffusion.Runner.text_prompt(settings, torch, generator)
                else:
                  for prompt in settings["file_prompt"]:
                    pass
                    # TODO
              elif settings["mode"] == "IMG2IMG":
                image = Manager.Diffusion.Runner.img_to_img(settings, torch, generator, init_image)
              elif settings["mode"] == "Inpainting":
                image = Manager.Diffusion.Runner.inpainting(settings, torch, generator, init_image, mask_image)
              elif settings["mode"] == "CLIP GUIDED PROMPT":
                Cleaner.clean_env()
                image = Manager.Diffusion.Runner.clip_guided_prompt(settings, torch, generator)
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
                image = Manager.Diffusion.Runner.img2img_postprocess(settings, image, generator)
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

      def img2img_postprocess(settings, image, generator):
        import torch
        print("running img2img postprocessing. Switching to img2img pipe")
        global pipe
        pipe = None
        Cleaner.clean_env()
        pipe_library = Manager.manage_imports("IMG2IMG")
        import os, subprocess, torch
        username, token = Manager.Diffusion.creds()
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
        scheduler = Manager.manage_imports(settings["scheduler"])
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
        image = Manager.Diffusion.Runner.img_to_img(img2img_settings, torch, generator, image)
        pipe = None
        del pipe
        print('Switching back to old pipe and then displaying the image')
        Cleaner.clean_env()
        Cache.Pipe.make(settings)
        Manager.Diffusion.Scheduler.make(settings)
        return image

      def get_general_imports(settings):
        torch, randint, sys = Manager.manage_imports('general_diffusion_run')
        if settings['precision'] == 'autocast':
          return torch, torch.autocast, randint, sys
        else:
          from contextlib import nullcontext
          return torch, nullcontext, randint, sys

      def text_prompt(settings, torch, generator):
        global pipe
        if settings['scheduler'] == 'ddim':
          image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], width=settings['width'], height=settings['height'], guidance_scale=settings['scale'], eta=settings["ddim_eta"], generator=generator)
        else:
          image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], width=settings['width'], height=settings['height'], guidance_scale=settings['scale'], generator=generator)
        return image["sample"][0]
        

      def file_prompt(settings, torch, generator):
        global pipe
        pass

      def img_to_img(settings, torch, generator, init_image):
        if settings['scheduler'] == 'ddim':
          image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, strength=settings['init_strength'], eta=settings["ddim_eta"], guidance_scale=settings['scale'], generator=generator)["sample"][0]
        else:
          image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, strength=settings['init_strength'], guidance_scale=settings['scale'], generator=generator)["sample"][0]
        return image

      def inpainting(settings, torch, generator, init_image, mask_image):
        global pipe
        if settings['scheduler'] == 'ddim':
          image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, mask_image=mask_image, strength=settings["inpaint_strength"], eta=settings["ddim_eta"], guidance_scale=settings['scale'], generator=generator)["sample"][0]
        else:
          image = pipe(prompt=settings['text_prompt'], negative_prompt=settings['negative_text_prompt'], num_inference_steps=settings['steps'], init_image=init_image, mask_image=mask_image, strength=settings["inpaint_strength"], guidance_scale=settings['scale'], generator=generator)["sample"][0]
        return image

      def clip_guided_prompt(settings, torch, generator):
        global pipe
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


global last_model

if __name__ == "__main__":
  

  # start pipe and get settings
  Cleaner.clean_env()
  manager = Manager()
  manager.eval_settings()

  
  # New dependencies
  from fastapi import FastAPI, Response
  from fastapi.responses import HTMLResponse
  from fastapi.middleware.cors import CORSMiddleware
  from starlette.responses import StreamingResponse
  from pydantic import BaseModel
  import base64
  from io import BytesIO
  import nest_asyncio
  import uvicorn
  import json


  # Create fast api app
  app = FastAPI()

  app.add_middleware(
      CORSMiddleware,
      allow_origins=['*'],
      allow_credentials=True,
      allow_methods=['*'],
      allow_headers=['*'],
  )

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
        collected_round = Manager.Diffusion.Runner.run(current_settings) # array of PIL images
        collected.extend(collected_round)
        print(f"Batch {batch_value} done")
        batch_value += batch_increase
      print("Done with batch")
    else:
      collected = Manager.Diffusion.Runner.run(current_settings) # array of PIL images
    
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
      
  # Api routes
  @app.post('/')
  async def rootApi(generationRequest: GenerationRequest, response: Response):
    requestParameters = generationRequest.dict()
    # requestParameters["num_iters"] = 1 # force num_iters to 1
    images = generate_to_base64(requestParameters)

    # IMAGE RETURN
    # response.headers["X-seed"] = str(images[0]["seed"])
    # return StreamingResponse(BytesIO(images[0]["image"]), media_type="image/png")

    # HTML RETURN
    # html = """<!DOCTYPE html><html><head><title>Generated Images</title></head><body>"""
    # for image in images:
    #   html += f"""<img src="{image["image"]}">"""
    # html += """</body></html>"""
    # return html
    # return HTMLResponse(content=html, status_code=200)

    # JSON RETURN
    # response.headers["X-seed"] = str(images[0]["seed"])
    # images is a list of dicts with keys "image" and "seed"
    # convert list to array

    # empty dict
    result = { "parameters": requestParameters }

    result["images"] = []
    result["images"].extend(images)
    result = json.dumps(result)

    return Response(content=result, media_type="application/json")
 

  # starting app 
  nest_asyncio.apply()
  uvicorn.run(app, port=7860, host='0.0.0.0')
 