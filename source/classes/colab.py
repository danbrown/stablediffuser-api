class Colab:
  def __init__(self):
    self.settings = self.UserSettings.set_settings()

  def clear():
    from IPython.display import clear_output; clear_output()
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
