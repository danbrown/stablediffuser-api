wget -q https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/replacements/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py -O /workspace/api/pipeline_stable_diffusion.py
wget -q https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/replacements/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py -O /workspace/api/pipeline_stable_diffusion_img2img.py
wget -q https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/replacements/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py -O /workspace/api/pipeline_stable_diffusion_inpaint.py

cp /workspace/api/pipeline_stable_diffusion.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
cp /workspace/api/pipeline_stable_diffusion_img2img.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
cp /workspace/api/pipeline_stable_diffusion_inpaint.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py

cp /workspace/api/safety_checker_patched.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py
# cp /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py /workspace/api/safety_checker.py
# cp /workspace/api/safety_checker.py /workspace/api/safety_checker_patched.py
