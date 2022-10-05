apt update
apt install wget

cp /workspace/api/pipeline_stable_diffusion.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
cp /workspace/api/pipeline_stable_diffusion_img2img.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
cp /workspace/api/pipeline_stable_diffusion_inpaint.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py

cp /workspace/api/safety_checker_patched.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py

# python launch.py --port 3000 --disable-console-progressbars --listen &

python /workspace/api/api.py