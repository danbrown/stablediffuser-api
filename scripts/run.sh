#!/bin/bash
cd /workspace/stable-diffusion-webui

# update the repo
git pull
echo "Repo Updated"

# install the requirements
pip install -r requirements.txt
echo "Requirements Installed"

# download the models
python3 source/download.py
echo "Models Downloaded"

# patch the models
cp /workspace/stable-diffusion-webui/patches/pipeline_stable_diffusion.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
cp /workspace/stable-diffusion-webui/patches/pipeline_stable_diffusion_img2img.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
cp /workspace/stable-diffusion-webui/patches/pipeline_stable_diffusion_inpaint.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
echo "Models Patched"

# patch safety filter
cp /workspace/stable-diffusion-webui/patches/safety_checker_patched.py /venv/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py
echo "Safety Filter Patched"

# python /workspace/stable-diffusion-webui/server.py