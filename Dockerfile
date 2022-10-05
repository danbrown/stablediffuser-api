FROM runpod/stable-diffusion:web-automatic

WORKDIR /workspace

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


# Install filebrowser
RUN curl -fsSL 'https://raw.githubusercontent.com/filebrowser/get/master/get.sh' | bash

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Dev: docker build --build-arg HF_AUTH_TOKEN=${HF_AUTH_TOKEN} ...
# Banana: currently, comment out ARG and set by hand ENV line.
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

# Which model to download and use; fork / downstream specific.
ADD DOWNLOAD_VARS.py .
ADD loadModel.py .
ADD download.py .
ADD send.py .
ADD app.py .

# Download the models
RUN python3 download.py

# Runtime vars (for init and inference); fork / downstream specific.
ADD APP_VARS.py .

CMD ["/start.sh"]
