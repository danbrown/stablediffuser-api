#!/bin/bash
echo "Container Started"
source /venv/bin/activate
cd /workspace

# update the repo
cd /workspace/banana
git pull
echo "Repo Updated"

# install the requirements
pip install -r requirements.txt
echo "Requirements Installed"

# download the models
python3 download.py
echo "Models Downloaded"

# run filebrowser
filebrowser -r /workspace -p $FILEBROWSER_PORT &
echo "Filebrowser Started"

# run the server (background)
python3 -u server.py &
echo "Server Started"

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
    echo "SSH Service Started"
fi

if [[ $JUPYTER_PASSWORD ]]
then
    ln -sf /examples /workspace
    ln -sf /root/welcome.ipynb /workspace

    cd /
    jupyter lab --allow-root --no-browser --port=$JUPYTER_PORT --ip=* \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace
    echo "Jupyter Lab Started"
fi

sleep infinity