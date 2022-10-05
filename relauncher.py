import subprocess
from source.config import SCRIPTS_PATH

print("Relaunching...")
print(f'{SCRIPTS_PATH}')
subprocess.Popen(f'{SCRIPTS_PATH}/run.sh', shell=True)
