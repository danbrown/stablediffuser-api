# This file set up utility functions

# Get a remote file and save it to disk (using wget)
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


# get the pip modules directory
def get_pip_modules_dir():
  import site
  return site.getsitepackages()[0]