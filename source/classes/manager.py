from .colab import Colab
from .cache import Cache

class Manager:
  def __init__(self):
    self.colab = Colab()

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

  