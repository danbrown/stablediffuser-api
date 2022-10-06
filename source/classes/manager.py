from .colab import Colab
from .cache import Cache

class Manager:
  def __init__(self):
    self.colab = Colab()
    self.pipe = None
    self.pipetype = None
    self.last_model = None

  def eval_settings(self, pipe=None, pipetype=None, last_model=None, settings=False):
    if not settings:
      settings = self.colab.settings

      
    if settings['mode'] == "PROMPT FILE":
      with open(settings['prompt_file'],'r') as file:
        import json
        self.colab.settings = json.loads(file.read())
        settings = self.colab.settings

    if not pipe: 
      pipe = self.pipe
    if not pipetype:
      pipetype = self.pipetype
    if not last_model:
      last_model = self.last_model

    import json
    print(json.dumps(settings, indent=2))
    try:
      if pipetype != settings['mode'] or pipe is None or settings['model_id'] != last_model:
        pipe, pipetype, last_model = Cache.Pipe.make(pipe=pipe, pipetype=pipetype, settings=settings)
    except NameError:
      pipe, pipetype, last_model = Cache.Pipe.make(settings)

    self.pipe = pipe
    self.pipetype = pipetype
    self.last_model = last_model
    
    return pipe, pipetype, last_model

  