from ..utils import manage_imports

class Scheduler:
  def make(settings):
    scheduler = manage_imports(settings["scheduler"])
    global pipe
    if settings["scheduler"] == 'default' or settings["scheduler"] == 'pndm':
      pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True)
    elif settings["scheduler"] == 'k-lms':
      pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    elif settings["scheduler"] == 'ddim':
      pipe.scheduler = scheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
