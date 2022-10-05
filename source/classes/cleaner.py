from .manager import Manager

class Cleaner:
  def clean_env():
    gc, torch = Manager.manage_imports('clean_env')
    gc.collect()
    torch.cuda.empty_cache()
