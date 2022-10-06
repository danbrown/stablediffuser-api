from ..utils import manage_imports
class Cleaner:
  def clean_env():
    gc, torch = manage_imports('clean_env')
    gc.collect()
    torch.cuda.empty_cache()
