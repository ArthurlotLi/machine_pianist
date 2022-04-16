#
# load_save.py
#
# Loading/saving a model. 

from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from pathlib import Path

def load_existing_model(saved_models:Path, model_id: str, filename:str):
  """
  Attempts to load a model. Returns None if it wasn't found.
  """
  model_location = saved_models.joinpath(model_id).joinpath(filename)
  return load_existing_model_path(model_location)
  

def load_existing_model_path(model_location:Path):
  """
  Given exact path. 
  """
  if not model_location.exists(): return None
  # Otherwise, it exists - try to load it. 
  try:
    model = load_model(str(model_location))
    return model
  except Exception as e:
    print("[ERROR] Load Model - Failed to load existing model file: %s" % model_location)
    print(e)
  return None

def save_model(saved_models:Path, model_id: str, model: Model):
  """
  Saves a model. 
  """
  # TODO: Implement if necessary. 
  pass