#
# train.py
#
# Training harness given a preprocessed dataset. 

from model.hparams import *
from model.model import machine_pianist
from model.dataset_utils import load_datasets
from data_processing.data_params import *

from pathlib import Path

def train_machine_pianist(clean_data: Path, model_id: str,
                          saved_models: Path):
  """
  Principal function that first defines the model before executing
  training according to the variables set in hparams. Expects the
  datasets path, the model identifier, and the saved_models location.
  """
  print("[INFO] Train - Beginning Machine Pianist training session for \"%s\"." % model_id)
  # Load the dataset. 
  train_df = load_datasets(clean_data, load_train= True)[0]
  # Define the model + callbacks according to hparams.
  model, callbacks = machine_pianist(saved_models, model_id)
  # Train.