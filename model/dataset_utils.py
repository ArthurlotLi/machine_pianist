#
# dataset_utils.py
#
# Utilities for interacting with existing datasets. Used by train and
# inference. 

from data_processing.data_params import *

from typing import Optional
from pathlib import Path
import pandas as pd

def load_datasets(clean_data: Path,
                  load_train: Optional[bool] = False, 
                  load_test: Optional[bool] = False):
  """
  Given which datasets to load + clean data location, loads them.
  Returns a variable sized list depending on what booleans were set
  to true. 
  
  The list will always be in the order of train, test.
  """
  assert clean_data.exists()
  assert load_train is True or load_test is True

  def load_dataset(dataset_path: Path):
    print("[INFO] Dataset Utils - Loading training dataset at: %s" % dataset_path)
    assert dataset_path.exists()
    dataset = pd.read_csv(dataset_path)
    print("[INFO] Dataset Utils - Loaded dataset of shape: %s." % str(dataset.shape))
    return dataset

  loaded_datasets = []
  if load_train is True:
    loaded_datasets.append(
      load_dataset(clean_data.joinpath(data_train_csv))
      )
  if load_test is True:
    loaded_datasets.append(
      load_dataset(clean_data.joinpath(data_test_csv))
      )

  return loaded_datasets