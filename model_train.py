#
# model_train.py
#
# Executes training on a preprocessed dataset. Straightforward.
#
# Usage:
# python model_train.py model1

from model.train import train_machine_pianist

import argparse
from pathlib import Path

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--clean_data", type=Path, default="./datasets/complete", help = 
    "The location of your datasets that you wish to use for training and or "
    "testing.")
  parser.add_argument("--saved_models", type=Path, default="saved_models", help=
    "The directory in which a subfolder with the model identifier will be"
    "created for checkpoints to be saved during training.")
  parser.add_argument("model_id", type=str, help=
    "The identifier for the model to be trained.")
  args = parser.parse_args()
  
  train_machine_pianist(**vars(args))