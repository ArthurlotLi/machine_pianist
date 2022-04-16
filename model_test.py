#
# model_test.py
#
# Allows for testing of model checkpoints after training has finished
# in order to find the best one (My preferred method instead of 
# early stopping.)
#
# Usage:
# python model_test.py ./saved_models/model1

from model.test import test_models

import argparse
from pathlib import Path

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--clean_data", type=Path, default="./datasets/complete", help = 
    "The location of your datasets that you wish to use for training and or "
    "testing.")
  parser.add_argument("--output_path", type=Path, default="./evaluation_results", help=
    "The directory in which the evaluation results + curves will be written.")
  parser.add_argument("model_location", type=Path, help=
    "The location for the models to be tested.")
  args = parser.parse_args()
  
  test_models(**vars(args))