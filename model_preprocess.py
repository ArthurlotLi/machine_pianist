#
# model_preprocess.py
#
# Given the contents of the provided datasets folder, generate a
# dataset with a train/test split for use with our model. 
#
# Usage:
# python model_preprocess.py

from data_processing.preprocess import *

import argparse
from pathlib import Path

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--clean_data", type=Path, default="./datasets", help=
    "The location of your datasets that you wish to use for training and or "
    "testing.")
  parser.add_argument("--dataset_types", type=str, default="maestro", help=
    "A comma-separated list of datasets to process. Default is just maestro.")
  parser.add_argument("--output_path", type=Path, default="./datasets/complete", help=
    "Path to the directory where you want the generated .csv files to be "
    "placed. ")
  args = parser.parse_args()

  dataset_to_function = {
    "maestro" : preprocess_maestro
  }

  for dataset in args.dataset_types.split(","):
    assert dataset in dataset_to_function
    dataset_to_function[dataset](clean_data=args.clean_data, 
                                 output_path=args.output_path)