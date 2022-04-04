#
# preprocess.py
#
# Given an input MIDI file, extract the appropriate information into a
# format that the model may understand explicitly. This process is 
# largely the same for train/test and inference data, with a few
# additions for extracting training/testing targets. 
#
# For all data:
# 1. Combine tracks if there are multiple tracks. (This is frequently
#    done to have separate staffs for left vs right hands)
# 2. Harmonize the MIDI's meta information (tempo, time signature, etc.)
#    it might be bad for sheet music representation, but our goal is to
#    generate music!
# 3. Extract all note information into a dataframe for the model to work
#    on. Each timestep t should be a vector of size 3, with:
#     - Note id
#     - Time
#     - Note on (True if given velocity=0 OR if msg is note_off)
# 4. Add an extra first note at time=0 for the initial control changes
#    aka pedal movements. 
#
# For training/testing data (extracting labels):
# 5. Extract velocities for all notes as solutions. 
# 6. Extract up to p (configurable granularity hparam) control changes
#    between notes - if we have more than p, calculate the closest 
#    approximation given the lowered resolution. Add the control changes
#    information to solution info, padding up to p if necessary. 
#
#     -> Each control change contains 3 pieces of info to predict:
#        1) control 2) value 3) time. The solution vector will therefore
#        be of size 3p + 1.

from data_params import *
from midi_utils import *

from pathlib import Path
from mido import MidiFile
import pandas as pd

def preprocess_targeted_dataset(clean_data: Path, output_path: Path):
  """
  Given the path to a set of natural MIDI recordings of humans playing
  the piano, generate a train/test dataset by preprocessing the files 
  and extracting solution information (velocities + <= p control changes).
  """
  pass

def preprocess_maestro(clean_data: Path, output_path: Path):
  """
  The Maestro dataset has a .csv that indicates whether a MIDI file
  should belongs in the train, test, or validation sets. Preprocess
  the dataset by first copying the files into appropriate directories
  and then running preprocess_targeted_dataset.

  Expects the dataset to be labeled as "maestro-v3.0.0" and the csv
  called "maestro-v3.0.0.csv".
  """
  pass

def preprocess_midi(midi_file: Path):
  """
  Given the path to a MIDI file, execute general preprocessing for
  either train/test or inference. Combine tracks, harmonize meta info,
  and construct a dataframe that the model can use. 

  Returns the constructed dataframe. 
  """
  pass

def generate_solutions(midi_file: Path, data: pd.DataFrame):
  """
  Given the path to a MIDI file, extract solutions for test/train, as
  it is expected that this file will contain performance data of actual
  humans. Compile note velocities as well as control changes into the
  solution column, executing control change downsampling if necessary.

  Returns the final dataframe with solutions for each timestep. 
  """
  pass