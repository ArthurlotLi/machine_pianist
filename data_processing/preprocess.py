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

from data_processing.data_params import *
from data_processing.midi_utils import *

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
  called "maestro-v3.0.0.csv", configured in data_params.
  """
  # Make sure everything is here.
  assert clean_data.exists()
  dataset_location = clean_data.joinpath(maestro_dataset_folder)
  assert dataset_location.exists()
  csv_location = dataset_location.joinpath(maestro_dataset_csv)
  assert csv_location.exists()
  output_path.mkdir(exist_ok=True)

  # Read the CSV and parse it. 
  maestro_csv = pd.read_csv(str(csv_location))
  train_set = []
  test_set = []

  # Not pandas-kosher, but it's a tiny csv and this makes it legible.
  for index, row in maestro_csv.iterrows():
    # For each line item, get the midi path and whether it should be
    # a train or test. All other info is irrelevant. 
    song_split = row["split"]
    # Given the nature of this project, we will pass on the val set.
    song_split = song_split.replace("validation", "train")
    song_midi = dataset_location.joinpath(row["midi_filename"])

    # Go ahead and preprocess each MIDI. 
    midi, X_df = preprocess_midi(song_midi)
    Y_df = generate_solutions(midi, X_df)

    if song_split == "train":
      train_set.append((X_df, Y_df))
    elif song_split == "test":
      test_set.append((X_df, Y_df))
    else:
      assert False
  
  # We've preprocessed everthing. Save to file. 
  # TODO: concat X and Y dataframes for test and train and save. 


def preprocess_midi(midi_file: Path):
  """
  Given the path to a MIDI file, execute general preprocessing for
  either train/test or inference. Combine tracks, harmonize meta info,
  and construct a dataframe that the model can use. 

  Returns the preprocessed midi + constructed dataframe.
  """
  midi = read_midi(midi_file)
  midi = combine_tracks(midi)
  midi = harmonize_meta(midi)

  return midi, generate_dataframe(midi)

def generate_solutions(midi: MidiFile, data: pd.DataFrame):
  """
  Given the path to a MIDI file, extract solutions for test/train, as
  it is expected that this file will contain performance data of actual
  humans. Compile note velocities as well as control changes into the
  solution column, executing control change downsampling if necessary.

  Returns the final dataframe with solutions for each timestep. 
  """
  generate_sol_dataframe(midi, data)