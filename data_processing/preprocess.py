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

from multiprocessing import Pool
from pathlib import Path
from mido import MidiFile
import pandas as pd
import numpy as np
from tqdm import tqdm

# For multiprocessing.
_num_processes = 10

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

  train_output_path = output_path.joinpath(data_train_csv)
  test_output_path = output_path.joinpath(data_test_csv)

  if test_output_path.exists() or train_output_path.exists():
    print("[WARNING] Preprocess - Found existing dataset at %s.\n\nPress [ENTER] to overwrite." % output_path)
    input()

  # Read the CSV and parse it. 
  maestro_csv = pd.read_csv(str(csv_location))
  train_set = []
  test_set = []

  jobs = []
  # Arbitrary ID to teach the model which subsequences are which
  # (Ideally to let the forget gates fire.)
  song_uid = 0
  # Not pandas-kosher, but it's a tiny csv and this makes it legible.
  for index, row in maestro_csv.iterrows():
    # For each line item, get the midi path and whether it should be
    # a train or test. All other info is irrelevant. 
    song_split = row["split"]
    # Given the nature of this project, we will pass on the val set.
    song_split = song_split.replace("validation", "train")
    song_midi = dataset_location.joinpath(row["midi_filename"])

    jobs.append((song_uid, song_split, song_midi))
    song_uid += 1

  job = Pool(_num_processes).imap(process_maestro_row, jobs)
  job_results = tqdm(job, desc="[INFO] Preprocess - Maestro Dataset", unit="songs", total=maestro_csv.shape[0])

  for X_Y_df, song_split in job_results:
    if song_split == "train":
      train_set.append(X_Y_df)
    elif song_split == "test":
      test_set.append(X_Y_df)
    else:
      assert False
  
  assert len(train_set) > 0 and len(test_set) > 0

  # Combine all of the dataframes together into two big matrices.
  print("[INFO] Preprocess - Combining all dataframes.")
  train_df = pd.concat(objs=train_set, axis=0)
  test_df = pd.concat(objs=test_set, axis=0)

  # We've preprocessed everthing. Save to file. 
  print("[INFO] Preprocess - Saving Train and Test matrices of sizes %s and %s respectively." % (str(train_df.shape), str(test_df.shape)))
  train_df.to_csv(str(train_output_path))
  test_df.to_csv(str(test_output_path))

  print("[INFO] Preprocess - All done! Happy training. ")

def process_maestro_row(args):
  song_uid, song_split, song_midi = args[0], args[1], args[2]
  # Go ahead and preprocess each MIDI. 
  midi, X_df = preprocess_midi(song_midi, song_uid)
  X_Y_df = generate_solutions(midi, X_df)
  return X_Y_df, song_split

def preprocess_midi(midi_file: Path, song_uid: int):
  """
  Given the path to a MIDI file, execute general preprocessing for
  either train/test or inference. Combine tracks, harmonize meta info,
  and construct a dataframe that the model can use. Requires an 
  arbitrary but unique identifier (can set to anything if just
  a singleton inference.)

  Returns the preprocessed midi + constructed dataframe.
  """
  midi = read_midi(midi_file)
  midi = combine_tracks(midi, song_uid)
  midi = harmonize_meta(midi)
  df = generate_dataframe(midi)

  # Add the UID as a column. 
  uid_col = np.full(shape= df.shape[0], fill_value=song_uid)
  df[data_uid_col] = uid_col

  return midi, df

def generate_solutions(midi: MidiFile, data: pd.DataFrame):
  """
  Given the path to a MIDI file, extract solutions for test/train, as
  it is expected that this file will contain performance data of actual
  humans. Compile note velocities as well as control changes into the
  solution column, executing control change downsampling if necessary.

  Returns the final dataframe with solutions for each timestep. 
  """
  return generate_sol_dataframe(midi, data)