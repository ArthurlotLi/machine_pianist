#
# dataset_utils.py
#
# Utilities for interacting with existing datasets. Used by train and
# inference. 

from data_processing.data_params import *
from model.hparams import *

from sklearn.preprocessing import StandardScaler
from typing import Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

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
      load_dataset(clean_data.joinpath(data_train_csv)))
  if load_test is True:
    loaded_datasets.append(
      load_dataset(clean_data.joinpath(data_test_csv)))

  return loaded_datasets

def generate_song_tensors(songs_df = None, songs_list = None, solutions = True,
                          scaler_X = None, scaler_Y = None):
  """
  Given a dataframe(s), generate tensors for model usage (train/test) by 
  grouping songs and padding them to the maximum number of notes.

  Returns two items - two tensors if solutions are present, otherwise
  one tensor + none. 
  """
  # ONE of the two options must be None, the other must not.
  assert songs_df is not None or songs_list is not None
  assert songs_df is None or songs_list is None

  # The scalers must be provided if solutions are disabled. 
  assert (scaler_X is not None and scaler_Y is not None) or solutions is True

  if songs_df is not None:
    songs_list = songs_df.groupby([data_uid_col])
  else:
    songs_list = enumerate(songs_list)

  padded_songs_X = []
  padded_songs_Y = []

  dropped_songs = 0
  total_songs = 0
  for i, song_df in tqdm(songs_list, desc="[INFO] Dataset Utils - Padding Songs"):
    total_songs += 1
    padded_df = pad_song_note_off(song_df, maximum_song_length, solutions)
    if padded_df is None:
      dropped_songs += 1
    else:
      padded_df = padded_df.loc[:, ~padded_df.columns.str.contains('^Unnamed')]
      # Split into X and Y if solutions are present.
      if solutions is True:
        Y = padded_df.loc[:,data_solution_cols]
        # Drop the solution column + unnamed columns.
        X = padded_df.drop(labels=data_solution_cols, axis=1)
        # Temp column to split grouped results during scaling.
        X["temp"] = i
        padded_songs_X.append(X)
        padded_songs_Y.append(Y)
      else:
        # Temp column to split grouped results during scaling.
        padded_df["temp"] = i
        padded_songs_X.append(padded_df)

  print("[INFO] Dataset Utils - Total dropped songs: %d out of %d." % (dropped_songs, total_songs))

  # Apply standard scalers here - fit new ones if they were not provided
  # (only during training preprocessing).
  print("[INFO] Dataset Utils - Merging songs together for scaling.")
  combined_X = pd.concat(objs=padded_songs_X, ignore_index=True)
  if solutions is True: combined_Y = pd.concat(objs=padded_songs_Y, ignore_index = True)

  temp_col = combined_X["temp"].tolist()
  combined_X.drop("temp", axis=1, inplace=True)

  # Fit the scalers if necessary. 
  if scaler_X is None or scaler_Y is None:
    assert solutions is True

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    print("[INFO] Dataset Utils - Fitting X standard scaler.")
    scaler_X.fit(combined_X)
    print("[INFO] Dataset Utils - Fitting Y standard scaler.")
    scaler_Y.fit(combined_Y)
  
  # Apply the scalers.
  print("[INFO] Dataset Utils - Scaling X tensors.")
  combined_X = pd.DataFrame(scaler_X.transform(combined_X), columns = combined_X.columns)
  combined_X["temp"] = temp_col
  if solutions is True: 
    print("[INFO] Dataset Utils - Scaling Y tensors.")
    combined_Y = pd.DataFrame(scaler_Y.transform(combined_Y), columns = combined_Y.columns)
    combined_Y["temp"] = temp_col

  # Now that everything's scaled, split everything up again. 
  padded_songs_X = combined_X.groupby("temp")
  if solutions is True:
    padded_songs_Y = combined_Y.groupby("temp")

  final_X_list = []
  final_Y_list = []

  for _, padded_X_df in padded_songs_X:
    padded_X_df.drop("temp", axis=1, inplace=True)
    final_X_list.append(padded_X_df.to_numpy())

  if solutions is True:
    for _, padded_Y_df in padded_songs_Y:
      padded_Y_df.drop("temp", axis=1, inplace=True)
      final_Y_list.append(padded_Y_df.to_numpy())

  # Combine all of these into tensors. 
  X_final = np.array(final_X_list)
  if solutions is True:
    Y_final = np.array(final_Y_list)
    print("[INFO] Dataset Utils - Generated tensors shape: X=%s, Y=%s" % (str(X_final.shape), str(Y_final.shape)))
    # Sanity checks. 
    assert X_final.shape[0] == Y_final.shape[0]
    assert X_final.shape[1] == Y_final.shape[1]
    return X_final, Y_final, scaler_X, scaler_Y
  else:
    print("[INFO] Dataset Utils - Generated tensor shape: X=%s" % (str(X_final.shape)))
    return X_final

def pad_song_note_off(song_df, max_notes, solutions=True):
  """
  Given a dataframe, assert the length is less than the max notes. If
  so, then pad the length up to the max notes. 
  """
  if song_df.shape[0] > max_notes:
    #print("\n[WARNING] Dataset Utils - Received a dataset with illegal length %d (max: %d)! Dropping..."
      #% (song_df.shape[0], maximum_song_length))
    #return None
    song_df = song_df.head(maximum_song_length)

  # The dataframe MUST have the correct size depending on if solutions
  # are provided. 
  if solutions is False:
    assert song_df.shape[1] == 4
  else:
    if song_df.shape[1] != 4 + len(data_solution_cols):
      print(song_df.columns)
      input()
    assert song_df.shape[1] == 4 + len(data_solution_cols)

  # Grab the song_id. We assume the songid is uniform for all rows.
  first_row = song_df.head(1)
  song_uid = int(first_row[data_uid_col])

  # We've made sure it's of a good shape. Pad it. 
  padding_necessary = max_notes - song_df.shape[0]

  if solutions is False:
    new_rows = [60, 0, 0, song_uid]
  else:
    new_rows = [60, 0, 0, song_uid, 0]
    # Add two zeros for each empty control change. 
    for _ in range(0, p_granularity_total):
      new_rows.append(0)
      new_rows.append(0)

  # Generate a dataframe with copies of the same row. 
  padding_df = pd.DataFrame([new_rows], 
                            index=range(padding_necessary), 
                            columns = list(song_df.columns.values))

  # Combine the two. 
  song_df = pd.concat(objs=[song_df, padding_df], axis=0)

  # Drop the song uid column. 
  song_df = song_df.drop(labels=[data_uid_col], axis=1)

  # At the end, we need to ensure the song is a matrix of the
  # appropriate shape. 
  if solutions is False:
    assert song_df.shape == (max_notes, 3)
  else:
    assert song_df.shape == (max_notes, 3 + len(data_solution_cols))
  
  return song_df
  