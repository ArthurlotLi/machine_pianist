#
# production_inference.py
#
# Primary entry point for users of this project in order to conduct
# inference, having the machine pianist play their midi file in a
# more "natural" way. 
#
# Allows for model inference, provided piano midi file(s) to load
# and conduct inference on. 

from model.load_save import load_existing_model_path
from model.dataset_utils import generate_song_tensors
from data_processing.preprocess import preprocess_midi
from data_processing.postprocess import generate_output_midi
from data_processing.data_params import *

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time
import argparse

class MachinePianist:
  def __init__(self, model_path: Path):
    """
    Loads the model immediately.
    """
    print("[INFO] Machine Pianist - Loading model at: %s." % model_path)
    self._model = load_existing_model_path(model_path)
    if self._model is None:
      print("[ERROR] Machine Pianist - Failed to load model at %s." % model_path)
      assert False

  def perform_midis(self, midi_files: list):
    """
    Provided a list of files to load, preprocess each song and combine
    into a dataframe that can be fed to the model. After the model has
    inferred output data, combine that output data with the midi, 
    generating a new output midi with "performance" information. 

    Returns a list of machine performance midis. 
    """
    print("[INFO] Machine Pianist - Preprocessing %d songs." % len(midi_files))
    # First preprocess all songs to get the new midis + dataframes. 
    preprocessed_songs = []
    preprocessed_dfs = []
    for i in range(0, len(midi_files)):
      midi = Path(midi_files[i])
      midi, song_X = preprocess_midi(midi_file = midi, song_uid=i)
      preprocessed_songs.append((i, midi))
      preprocessed_dfs.append(song_X)
    
    assert len(preprocessed_songs) > 0 and len(preprocessed_dfs) > 0

    # Now let's make sure to pad every single song here. 
    X = generate_song_tensors(songs_list=preprocessed_dfs, solutions=False)

    # We now have a complete X dataframe. Conduct inference. 
    print("[INFO] Machine Pianist - Performing songs...")
    start_time = time.time()
    Y_hat = self._model.predict(X)
    print("[INFO] Machine Pianist - Songs performed! Playtime: %.2f seconds." % (time.time() - start_time))

    # Get the postprocessed songs and return them.
    return generate_output_midi(preprocessed_songs, Y_hat)

# For debug usage.
#
# Usage (Graph only):
# python production_inference -g -p
#
# Usage (Play):
# python production_inference
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-g", default=False, action="store_true")
  parser.add_argument("-p", default=True, action="store_false")
  args = parser.parse_args()

  midi_files = [
    "./midi_test/Undertale_-_Spider_Dance_-_Lattice.mid",
    "./midi_test/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--1.midi"
  ]

  model_path = Path("./production_models/model1/machine_pianist.h5")

  from utils.midi_player import *

  if args.g is True:
    for midi_path in midi_files:
      midi, song_X = preprocess_midi(midi_file = midi_path, song_uid=0)
      graph_velocities_notes(midi)

  pianist = MachinePianist(model_path)
  midis = pianist.perform_midis(midi_files)

  player = PianoPlayer()
  for midi in midis:
    if args.g is True:
      graph_velocities_notes(midi)
    #print_first_x(midi, 500, notes_only=True)
    if args.p is True:
      player.play_mido(midi, block=True, verbose=True)