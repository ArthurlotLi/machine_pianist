#
# midi_utils.py
#
# Preprocessing utilities working in the weeds with MIDI files to
# extract required info. Keep all the really complex stuff here. 

from data_processing.data_params import *
from utils.midi_player import *
from utils.mergemid import *

from pathlib import Path
from mido import MidiFile

_temp_file = "temp_file"
_temp_file2 = "temp_file2"

def read_midi(midi_file: Path):
  return MidiFile(midi_file)

def combine_tracks(midi: MidiFile):
  """
  Given an input midi, if it has more than one track, automatically
  merges the tracks together into one. This is made with the 
  assumption that the tracks are there to indicate different hands
  (potentially different players for duets).

  This uses some WONDERFUL code from github user m13253 that works
  on the low-level MIDI bytes that does what mido could not 
  effortlessly. 
  """
  if len(midi.tracks) > 1:
    print("[DEBUG] Midi Utils - Combining %d tracks." % len(midi.tracks))
    print_first_x(midi, 30)
    player = PianoPlayer()
    player.play_mido(midi, None, block=True, verbose=True)

    midi.save(_temp_file)
    mergemid(_temp_file, _temp_file2)
    merged_midi = read_midi(_temp_file2)

    print("[DEBUG] Midi Utils - Done. Result:")
    print_first_x(merged_midi, 60)
    player = PianoPlayer()
    player.play_mido(merged_midi, None, block=True, verbose=True)

    return merged_midi
  else:
    # Just one track. Don't bother. 
    return midi


# Function for automatically converting Meta data. 

# Function for extracting up to p control changes, downsampling if
# necessary. 