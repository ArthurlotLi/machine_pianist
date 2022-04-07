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
    #print_first_x(midi, 10)

    midi.save(_temp_file)
    mergemid(_temp_file, _temp_file2)
    merged_midi = read_midi(_temp_file2)

    #print("[DEBUG] Midi Utils - All done. Result:")
    #print_first_x(merged_midi, 10)

    return merged_midi
  else:
    # Just one track. Don't bother. 
    return midi


def harmonize_meta(midi: MidiFile):
  """
  Given an input midi file, convert the song's meta information if
  necessary.
  - tempo
  - time signature numerator
  - time signature denominator
  - clocks_per_click
  - notated_32nd_notes_per_beat
  """
  print("[INFO] Midi Utils - Harmonizing meta information.")
  all_meta_info = []
  
  # Only one track allowed.
  assert len(midi.tracks) == 1

  # Gather all meta information present in the track.
  for i, track in enumerate(midi.tracks):
    for j in range(0, len(track)):
      msg = track[j]
      if msg.is_meta:
        all_meta_info.append(msg)

  # Remove duplicates (in the case that we merged more than 1 track)
  unique_meta_info = []
  for msg in all_meta_info:
    if msg not in unique_meta_info:
      unique_meta_info.append(msg)

  print("[DEBUG] Midi Utils - All meta info:")
  for msg in unique_meta_info:
    print("        %s" % msg)

  # We can expect the following meta messages:
  # - MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0)
  # - MetaMessage('key_signature', key='Ab', time=0)
  # - MetaMessage('set_tempo', tempo=260870, time=0)
  # - MetaMessage('midi_port', port=0, time=0)
  # - MetaMessage('end_of_track', time=1)

  # If the set_tempo message is not provided, the default is 500000 (120 bpm)
  # Use tick2second() from mido to convert to absolute seconds since start.
  # Use this info to convert songs with different tempos. 


  #print("[DEBUG] Midi Utils - Done. Result:")
  #print_first_x(midi, 30)
  #player = PianoPlayer()
  #player.play_mido(midi, None, block=True, verbose=True)
  #return midi

# Function for extracting up to p control changes, downsampling if
# necessary. 