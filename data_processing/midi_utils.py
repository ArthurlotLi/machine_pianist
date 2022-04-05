#
# midi_utils.py
#
# Preprocessing utilities working in the weeds with MIDI files to
# extract required info. Keep all the really complex stuff here. 

from data_processing.data_params import *
from utils.midi_player import print_first_x

from pathlib import Path
from mido import MidiFile, MidiTrack

def read_midi(midi_file: Path):
  return MidiFile(midi_file)

def combine_tracks(midi: MidiFile):
  """
  Given an input midi, if it has more than one track, automatically
  merges the tracks together into one. This is made with the 
  assumption that the tracks are there to indicate different hands
  (potentially different players for duets).
  """
  if len(midi.tracks) > 1:
    print("[DEBUG] Midi Utils - Combining %d tracks." % len(midi.tracks))
    print_first_x(midi, 20)
    combined_track = MidiTrack()
    for i, track in enumerate(midi.tracks):
      # Each track is of type MidiTrack. midi.tracks is simply a list
      # of tracks. 
      if i == 0: 
        combined_track = track
      else: 
        combined_track += track
    merged_midi = midi
    merged_midi.tracks = [combined_track]
    print("[DEBUG] Midi Utils - Done. Result:")
    print_first_x(merged_midi, 20)
    return merged_midi
  else:
    # Just one track. Don't bother. 
    return midi


# Function for automatically converting Meta data. 

# Function for extracting up to p control changes, downsampling if
# necessary. 