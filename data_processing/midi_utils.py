#
# midi_utils.py
#
# Preprocessing utilities working in the weeds with MIDI files to
# extract required info. Keep all the really complex stuff here. 

from data_processing.data_params import *
from utils.midi_player import *
from utils.mergemid import *

from pathlib import Path
from mido import MidiFile, tick2second, second2tick
import pandas as pd

_temp_file = "temp_file"
_temp_file2 = "temp_file2"

def read_midi(midi_file: Path):
  return MidiFile(midi_file)

def combine_tracks(midi: MidiFile, song_uid: int):
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
    #print("[DEBUG] Midi Utils - Combining %d tracks." % len(midi.tracks))
    #print_first_x(midi, 10)
    temp_file = _temp_file + str(song_uid)
    temp_file2 = _temp_file2 + str(song_uid)

    midi.save(temp_file)
    mergemid(temp_file, temp_file2)
    merged_midi = read_midi(temp_file2)
    os.remove(temp_file)
    os.remove(temp_file2)

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
  #print("[INFO] Midi Utils - Harmonizing meta information.")
  #print_first_x(midi, 30)
  #player = PianoPlayer()
  #start_time = time.time()
  #player.play_mido(midi, None, block=True, verbose=True)
  #print("Total time: %.2f" % (time.time() - start_time))

  # Only one track allowed.
  assert len(midi.tracks) == 1

  # Gather all meta information present in the track.
  all_meta_info = []
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

  def print_meta(unique_meta_info):
    print("[INFO] Midi Utils - All meta info:")
    for msg in unique_meta_info:
      print("        %s" % msg)

  #print_meta(unique_meta_info)

  # We can expect the following meta messages:
  # - MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0)
  # - MetaMessage('key_signature', key='Ab', time=0)
  # - MetaMessage('set_tempo', tempo=260870, time=0)
  # - MetaMessage('midi_port', port=0, time=0)
  # - MetaMessage('end_of_track', time=1)
  # These defaults are if the message is not provided. We only care 
  # about the tempo at this point.
  orig_tempo = 500000
  present_set_tempo = False

  for msg in unique_meta_info:
    if msg.type == 'set_tempo':
      if present_set_tempo is False:
        # Save the details if this our first time seeing this message.
        orig_tempo = msg.tempo
      else:
        # Validate, if there are duplicate time_signature messages,
        # that they are all the same. If not, PROBLEMS. There are some
        # songs that DO change the tempo mid-song... we don't support
        # those right now. 
        if not(orig_tempo == msg.tempo):
          print_meta(unique_meta_info)
          print("[ERROR] Midi Utils - Duplicate set_tempo meta messages for track are different!")
          return None

  # If the tempo matches what we need, no work is necessary. 
  if orig_tempo == music_set_tempo:
    return midi

  # Otherwise we NEED to revamp the entire midi file such that the 
  # requisite tempo is used. All of our training data is at 500,000, 
  # so to have the best generalization our incoming data needs to be
  # converted to 500,000 as well. Get the absolute seconds since 
  # start for all notes before converting.
  #
  # There will be a slight, largely unnoticable difference between
  # the duration of the source and final. (~0.5 seconds for 120 
  # second songs). TODO to improve.
  absolute_seconds = []
  absolute_time_since_start = 0
  for j in range(0, len(midi.tracks[0])):
    msg = track[j]
    additional_seconds = tick2second(msg.time, midi.ticks_per_beat, orig_tempo)
    absolute_time_since_start += additional_seconds
    absolute_seconds.append(absolute_time_since_start)

  # Change the tempo now. 
  for msg in unique_meta_info:
    if msg.type == 'set_tempo':
      msg.tempo = music_set_tempo

  # And now map the absolute seconds to the new times of every message. 
  time_of_last = 0
  for j in range(0, len(midi.tracks[0])):
    track[j].time = round(second2tick(absolute_seconds[j] - time_of_last, midi.ticks_per_beat, music_set_tempo))
    time_of_last = absolute_seconds[j]

  #print("[DEBUG] Midi Utils - Done. Result:")
  #print_first_x(midi, 30)
  #player = PianoPlayer()
  #start_time = time.time()
  #player.play_mido(midi, None, block=True, verbose=True)
  #print("Total time: %.2f" % (time.time() - start_time))
  return midi

def generate_dataframe(midi: MidiFile):
  """
  Given a preprocessed midi, go ahead and generate the X matrix, with
  each horizontal vector being a single note_on note. 

  The components of each row:
   - note
   - time
   - note_on (1 or 0)
  """

  # Only one track allowed.
  assert len(midi.tracks) == 1

  rows = []

  # Add a first initial note at the start of the song. This always
  # ensures that control changes before the start of the song are
  # predicted. The note selected is inconsequential. 
  rows.append((60, 0, 0))

  track = midi.tracks[0]

  for j in range(0, len(track)):
    msg = track[j]
    if msg.type == "note_on" or msg.type == "note_off":
      note = msg.note
      time = msg.time
      if msg.type == "note_off":
        note_on = 0
      elif msg.velocity == 0:
        note_on = 0
      else:
        note_on = 1
      
      row = (note, time, note_on)
      rows.append(row)
  
  # We've now gathered all of the rows, ignoring any velocity info
  # that is not 0. Return this dataframe. 
  song_df = pd.DataFrame(
    rows, 
    columns=[data_note_col, data_time_col, data_note_on_col])

  return song_df

def generate_sol_dataframe(midi: MidiFile, data: pd.DataFrame):
  """
  Given the preprocessed midi as well as the dataframe of the extracted
  rows, generate solutions. This goes back and uses the non-zero 
  velocities, assuming the subject midi file is actually a recording
  of a human playing. 

  TODO: Process control changes here! I'm out of time in this sprint,
  so we're doing the MVP!
  """
  # Only one track allowed.
  assert len(midi.tracks) == 1

  rows = []
  # Add the solution to the first note. It's obviously velocity 0.
  # TODO: Control changes. 
  rows.append((0,))

  track = midi.tracks[0]

  for j in range(0, len(track)):
    msg = track[j]
    # TODO: Control changes. 
    if msg.type == "note_on" or msg.type == "note_off":
      sol_velocity = msg.velocity
      
      row = (sol_velocity,)
      rows.append(row)
  
  # TODO: Control changes. You get that yet?
  solution_df = pd.DataFrame(rows, columns = [data_velocity_col])
  assert data.shape[0] == solution_df.shape[0]

  # Combine the solutions Y with the X dataframe.
  combined_data = pd.concat(objs=[data, solution_df], axis=1)

  return combined_data