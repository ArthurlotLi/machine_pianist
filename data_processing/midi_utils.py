#
# midi_utils.py
#
# Preprocessing utilities working in the weeds with MIDI files to
# extract required info. Keep all the really complex stuff here. 

from torch import absolute
from data_processing.data_params import *
from utils.midi_player import *
from utils.mergemid import *

from pathlib import Path
from mido import MidiFile, tick2second, second2tick
import pandas as pd
import numpy as np

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
        all_meta_info.append((msg, j))

  # Remove duplicates (in the case that we merged more than 1 track)
  unique_meta_info = []
  for msg, j in all_meta_info:
    if (msg, j) not in unique_meta_info:
      unique_meta_info.append((msg, j))

  def print_meta(unique_meta_info):
    print("[INFO] Midi Utils - All meta info:")
    for msg, j in unique_meta_info:
      print("        %s [%d]" % (msg, j))

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
  music_tempos = []

  # Get ALL of the tempos (people may change the tempo to change how
  # the sheet music fits on the pages)
  for msg, j in unique_meta_info:
    if msg.type == 'set_tempo':
      music_tempos.append((msg.tempo, j))

  assert len(music_tempos) > 0

  # If the tempo matches what we need, no work is necessary. 
  if len(music_tempos) == 1 and music_tempos[0][0] == music_set_tempo:
    return midi

  print("[INFO] Midi Utils - Refactoring %d midi tempo(s): %s -> %d." % (len(music_tempos), str(music_tempos), music_set_tempo))
  #print_first_x(midi, 30)
  #print(midi.ticks_per_beat)
  #input()

  # Otherwise we NEED to revamp the entire midi file such that the 
  # requisite tempo is used. All of our training data is at 500,000, 
  # so to have the best generalization our incoming data needs to be
  # converted to 500,000 as well. Get the absolute seconds since 
  # start for all notes before converting.
  #
  # There will be a slight, largely unnoticable difference between
  # the duration of the source and final. (~0.5 seconds for 120 
  # second songs). TODO to improve.
  def change_tempo_range(track, start_i, end_i, orig_tempo):
    """ Given a range of msg indices, change msg times. """
    absolute_seconds = []
    absolute_time_since_start = 0
    for j in range(start_i, end_i):
      msg = track[j]
      additional_seconds = tick2second(msg.time, midi.ticks_per_beat, orig_tempo)
      absolute_time_since_start += additional_seconds
      absolute_seconds.append(absolute_time_since_start)

    # And now map the absolute seconds to the new times of every message. 
    time_of_last = 0
    for j in range(start_i, end_i):
      track[j].time = round(second2tick(absolute_seconds[j - start_i] - time_of_last, midi.ticks_per_beat, music_set_tempo))
      time_of_last = absolute_seconds[j - start_i]
    return track

  for i in range(0, len(music_tempos)):
    # Calculate the indices of the changes we need to make. 
    if i < len(music_tempos) - 1:
      end_i = music_tempos[i+1][1]
    else:
      end_i = len(midi.tracks[0])
    start_i = music_tempos[i][1]
    orig_tempo = music_tempos[i][0]
    midi.tracks[0] = change_tempo_range(midi.tracks[0], start_i, end_i, orig_tempo)

  # Change the tempos of all meta messages.  
  for msg, _ in unique_meta_info:
    if msg.type == 'set_tempo':
      msg.tempo = music_set_tempo

  #print("[DEBUG] Midi Utils - Done. Result:")
  #print_first_x(midi, 30)
  #input()
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

  absolute_time_since_start = 0
  absolute_time_of_last_note = 0
  for j in range(0, len(track)):
    msg = track[j]
    additional_seconds = tick2second(msg.time, midi.ticks_per_beat, music_set_tempo)
    absolute_time_since_start += additional_seconds

    if msg.type == "note_on" or msg.type == "note_off":
      note = msg.note

      # Improvement - do NOT use the time of the note, which, in our
      # training dataset at least, is the time since the last CONTROL
      # CHANGE (most likely)
      seconds_since_last_note = absolute_time_since_start - absolute_time_of_last_note
      absolute_time_of_last_note = absolute_time_since_start

      if msg.type == "note_off":
        note_on = 0
      elif msg.velocity == 0:
        note_on = 0
      else:
        note_on = 1
      
      row = (note, seconds_since_last_note, note_on)
      rows.append(row)
  
  # Add a final note at the end of the song, whose main purpose is to
  # provide seconds_since_last information for control changes.
  seconds_since_last_note = absolute_time_since_start - absolute_time_of_last_note
  rows.append((60, seconds_since_last_note, 0))

  # We've now gathered all of the rows, ignoring any velocity info
  # that is not 0. Return this dataframe. 
  song_df = pd.DataFrame(
    rows, 
    columns=[data_note_col, data_time_col, data_note_on_col])

  return song_df

def extract_offsets(midi: MidiFile, data: pd.DataFrame,
                    online_midi: MidiFile):
  """
  Given the preprocessed midi, it's initial X dataframe, as well as
  the online midi from which offset values are to be derived, extract
  offsets. Perhaps the most complex part of this entire preprocessing.

  1. Verify the performance midi against the online midi. Remove extra
     notes from the MIDI as well as the X dataframe. Ignore any notes
     missing in the performance midi present in the online one. By all
     accounts, these notes have never existed and processing (including
     for control changes) will never know they were there.

  2. For each remaining note in the performance midi, calculate the
     OFFSET from when it is played in the online midi and when it is 
     actually played - the subjective timing added either purposefully
     or inadvertently that we wish to emulate with out model. 

     These offsets should be calculated with absolute seconds in mind.
     Starting from the absolute seconds of the last processed note, 
     how long was the delay between that and the performed note? How
     long was the delay for the online equivalent? (The last processed
     note should be the same note regardless of order). The offset is
     the difference between these delays and can be positive or 
     negative. 
     
     The "last note" to use for this should be the last note in the 
     ONLINE midi. This may result in wonkiness, with the performed
     note perhaps having happened BEFORE the "last note" in the online
     midi. This is fine. 
      
  3. Save all offsets in a list to be added as a final column to the 
     dataframe in the next preprocessing step. 

  4. Replace all times in the dataframe with the absolute seconds since
     the last note in the ONLINE midi, overwriting data from the  
     performed midi. DO NOT make these same changes to the midi itself,
     as the control changes will continue to be processed as if the 
     times were the same. We are essentially splitting the genuine
     time of the performed notes into two pieces of info - the online
     midi's timing, as well as the offset information for the model
     to predict. 
  """
  return midi, data, []


def generate_sol_dataframe(midi: MidiFile, data: pd.DataFrame,
                           offsets: list):
  """
  Given the preprocessed midi as well as the dataframe of the extracted
  rows, generate solutions. This goes back and uses the non-zero 
  velocities, assuming the subject midi file is actually a recording
  of a human playing. It also goes through and processes control  
  changes, adding the control changes happening in the aftermath of a
  single note as part of the solution vector. 
  """
  # Only one track allowed.
  assert len(midi.tracks) == 1

  rows = []
  track = midi.tracks[0]

  def generate_x_empty_controls(x):
    """ Generate a 2 * x sized array of 0s. """
    return_list = []
    for _ in range(0, x):
      return_list.append(0)
      return_list.append(0)
    return return_list

  def average_pad_control_changes(control_changes, p, total_time_to_next_note):
    """
    Given a list of tuples for control changes (value, time), do one
    of three things depending on the maximum granularity (p). 
    1. num_changes < p - in which case, pad up to p. 
    2. num_changes == p - in which case, we're done.
    3. num_changes > p - sample equidistant values.

    Every single time value needs to be "squished" between 0 and 1,
    with 1 being the time of the next note, and 0 being the time of
    the current note. 

    Returns a list of finalized control changes, unpacking all tuples
    into a single vector. 
    """

    def squish_time(time, total_time_to_next_note = total_time_to_next_note):
      """
      Given an end time (Ex) 53), as well as a time (Ex) 23), calculate
      the percentage of progress from 0 to end time. (Ex) 0.43396226)
      """
      assert total_time_to_next_note >= 0
      if total_time_to_next_note == 0:
        return 0
      squished_time = float(time) / float(total_time_to_next_note)
      assert squished_time >= 0.0 and squished_time <= 1.0
      return squished_time

    def even_select(N, M):
      """ 
      Given N elements, sample M (<= N) values as evenly spaced as
      possible. Fantastic code... not mine! 
      https://stackoverflow.com/questions/46494029/nearly-evenly-select-items-from-a-list 
      """
      if M > N/2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N-M)
        indices = [q*i + min(i, r) for i in range(N-M)]
        cut[indices] = True
      else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        indices = [q*i + min(i, r) for i in range(M)]
        cut[indices] = False

      return cut

    controls_vector = []
    if len(control_changes) <= p:
      # 1 & 2) Pad up to p if necessary.
      for item in control_changes:
        controls_vector.append(item[0]) # value
        controls_vector.append(squish_time(item[1])) # time
      # Pad the remainder (if necessary) 
      for _ in range(0, (p - len(control_changes))):
        controls_vector.append(0)
        controls_vector.append(0)
    else:
      # 3) Sample p equidistant values. 
      assert p >= 2
      # Always take the first control change.
      controls_vector.append(control_changes[0][0])
      controls_vector.append(squish_time(control_changes[0][1]))

      # For the values in between, sample equidistantly. 
      cut = even_select(len(control_changes) - 2, p - 2)
      assert len(cut) == len(control_changes) -2
      for i in range(len(cut)): 
        index = i + 1 # offset against first control change. 
        if cut[i] == 0:
          item = control_changes[index]
          controls_vector.append(item[0]) # value
          controls_vector.append(squish_time(item[1])) # time

      # Always add the final control change. 
      controls_vector.append(control_changes[-1][0])
      controls_vector.append(squish_time(control_changes[-1][1]))
      
    assert len(controls_vector) == p*2
    return controls_vector

  # Add the solution to the first note. It's obviously velocity 0.
  empty_note = (0, 0)
  note_msgs = [empty_note]
  for j in range(0, len(track)):
    msg = track[j]
    if msg.type == "note_on" or msg.type == "note_off":
      sol_velocity = msg.velocity
      note_msgs.append((j, sol_velocity))
  # Add the solutions for the final row. 
  note_msgs.append((len(track), 0))

  # For each note that we read, add control change information between
  # when it is played and when the next note (or end of song) occurs.
  for i in range(0, len(note_msgs)):
    note_index, sol_velocity = note_msgs[i][0], note_msgs[i][1]

    # Figure out the last note between which we'll be processing 
    # control changes. 
    if i < (len(note_msgs) - 1):
      next_note_index = note_msgs[i + 1][0]
    else:
      # End of the song. 
      next_note_index = len(track)
    
    new_row = [sol_velocity]

    if note_index == next_note_index:
      # Edge case of the first note being the first message. Fill with
      # p empty controls. 
      new_row += generate_x_empty_controls(p_granularity_total)
    else:
      # For the time of controls, we cannot use the ACTUAL time,
      # as trying predict the actual time will lead to chaos when
      # postprocessing the result. Instead, we will add a value 
      # from 0 to 1 indicating percentage of time to the next note. 
      # Make this easier for everyone. 
      total_time_to_next_note = 0

      # Get all of the controls between this note and the next. 
      control_64 = [] # Sustain
      control_66 = [] # Sostenuto 
      control_67 = [] # Soft
      num_changes = 0
      for j in range(note_index+1, next_note_index):
        msg = track[j]
        if msg.type == "control_change":
          value = msg.value
          time = msg.time
          if msg.control == 64:
            control_64.append((value, total_time_to_next_note + time))
          elif msg.control == 66:
            control_66.append((value, total_time_to_next_note + time))
          elif msg.control == 67:
            control_67.append((value, total_time_to_next_note + time))
          else:
            print("[ERROR] Midi Utils - Encountered unknown control %d!" % msg.control)
            assert False
          num_changes += 1
          total_time_to_next_note += msg.time
      
      # Add the amount of time between the last control change and
      # the next note. 
      if next_note_index < len(track):
        total_time_to_next_note += track[next_note_index].time

      # Now we have all the control changes. We have a few options here.
      # For each p (64, 66, and 67), average or pad if necessary.
      new_row += average_pad_control_changes(control_64, p_granularity_64, total_time_to_next_note)
      new_row += average_pad_control_changes(control_66, p_granularity_66, total_time_to_next_note)
      new_row += average_pad_control_changes(control_67, p_granularity_67, total_time_to_next_note)

    # At the end, do some sanity checks. 
    assert len(new_row) == ((p_granularity_total*2) + 1)
    rows.append(new_row)

  # Make sure the columns are labeled properly and generate the final
  # dataframe to be exported. 
  solution_df = pd.DataFrame(rows, columns = data_solution_cols)
  assert data.shape[0] == solution_df.shape[0] # Num rows. 

  # Combine the solutions Y with the X dataframe.
  combined_data = pd.concat(objs=[data, solution_df], axis=1)

  return combined_data