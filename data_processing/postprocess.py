#
# postprocess.py
#
# Given the output of the model, combine it with the original MIDI
# file to produce the final augmented MIDI file with human-like 
# qualities (hopefully)

from data_processing.data_params import *
from model.hparams import *

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from mido import MidiTrack, Message, tick2second, second2tick, merge_tracks
import numpy as np
import pandas as pd

def generate_output_midi(preprocessed_songs: list, Y_hat: list, X: np.array,
                         scaler_X: StandardScaler, scaler_Y: StandardScaler):
  """
  Expects a list of tuples for preprocessed songs - (song_id, midi).
  Also expects the output of the model, which is a tensor of shape
  (# songs, maximum_song_length, 1). Inserts all velocities and 
  ignores all original note-offs. Adds all metadata as well. 

  Returns a mido midi object. 
  """
  assert len(preprocessed_songs) == Y_hat.shape[0]

  # Reverse scaling of the provided X so we can use it. 
  X_orig_shape = X.shape
  X = scaler_X.inverse_transform(pd.DataFrame(X.reshape(-1, len(data_input_cols)), columns=data_input_cols))
  X = X.reshape(X_orig_shape)

  output_midis = []
  for a in tqdm(range(0, len(preprocessed_songs)), desc="[INFO] Postprocess - Applying performance"):
    midi = preprocessed_songs[a][1]
    midi_predictions = Y_hat[a]

    # Before anything else, reverse scaling.
    midi_predictions = scaler_Y.inverse_transform(pd.DataFrame(midi_predictions, columns=data_solution_cols))

    # Only one track allowed.
    assert len(midi.tracks) == 1

    def control_solutions(prediction_row):
      """
      Helper function for controls. Extracts ALL control data from a
      prediction row. 
      """
      def process_controls(control_changes, current_index, 
                            prediction_row, p, control_num):
        """
        Helper function for controls. For each control (pedal),
        extract the value and time percentage. Drop all values
        that are all 0s. 
        """
        for _ in range(0, p):
          current_index += 1
          value = prediction_row[current_index]
          current_index += 1
          time_percentage = prediction_row[current_index]

          clamp_values = False

          if clamp_values:
            # Allow the model to specify control changes to a certain
            # degree, without allowing it to fully control changes. 
            # Use the "confidence" of the model to output a pedal 
            # position. 
            """
            sorted_control_cutoffs = {
              0.65: 0,
              0.7: 85,
              0.75: 92,
              0.8: 95,
              0.85: 102,
              0.9: 108,
              0.95: 117,
            }
            """
            sorted_control_cutoffs = {
              0.55: 0,
            }

            cutoff_applied = False
            for cutoff in sorted_control_cutoffs:
              if value <= cutoff:
                value = sorted_control_cutoffs[cutoff]
                cutoff_applied = True
                break
            if cutoff_applied is False:
              value = 127
          else:
            value = max(min(round(127*value), 127), 0)

          control_changes[time_percentage] = (control_num, value)
        return current_index, control_changes

      # Process the control change data. Control change data comes
      # in twos - (value, time_percentage). The latter is a value
      # that indicates when the control change happens between the
      # subject note and the next note. 
      control_changes = {} # Dict keyed by time percentage. 
      current_index = 0
      current_index, control_changes = process_controls(control_changes, current_index, 
                                                        prediction_row,
                                                        p_granularity_64, 64)
      #current_index, control_changes = process_controls(control_changes, current_index, 
                                                        #prediction_row,
                                                        #p_granularity_66, 66)
      #current_index, control_changes = process_controls(control_changes, current_index, 
                                                        #prediction_row,
                                                        #p_granularity_67, 67)
      assert current_index == len(prediction_row)- 1

      # Now we need to make this chronological, regardless of which
      # control. 
      sol_controls = dict(sorted(control_changes.items(), key=lambda item: item[0]))
      return sol_controls

    # Ignore the first note of the predictions, as this is the 
    # padding note we always add in midi_utils. Ultimately, 
    # we should get to the last note in our midi file and then
    # finish, ignoring the rest of the padded notes added to the
    # end. 
    predictions_index = 1
    notes = []

    # Add first note control information.
    sol_controls = control_solutions(midi_predictions[0])
    notes.append((60, 0, 0, sol_controls))

    absolute_time_since_start = 0
    absolute_time_last_msg = 0
    for i in range(len(midi.tracks[0])):
      msg = midi.tracks[0][i]

      additional_seconds = tick2second(msg.time, midi.ticks_per_beat, music_set_tempo)
      absolute_time_since_start += additional_seconds

      if msg.type == "note_on" or msg.type == "note_off":
        # Parse velocity information. 
        if msg.type != "note_off" and msg.velocity != 0:
          # This is a note on. Ignore whatever the model predicts for
          # note off. 
          prediction_row = midi_predictions[predictions_index]
          sol_velocity = abs(round(prediction_row[0]))
        else:
          sol_velocity = 0

        # Parse control change information.
        sol_controls = control_solutions(prediction_row)

        notes.append((msg.note, absolute_time_since_start - absolute_time_last_msg, sol_velocity, sol_controls))
        absolute_time_last_msg = absolute_time_since_start
        predictions_index += 1

    # Get the note after the very last note - this should be a dummy
    # note. We just care about the end song seconds value.
    end_row = X[a][predictions_index]
    assert end_row[0] == 60 and end_row[2] == 0
    end_song_seconds = end_row[1]

    # We have now filled out our notes list, ordered chronologically.
    # We now need to generate our output midi, which will have all
    # the information harmonized within. Each entry in this list should
    # be a tuple: (note, time, velocity, controls), with the controls dict
    # being a sorted dict by time_percentage - values being tuples
    # of (control num, value).
    meta_msgs = []
    for msg in midi.tracks[0]:
      if msg.is_meta and msg.type != "end_of_track":
        # We force all meta messages to be 0. Since we harmonize all
        # tempos, this should have no effect. Ignore the end of track
        # message - we'll add that ourselves. 
        msg.time = 0
        meta_msgs.append(msg)
      
    output_track = MidiTrack()
    # Add all meta messages to the output track at the beginning.
    for msg in meta_msgs: output_track.append(msg)

    # Create the output track. 
    seconds_between_last_note_and_last_msg = 0
    for i in range(0,len(notes) - 1):
      note_info = notes[i]
      note = round(note_info[0])
      assert note_info[1] >= seconds_between_last_note_and_last_msg
      time = round(second2tick(note_info[1] - seconds_between_last_note_and_last_msg, 
                               midi.ticks_per_beat, music_set_tempo))
      velocity = round(note_info[2])
      controls = note_info[3]
      new_note = Message(type="note_on", note=note, time=time, 
                         velocity=velocity)
      output_track.append(new_note)

      # Starting from our current note, set this up for the next guy. 
      seconds_between_last_note_and_last_msg = 0

      # Get the absolute time of the next note, or end of the track. 
      if i < (len(notes)-2):
        time_to_next_note_seconds = notes[i+1][1]
      else:
        time_to_next_note_seconds = end_song_seconds

      if len(controls) > 0:
        # Given ticks to end, append notes. 
        for time_percentage in controls:
          control_change = controls[time_percentage]
          control_num = control_change[0]
          value = round(control_change[1])
          value = min(value, 127)
          value = max(value, 0)

          # Account for some odd numbers that might leak out of the
          # model.   
          if time_percentage < 0:
            time_percentage = 0

          # Calculate the ticks to add. 
          time_since_subject_note = time_to_next_note_seconds*time_percentage
          assert time_since_subject_note >= seconds_between_last_note_and_last_msg
          time_since_last_msg = time_since_subject_note - seconds_between_last_note_and_last_msg
          time = round(second2tick(time_since_last_msg, midi.ticks_per_beat, music_set_tempo))

          new_control = Message(type="control_change", control=control_num, 
                                value = value, time=time)
          output_track.append(new_control)

          seconds_between_last_note_and_last_msg =time_to_next_note_seconds - (time_to_next_note_seconds - time_since_subject_note)

    # The output_track has now been fully created. Overwrite the
    # original midi's track. And we're dooooone! 
    # 
    # Note: use merge_tracks with a single track because this function
    # deliberately fixes any end of track messages and wraps things
    # up. 
    midi.tracks[0] = merge_tracks([output_track])

    output_midis.append(midi)
  
  # All done! 
  return output_midis