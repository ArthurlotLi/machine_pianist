#
# postprocess.py
#
# Given the output of the model, combine it with the original MIDI
# file to produce the final augmented MIDI file with human-like 
# qualities (hopefully).

from tqdm import tqdm

def generate_output_midi(preprocessed_songs: list, Y_hat: list):
  """
  Expects a list of tuples for preprocessed songs - (song_id, midi).
  Also expects the output of the model, which is a tensor of shape
  (# songs, maximum_song_length, 1). Inserts all velocities and 
  ignores all original note-offs. Adds all metadata as well. 

  Returns a mido midi object. 

  # TODO: Handle the control change data. 
  """
  assert len(preprocessed_songs) == Y_hat.shape[0]

  output_midis = []
  for i in tqdm(range(0, len(preprocessed_songs)), desc="[INFO] Postprocess - Applying performance"):
    midi = preprocessed_songs[i][1]
    midi_predictions = Y_hat[i]

    # Only one track allowed.
    assert len(midi.tracks) == 1

    # Ignore the first note of the predictions, as this is the 
    # padding note we always add in midi_utils. Ultimately, 
    # we should get to the last note in our midi file and then
    # finish, ignoring the rest of the padded notes added to the
    # end. 
    predictions_index = 1
    for msg in midi.tracks[0]:
      if msg.type == "note_on" or msg.type == "note_off":
        
        if msg.type != "note_off" and msg.velocity != 0:
          # This is a note on. Ignore whatever the model predicts for
          # note off. 
          msg.velocity = abs(round(midi_predictions[predictions_index][0]))

        predictions_index += 1
    
    output_midis.append(midi)
  
  # All done! 
  return output_midis