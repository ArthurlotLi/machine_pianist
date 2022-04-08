#
# midi_player.py
#
# Simple utility for representation and testing - play a midi file
# sample.

# Avoid the welcome message. 
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame 
from mido import MidiFile

class PianoPlayer:
  _mido_temp_file = "temp_file"

  def __init__(self):
    print("[DEBUG] Initializing PianoPlayer...")
    pygame.init()
    print("[DEBUG] PianoPlayer initialized successfully.")

  def play_mido(self, midi: MidiFile, play_for=None, block=False, verbose=True):
    """
    Given a mido midi file, save the midi file to a temp file and
    play it, deleting it afterwards. 
    """
    midi.save(self._mido_temp_file)
    self.local_load_and_play(self._mido_temp_file, play_for=play_for, block=block, verbose=verbose)
    os.remove(self._mido_temp_file)

  def local_load_and_play(self, location, play_for=None, block=False, verbose=True):
    """
    Given a directory path, load and play a midi file locally on
    this computer. 
    """
    if verbose: print("[INFO] PianoPlayer playing song located: " + str(location) + ".")
    try:
      pygame.mixer.music.load(location)
      pygame.mixer.music.play()
      if block or (play_for is not None and play_for > 0):
        # If we're blocking, wait until we're done. 
        if verbose: print("[DEBUG] PianoPlayer blocking...")
        while pygame.mixer.music.get_busy() and (play_for is None or play_for > 0):
          pygame.time.wait(500)
          if play_for is not None:
            play_for -= 0.5
        if verbose: print("[DEBUG] PianoPlayer song complete.")
    except Exception as e:
      print("[ERROR] PianoPlayer was unable to locally play song from location '" + str(location) + "'. Exception: ")
      print(e)

def print_first_x(mid, x):
  """ 
  Print the first x lines of a midi files.  
  """
  for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for j in range(0, min(x, len(track))):
      print(track[j])