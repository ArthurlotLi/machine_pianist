#
# midi_player.py
#
# Simple utility for visualization and testing - play a midi file
# sample.

import pygame

class PianoPlayer:
  def __init__(self):
    print("[DEBUG] Initializing PianoPlayer...")
    pygame.init()
    print("[DEBUG] PianoPlayer initialized successfully.")

  # Given a directory path, load and play a midi file locally on
  # this computer. 
  def local_load_and_play(self, location, play_for=None, block=False, verbose=True):
    if verbose: print("[INFO] PianoPlayer playing song located: " + str(location) + ".")
    try:
      pygame.mixer.music.load(location)
      pygame.mixer.music.play()
      if block or (play_for is not None and play_for > 0):
        # If we're blocking, wait until we're done. 
        if verbose: print("[DEBUG] PianoPlayer blocking...")
        while pygame.mixer.music.get_busy() and play_for is None or play_for > 0:
          pygame.time.wait(500)
          play_for -= 0.5
        if verbose: print("[DEBUG] PianoPlayer song complete.")
    except Exception as e:
      print("[ERROR] PianoPlayer was unable to locally play song from location '" + str(location) + "'. Exception: ")
      print(e)