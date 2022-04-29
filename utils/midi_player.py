#
# midi_player.py
#
# Simple utility for representation and testing - play a midi file
# sample.

# Avoid the welcome message. 
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame 
import time
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
      if verbose: start_time = time.time()
      pygame.mixer.music.play()
      if block or (play_for is not None and play_for > 0):
        # If we're blocking, wait until we're done. 
        if verbose: print("[DEBUG] PianoPlayer blocking...")
        while pygame.mixer.music.get_busy() and (play_for is None or play_for > 0):
          pygame.time.wait(500)
          if play_for is not None:
            play_for -= 0.5
        if verbose: print("[DEBUG] PianoPlayer completed in %.2f seconds." % (time.time()-start_time))
    except Exception as e:
      print("[ERROR] PianoPlayer was unable to locally play song from location '" + str(location) + "'. Exception: ")
      print(e)

def print_first_x(mid, x, notes_only = False):
  """ 
  Print the first x lines of a midi files.  
  """
  for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    if notes_only:
      notes = []
      for j in range(0, len(track)):
        if track[j].type == "note_on":
          notes.append(track[j])
      for j in range(0, min(x, len(notes))):
        print(notes[j])
    else:
      for j in range(0, min(x, len(track))):
        print(track[j])
    
def midi_statistics(mid):
  """ 
  Print the number of notes in the song. 
  """
  for i, track in enumerate(mid.tracks):
    print('[INFO] Midi Player - Track {}: {} - Length: {}'.format(i, track.name, str(len(track))))

def graph_velocities(mid):
  """
  Visualize the velocities of a midi file. Only one track is used.
  Meant to demonstrate the behavior of human vs automatic Midi 
  performances. 
  """
  import matplotlib.pyplot as plt
  graph_width_inches = 13
  graph_height_inches = 7

  assert len(mid.tracks) == 1

  velocity_history = []
  for j in range(0, len(mid.tracks[0])):
    msg = mid.tracks[0][j]
    if msg.type == "note_on":
      velocity_history.append((j, msg.velocity))
  
  # Graph the history. 
  fig = plt.figure(1)
  fig.suptitle("Machine Pianist - Note velocity over time")
  fig.set_size_inches(graph_width_inches,graph_height_inches)
  plt.scatter(*zip(*velocity_history))
  plt.xlabel("MIDI Message")
  plt.ylabel("Note Velocity")
  plt.show()

def graph_velocities_notes(mid):
  """
  Visualize the velocities AND notes of a midi file. Only one track 
  is used. Mean to visualize the progression of the song alongside
  the velocities predicted by the model. 
  """
  import matplotlib.pyplot as plt
  graph_width_inches = 13
  graph_height_inches = 7

  assert len(mid.tracks) == 1

  note_history = []
  note_history_color = "grey"
  velocity_history = []
  velocity_history_color = None
  for j in range(0, len(mid.tracks[0])):
    msg = mid.tracks[0][j]
    if msg.type == "note_on":
      note_history.append((j, msg.note))
      velocity_history.append((j, msg.velocity))
  
  # Graph the history. 
  fig = plt.figure(1)
  fig.suptitle("Machine Pianist - Notes and Velocities over time")
  fig.set_size_inches(graph_width_inches,graph_height_inches)
  plt.scatter(*zip(*note_history), color=note_history_color)
  plt.scatter(*zip(*velocity_history), color=velocity_history_color)
  #plt.scatter(*zip(*velocity_history, *note_history), color=["white", "black"])
  plt.xlabel("MIDI Message")
  plt.ylabel("Note Velocity/Note ID")
  plt.legend(['Song Notes', 'Velocities'], loc="upper left")
  plt.show()

def graph_controls_notes(mid):
  """
  Visualize the velocities AND notes of a midi file AS WELL AS the 
  control changes. Meant to demonstrate the entire picture of 
  the performance data. 
  """
  import matplotlib.pyplot as plt
  graph_width_inches = 13
  graph_height_inches = 7

  assert len(mid.tracks) == 1

  note_history = []
  note_history_color = "grey"
  control_64_history = []
  control_64_color = "green"
  control_67_history = []
  control_67_color = "purple"
  for j in range(0, len(mid.tracks[0])):
    msg = mid.tracks[0][j]
    if msg.type == "note_on":
      note_history.append((j, msg.note))
    elif msg.type == "control_change":
      control = msg.control 
      if control == 64:
        control_64_history.append((j, msg.value))
      elif control == 67:
        control_67_history.append((j, msg.value))
  
  # Graph the history. 
  fig = plt.figure(1)
  fig.suptitle("Machine Pianist - Notes and Velocities over time")
  fig.set_size_inches(graph_width_inches,graph_height_inches)
  plt.scatter(*zip(*note_history), color=note_history_color)
  if len(control_64_history) > 0:
    plt.scatter(*zip(*control_64_history), color=control_64_color)
  if len(control_67_history) > 0:
    plt.scatter(*zip(*control_67_history), color=control_67_color)
  #plt.scatter(*zip(*velocity_history, *note_history), color=["white", "black"])
  plt.xlabel("MIDI Message")
  plt.ylabel("Control Value/Note ID")
  plt.legend(['Song Notes', 'Sustain Control (64)', 'Soft Control (67)'], loc="upper left")
  plt.show()