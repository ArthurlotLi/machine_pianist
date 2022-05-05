#
# midi_test.py
#
# Misc utility meant to explicitly show the contents of the midi
# files provided as part of the MAESTRO Dataset. 

from utils.midi_player import PianoPlayer, print_first_x

from mido import MidiFile
_test_file = "./midi_test/temp_file"
#_test_file = "./midi_test/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--1.midi"
_test_file2 = "./midi_test/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi"
_downloaded_file = "./midi_test/Online Version.mid"

_spider_dance = "./midi_test/Undertale_-_Spider_Dance_-_Lattice.mid"

def output_midi_lines(lines_to_output):
  """ 
  Output x files of our test file.
  """
  output_target = _test_file2
  mid_test = MidiFile(output_target)
  print("\n[INFO] MIDI Test -First %s lines of %s:" % (lines_to_output, output_target))
  print_first_x(mid_test, lines_to_output)

def repeat_play_section(duration):
  """
  For testing, play the first x seconds of the test file. Indefinitely.
  """
  player = PianoPlayer()

  while True:
    print("[INFO] MIDI Test - Playing MAESTRO dataset MIDI file for %d seconds." % duration)
    player.local_load_and_play(_test_file, play_for=duration, verbose = False)

def compare_test_to_downloaded(demo_duration_natural, demo_duration_artificial, lines_to_output):
  """
  A proof of concept experiment - compare the midi file of a particular
  song (Prelude No. 14 in F# Minor - Johann Sebastian Bach) from the
  MAESTRO dataset to one downloaded from musescore online. 

  The two MIDI files, while representing the same song, should sound
  completely dissimilar in that the latter will not contain sustain
  information or force information that is generated only from a 
  live performance. 

  It makes little sense for online composers to provide this information
  when the intent is to serve sheet music (that doesn't indicate this
  info). Thus, the niche for this ML project. 
  """
  player = PianoPlayer()

  print("[INFO] MIDI Test - Playing MAESTRO dataset MIDI file. This should sound natural...")
  player.local_load_and_play(_test_file, play_for=demo_duration_natural, verbose = False)
  print("[INFO] MIDI Test - Playing online-sourced MIDI files. This should NOT sound natural...")
  player.local_load_and_play(_downloaded_file, play_for=demo_duration_artificial, verbose = False)

  print("\n[INFO] MIDI Test - Comparison of first 10 lines of both MIDI files:")
  mid_test = MidiFile(_test_file)
  mid_download = MidiFile(_downloaded_file)
  
  print("\n[INFO] MIDI Test - MAESTRO MIDI:")
  print_first_x(mid_test, lines_to_output)
  print("\n[INFO] MIDI Test - Online-sourced MIDI:")
  print_first_x(mid_download, lines_to_output)