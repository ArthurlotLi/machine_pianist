#
# proof_of_concept.py
#
# Simple utility meant to demonstrate the inherent differences between 
# a typical input MIDI file and an "naturally sourced" MIDI file.
#
# Also includes various utilities to investigate a variety of MIDI
# files sourced from online. Sources:
# [1] https://musescore.com/classicman/scores/1444781
# [2] https://musescore.com/user/441326/scores/5543583 
#
# Usage: 
# python proof_of_concept.py

from midi_test.midi_test import *

import argparse

repeat_duration = 4

print_lines_amount = 200

demo_duration_natural = 30
demo_duration_artificial = 30
lines_to_output = 150

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", default=False, action="store_true")
  parser.add_argument("-p", default=False, action="store_true")
  args = parser.parse_args()

  if args.r:
    repeat_play_section(repeat_duration)
  elif args.p:
    output_midi_lines(print_lines_amount)
  else:
    compare_test_to_downloaded(demo_duration_natural = demo_duration_natural, 
                              demo_duration_artificial = demo_duration_artificial,
                              lines_to_output = lines_to_output)