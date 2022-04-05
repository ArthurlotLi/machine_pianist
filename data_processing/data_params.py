#
# data_params.py
#
# Configuration for the way the data is preprocessed/postprocessed
# before and after training and inference. These should stay constant
# after a model has been trained. 

# Standardized MetaMessage information. All incoming tracks should
# be converted to match this tempo. 
music_set_tempo = 500000 # Microseconds per quarter note.
music_time_signature_numerator = 4
music_time_signature_denominator = 4
music_clocks_per_click = 24
music_notated_32nd_notes_per_beat = 8

# An incredibly important hyperparameter. p is the number of supported
# control changes between notes @ t-1 and t. If there are more than this
# number of control changes in the source, the control changes will be
# averaged into the best approximation given the lowered resolution. 
# 
# Increasing this may potentially confuse the model but also may
# increase the naturalness of pedal changes. Decreasing simplifies
# things, but may make things easier. Experiment with this, and change
# the capacity of the model in tandem. 
p_granularity = 10

# Maestro dataset info. 
maestro_dataset_folder = "maestro_test" # TODO: Obvi, change me.
#maestro_dataset_folder = "maestro-v3.0.0"
maestro_dataset_csv = "maestro-v3.0.0.csv"