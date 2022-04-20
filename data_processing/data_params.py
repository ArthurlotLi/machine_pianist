#
# data_params.py
#
# Configuration for the way the data is preprocessed/postprocessed
# before and after training and inference. These should stay constant
# after a model has been trained. 

# Standardized MetaMessage information. All incoming tracks should
# be converted to match this tempo. 
music_set_tempo = 500000 # Microseconds per quarter note.

# An incredibly important hyperparameter. p is the number of supported
# control changes between notes @ t-1 and t. If there are more than this
# number of control changes in the source, the control changes will be
# averaged into the best approximation given the lowered resolution. 
# 
# Increasing this may potentially confuse the model but also may
# increase the naturalness of pedal changes. Decreasing simplifies
# the song, but may make things easier. Experiment with this, and change
# the capacity of the model in tandem. 
p_granularity_64 = 15 # Sustain pedal info.
p_granularity_66 = 5 # Sostenuto pedal info.
p_granularity_67 = 5 # Soft pedal info.
p_granularity_total = p_granularity_64 + p_granularity_66 + p_granularity_67

# Maestro dataset info. 
#maestro_dataset_folder = "maestro_test" # TODO: Obvi, change me.
maestro_dataset_folder = "maestro-v3.0.0"
maestro_dataset_csv = "maestro-v3.0.0.csv"

data_train_csv = "train.csv"
data_test_csv = "test.csv"

data_note_col = "note"
data_time_col = "time_since_last"
data_note_on_col = "note_on"
data_uid_col = "song"
data_velocity_col = "velocity"

data_solution_cols = [data_velocity_col]
for i in range(0, p_granularity_64): 
  data_solution_cols.append("64v%d" % i)
  data_solution_cols.append("64t%d" % i)
for i in range(0, p_granularity_66): 
  data_solution_cols.append("66v%d" % i)
  data_solution_cols.append("66t%d" % i)
for i in range(0, p_granularity_67): 
  data_solution_cols.append("67v%d" % i)
  data_solution_cols.append("67t%d" % i)
assert len(data_solution_cols) == ((p_granularity_total*2) + 1)