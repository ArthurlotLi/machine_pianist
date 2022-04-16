#
# hparams.py
#
# Various configurable hparams for the produced model. These affect
# the performance of the final model in a wide variety of ways and 
# should be tweaked experimentally to find the best fit. 

# Prevent tensorflow from automatically hogging the entire GPU. This
# should only be enabled to view VRAM loads in real time, and disabled
# during proper training. 
allow_tf_growth = False

training_verbose = True

# To load models, this filename is expected:
saved_model_name = "machine_pianist.h5"

# Learning Hyperparameters.
learning_rate = 0.0001
epochs = 1000
batch_size = 32
validation_split = 0.2

# Our output layer is linear with a single output vector - thus
# MSE is the preferred loss function under maximum likelihood
# (distribution of the target variable is Gaussian).
loss_function = "mean_squared_error" 
metrics = ["mse"]

# The key that scales the model + the songs it can play. All songs,
# when read it, will be padded up to this length with note-offs
# at time=0. Songs with more notes than this will be rejected.
maximum_song_length = 30000 

# Model architecture. We keep all layers at equal length
# Expects (timesteps, feature).
input_dim = (maximum_song_length,3) 
gru_width = 64
gru_depth = 2 # minimum of 1.
assert gru_depth >= 1
fully_connected_width = 64

# Additional Regularization.
input_dropout = 0.8
hidden_dropout = 0.5

# Callbacks. 
mcp_monitor = "mse"
mcp_save_best_only = False