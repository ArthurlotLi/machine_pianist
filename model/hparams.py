#
# hparams.py
#
# Various configurable hparams for the produced model. These affect
# the performance of the final model in a wide variety of ways and 
# should be tweaked experimentally to find the best fit. 

# Prevent tensorflow from automatically hogging the entire GPU. This
# should only be enabled to view VRAM loads in real time, and disabled
# during proper training. 
allow_tf_growth = True

# Learning Hyperparameters.
learning_rate = 0.0001 
epochs = 5
batch_size = 64
validation_split = 0.2

# Our output layer is linear with a single output vector - thus
# MSE is the preferred loss function under maximum likelihood
# (distribution of the target variable is Gaussian).
loss_function = "mean_squared_error" 
metrics = ["mse"]

# Model architecture. We keep all layers at equal length
input_dim = (4,1) # 4 input columns: note, time, note_on, song.
gru_width = 128
gru_depth = 3 # minimum of 1.
assert gru_depth >= 1

# Additional Regularization.
input_dropout = 0.8
hidden_dropout = 0.5

# Callbacks. 
mcp_monitor = "mse"
mcp_save_best_only = False