#
# model.py
#
# Definition of the model itself. A deep recurrent neural network 
# designed for equal-length sequence-to-sequence inference. Using
# TensorFlow, simply because it's been a while since I used it. 

from re import M
from model.hparams import *
from data_processing.data_params import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Bidirectional
from tensorflow.keras.layers import GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from pathlib import Path

def machine_pianist(saved_models:Path, model_id: str):
  """
  Defines the model structure according to the hparams specified. 
  Proceeds to compile with the requisite optimizer + callbacks.
  
  Returns the model + list of callbacks. 
  """
  print("[INFO] Model - Defining Machine Pianist model architecture.")

  if allow_tf_growth:
    # For tensorflow - stop allocating the entire VRAM. For VRAM usage
    # visualization, NOT for actual train time usage. 
    print("[WARNING] Model - Tensorflow VRAM growth is ENABLED. Disable this for proper training!")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    _ = tf.compat.v1.Session(config=config)

  model_location = saved_models.joinpath(model_id)
  model_location.mkdir(exist_ok = True)

  model = _model()
  # Output a summary of the model architecture + statistics
  model.summary()

  # Use the default beta_1, beta_2, and decay.
  opt = Adam(learning_rate=learning_rate)

  # Compile the model. 
  model.compile(optimizer=opt, loss=loss_function, metrics=metrics)

  callbacks = get_callbacks(saved_models, model_id)

  return model, callbacks

def get_callbacks(saved_models:Path, model_id: str):
  """
  Get callbacks. Callable directly. 
  """
  model_location = saved_models.joinpath(model_id)
  model_location.mkdir(exist_ok = True)

  # Callbacks. 
  mcp = ModelCheckpoint(filepath=model_location.joinpath('%s_{val_mse:.5f}_{mse:.5f}_{epoch:02d}.h5' % model_id),
                        monitor=mcp_monitor, 
                        verbose=1, 
                        save_best_only=mcp_save_best_only)
  
  callbacks = [mcp]
  return callbacks

def _model():
  X_input = Input(shape=input_dim)
  X = None

  # First GRU layer.
  #X = Bidirectional(GRU(units = gru_width, return_sequences=True))(X_input)
  X = GRU(units = gru_width, return_sequences=True)(X_input)
  X = BatchNormalization()(X)
  X = Dropout(input_dropout)(X)

  # Hiden GRU layers.
  for _ in range(1, gru_depth):
    #X = Bidirectional(GRU(units = gru_width, return_sequences=True))(X)
    X = GRU(units = gru_width, return_sequences=True)(X)
    X = BatchNormalization()(X)
    X = Dropout(hidden_dropout)(X)
  
  X = Dense(fully_connected_width, activation="relu")(X)
  X = Dropout(hidden_dropout)(X)

  # The size of the output layer should equal velocity + control
  # changes + offset info.
  X = Dense(len(data_solution_cols))(X)

  model = Model(inputs = X_input, outputs = X)
  return model