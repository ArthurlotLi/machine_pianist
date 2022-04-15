#
# model.py
#
# Definition of the model itself. A deep recurrent neural network 
# designed for equal-length sequence-to-sequence inference. Using
# TensorFlow, simply because it's been a while since I used it. 

from model.hparams import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, TimeDistributed
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

  # Callbacks. 
  mcp = ModelCheckpoint(filepath=model_location.joinpath('model_%s_{val_mse:.5f}_{mse:.5f}_{epoch:02d}.h5' % model_id),
                        monitor=mcp_monitor, 
                        verbose=1, 
                        save_best_only=mcp_save_best_only)
  
  callbacks = [mcp]
  return model, callbacks

def _model():
  """
  Returns the model generated according to hparams.
  """
  X_input = Input(shape=input_dim)
  X = None

  # First GRU layer. 
  X = GRU(units = gru_width, return_sequences=True)(X_input)
  X = BatchNormalization()(X)
  X = Dropout(input_dropout)(X)

  # Hiden GRU layers.
  for _ in range(1, gru_depth):
    X = GRU(units = gru_width, return_sequences=True)(X)
    X = BatchNormalization()(X)
    X = Dropout(hidden_dropout)(X)
  
  # Output layer - add a ReLU nonlinearity beforehand.
  X = Activation('relu')(X) 
  X = TimeDistributed(Dense(1, activation="linear"))(X)

  model = Model(inputs = X_input, outputs = X)
  return model