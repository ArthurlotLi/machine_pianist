#
# train.py
#
# Training harness given a preprocessed dataset. 

from model.hparams import *
from model.model import machine_pianist, get_callbacks
from model.dataset_utils import load_datasets, generate_song_tensors
from model.load_save import load_existing_model
from data_processing.data_params import *

from pathlib import Path
import matplotlib.pyplot as plt

def train_machine_pianist(clean_data: Path, model_id: str,
                          saved_models: Path):
  """
  Principal function that first defines the model before executing
  training according to the variables set in hparams. Expects the
  datasets path, the model identifier, and the saved_models location.
  """
  print("[INFO] Train - Beginning Machine Pianist training session for \"%s\"." % model_id)
  # Load the dataset. 
  train_df = load_datasets(clean_data, load_train= True)[0]
  X, Y = _extract_X_Y(train_df)

  # Attempt to load an existing model if it exists. Expects the 
  # warm-start model to have the name specified in hparams.
  model = load_existing_model(saved_models, model_id, saved_model_name)
  if model is not None:
    model.summary()
    print("\n[INFO] Train - Loaded existing model %s (summarized above). Execute warm start?\n\nPress [ENTER] to confirm." % saved_model_name)
    input()
    callbacks = get_callbacks(saved_models, model_id)
  else:
    # Define the model + callbacks according to hparams.
    model, callbacks = machine_pianist(saved_models, model_id)

  # Train.
  train_history = _execute_train(model, callbacks, X, Y)
  # Now save train history info. 
  _graph_model_history(saved_models, model_id, train_history)

def _extract_X_Y(train_df):
  """
  Given the training dataframe, extract the tensors that we'll need
  for train and test. These tensors will be of the following sizes:

  input: (maximum_song_length, 4)
  output: (maximum_song_length, 1) # TODO: Process more than just velocities
  """
  # Drop unnamed columns. 
  train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]

  # Convert these dataframes into matrices, separated by song. Then
  # combine the matrices to create tensors. 
  X, Y = generate_song_tensors(songs_df = train_df, solutions=True)

  return X, Y

def _execute_train(model, callbacks, X, Y):
  """
  Execute training session for the specified amount of epochs, having
  been provided the model and callbacks as well as the datasets.
  """
  print("[INFO] Train - Starting training session.")
  train_history = model.fit(X, Y, shuffle=True,
                            epochs = epochs,
                            callbacks=callbacks,
                            validation_split=validation_split,
                            verbose=training_verbose,
                            batch_size=batch_size)
  return train_history

def _graph_model_history(saved_models, model_num, history):
  """
  Graphs and saves training history, with two graphs - one for mse,
  the other for loss. 
  """
  print("[INFO] Generating model history graph for model " + str(model_num) + ".")

  # Constants for both graphs.
  graph_width_inches = 13
  graph_height_inches = 7

  # Generate mse graph
  title = "Machine Pianist \"" + str(model_num) + "\" Training History [mse]"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)
  plt.plot(history.history['mse'])
  plt.plot(history.history['val_mse'])
  plt.ylabel('mse')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc="upper left")

  # Save the graph. 
  location = str(saved_models) + "/" + str(model_num) + "/" + str(model_num) + "_mse"
  try:
    fig.savefig(location)
    print("[DEBUG] Graph successfully saved to: " + str(location) + ".")
  except Exception as e:
    print("[ERROR] Unable to save graph at location: " + str(location) + ". Exception:")
    print(e)
  
  plt.close("all")

  # Generate loss graph
  title = "Machine Pianist \"" + str(model_num) + "\" Training History [Loss]"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc="upper left")

  # Save the graph. 
  location = str(saved_models) + "/" + str(model_num) + "/" + str(model_num) + "_loss"
  try:
    fig.savefig(location)
    print("[DEBUG] Graph successfully saved to: " + str(location) + ".")
  except Exception as e:
    print("[ERROR] Unable to save graph at location: " + str(location) + ". Exception:")
    print(e)
  
  plt.close("all")