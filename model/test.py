#
# test.py
#
# Harness allowing for chain testing of model checkpoints. Produces a
# comprehensive training history graph, with curves for train, dev,
# and test. 
#
# My favorite way of determining when to stop. 
#
# Please be warned this is a very slightly adapted trigger word
# detection file and is thus mostly the code of a younger me. 

from model.dataset_utils import load_datasets, generate_song_tensors
from model.load_save import load_existing_model_path

import os
import multiprocessing
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

_minibatch_size = 10
_use_gpu = False

def test_models(model_location:Path, clean_data: Path, output_path: Path):
  """
  Given directory of model(s) to test, execute a chain test. 
  """
  print("[INFO] Test - Initializing Test Model Chain...")
  assert model_location.exists()

  chain_test_results = {}
  chain_test_results_mse_map = {}

  # Load the test dataset and preprocess it. 
  test_df = load_datasets(clean_data = clean_data, load_test=True)[0]
  test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
  X_test, Y_test = generate_song_tensors(songs_df = test_df, solutions=True)

  minibatch = []
  
  files = os.listdir(str(model_location))
  model_files = []

  filename_uid = 0 # for sorting.
  for j in range(0, len(files)):
    filename = files[j]
    if filename.endswith("h5"):
      model_files.append(filename)
    
  for j in tqdm(range(0, len(model_files)), desc = "[INFO] Test - Testing Progress", unit="models"):
    minibatch.append(model_files[j])
    # If a minibatch has been filled OR we're at the end of files. 
    if len(minibatch) == _minibatch_size or j == len(model_files)-1:
      try:
        print("[INFO] Test - Processing minibatch: " + str(minibatch))
        ret_dict = {}
        queue = multiprocessing.Queue()
        queue.put(ret_dict)

        minibatch_processes = {}

        for i in range(0, len(minibatch)):
          file = minibatch[i]
          print("[INFO] Test - Creating new subprocess for model " + str(file) + ".")
          
          # Execute test as a separate process. Use a queue to
          # obtain results. 
          p = multiprocessing.Process(target=test_model_worker, 
                                      args=(queue, str(model_location.joinpath(file)), 
                                      X_test, Y_test, "mse" + str(i)))
          minibatch_processes[i] = (p, file)

        # After all of the processes have been created. Kick off in parallel.
        for item in minibatch_processes:
          tuple = minibatch_processes[item]
          print("\n\n[INFO] Test - Executing new process for model " + tuple[1] + ".\n")
          tuple[0].start()

        # Now wait for all of them.
        for p in minibatch_processes:
          tuple[0].join()
        ret_dict_result = queue.get()
        print("\n[INFO] Test - Processes complete; results:")
        print(ret_dict_result) 
        print("")
        for item in ret_dict_result:
          item_identifier = int(item.replace("mse",""))
          if item_identifier in minibatch_processes:
            filename = minibatch_processes[int(item.replace("mse",""))][1]
            mse = ret_dict_result[item]
          
            if mse is None:
              print("[WARN] Test - Received empty mse!")
              chain_test_results[filename_uid] = "00.00000000 - " + str(filename) + " TEST FAILED!\n"
              chain_test_results_mse_map[-1] = filename_uid
            else:
              chain_test_results[filename_uid] = "%.8f - " % (mse) + str(filename) + "\n"
              # If a model of that exact mse exists already, append
              # a tiny number to it until it's unique. 
              if mse in chain_test_results_mse_map:
                sorting_mse = None
                while sorting_mse is None:
                  mse = mse + 0.000000000000001 # mse has 15 decimal precision. Append by 1 to break ties.
                  if mse not in chain_test_results_mse_map:
                    sorting_mse = mse
                chain_test_results_mse_map[sorting_mse] = filename_uid
              else:
                chain_test_results_mse_map[mse] = filename_uid
            
            filename_uid = filename_uid + 1
          
        print("\n[INFO] Test - Minibatch: " + str(minibatch) + " processing complete.\n")
      except Exception as e:
        # Use a try/except so that we still write the remaining stuff 
        # to file in case of a failure or the user cancels the rest.
        print("\n\n[ERROR] Test - !!!! Failed to process model " + str(filename) + "! Exception:")
        print(e)
        print("\n")
      minibatch = []

  if(filename_uid == 0):
    print("[WARNING] Test - No models found at location \"%s\". Please specify another location with an argument (example: python test_model_chain.py ./model_checkpoints) or move/copy the model(s) accordingly." 
      % model_location)
    return

  # Sort the results.
  chain_test_results_mse_map = dict(sorted(chain_test_results_mse_map.items(), key=lambda item: item[0]))

  # All results obtained. Write to file. 
  write_results(chain_test_results, chain_test_results_mse_map, output_path)

# Executed as a separate process so it can be purged as a
# seperate process, allowing Tensorflow to clear out the
# memory of the GPU and not freak out when we train another
# right after. If the GPU is disabled, this still allows the 
# memory to be handled properly. 
def test_model_worker(queue, model_path, X_test, Y_test, mse_dict_name = "mse"):

  # Allow tensorflow growth during testing, so as to allow for 
  # multiprocessing to happen. 
  #config = tf.compat.v1.ConfigProto()
  #config.gpu_options.allow_growth=True
  #_ = tf.compat.v1.Session(config=config)

  if _use_gpu is False:
    # Expliclty stop the GPU from being utilized. Use this option
    # in the case that your dev set is really small. 
    # In case you have a CUDA enabled GPU and don't want to use it. 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

  # Load the model. 
  model = load_existing_model_path(Path(model_path))
  assert model is not None
  
  # Evaluate the model. 
  loss, mse = model.evaluate(X_test, Y_test)
  print("[INFO] Test - Dev set MSE is: ", mse) 

  ret_dict = queue.get()
  ret_dict[mse_dict_name] = mse
  queue.put(ret_dict)
  
def write_results(chain_test_results, chain_test_results_mse_map, output_path):
  try:
    results_folder_contents = os.listdir(str(output_path))
    result_index = 0
    file_name_prefix = "chain_test_results_"
    file_name_suffix = ".txt"
    for file in results_folder_contents:
      file_number_str = file.replace(file_name_prefix, "").replace(file_name_suffix, "")
      file_number = -1
      try:
        file_number = int(file_number_str)
        if(file_number >= result_index):
          result_index = file_number + 1
      except:
        print("[WARN] Test - Unexpected file in results directory. Ignoring...")

    filename = str(output_path) + "/"+file_name_prefix+str(result_index)+file_name_suffix
    f = open(filename, "w")
    print("\n[INFO] Test - Chain test complete. Writing results to file '"+filename+"'...")

    f.write("=================================\nSORTED Chain Test Results\n=================================\n\n")
    # Write results of each model, sorted.
    for key in chain_test_results_mse_map:
      f.write(chain_test_results[chain_test_results_mse_map[key]])

    f.close()
  except:
    print("[ERROR] Test - Failed to write results to file!")
  
  graph_history(filename, chain_test_results, chain_test_results_mse_map)

  print("[INFO] Test - Write complete. Have a good night...")

# Given all the test accuracies that we've generated, let's 
# graph every single one. Note that we expect a very specific
# file name structure for these model iterations:
#
# '+str(modelnum)+'_{val_mse:.5f}_{mse:.5f}_{epoch:02d}' + ".h5"
# Ex) model1_63.42835_102.71994_80.h5
#
# Expects full location of the file that has been written -
# the suffix will be stripped but otherwise the graph will be
# written to file with that exact name. 
def graph_history(filename, chain_test_results, chain_test_results_mse_map):
  print("[INFO] Test - Generating test mse history graph.")

  graph_width_inches = 13
  graph_height_inches = 7
  
  graph_location = filename.replace(".txt", "")

  title = "Chain Test MSE History"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)

  # Gather points. Every point should be indexed by epoch. 
  indices = []
  test_mses = []
  val_mses = []
  train_mses = []

  # Each item should be structured as such:
  # 99.69913960 - model1_63.42835_102.71994_80.h5
  for key in chain_test_results_mse_map: 
    string = chain_test_results[chain_test_results_mse_map[key]]
    try:
      epoch = None
      test_mse = None
      val_mse = None
      train_mse = None

      string_split_apos = string.split(" - ")
      test_mse = float(string_split_apos[0].strip())

      result_string_split = string_split_apos[1].split("_")
      epoch = int(result_string_split[3].split(".h")[0].strip())
      val_mse = float(result_string_split[2].strip())
      train_mse = float(result_string_split[1].strip())
      
      indices.append(epoch)
      test_mses.append(test_mse)
      val_mses.append(val_mse)
      train_mses.append(train_mse)
    except Exception as e:
      print("[WARNING] Test - Encountered an exception while parsing string " + str(string) + ":")
      print(e)

  # We now have everything in our arrays. Combine them to graph 
  # them.   
  data= []
  for i in range(0, len(indices)):
    data.append([indices[i], test_mses[i], val_mses[i], train_mses[i]])

  df = pd.DataFrame(data, columns = ["epoch", "test_mse", "val_mse", "train_mse"])

  # Sort the dataframe by epoch first so lines are drawn properly.
  df.sort_values("epoch", axis=0, inplace=True)

  df.set_index("epoch", drop=True, inplace=True)
  df = df.astype(float)

  # With our dataframe, we can finally graph the history. 
  plt.plot(df["val_mse"])
  plt.plot(df["test_mse"])
  plt.plot(df["train_mse"])
  plt.ylabel("MSE")
  plt.xlabel("Epoch")
  plt.legend(["val_mse", "test_mse", "train_mse"], loc="upper left")

  # Save the graph. 
  try:
    fig.savefig(graph_location)
    print("[DEBUG] Test - Graph successfully saved to: " + str(graph_location) + ".")
  except Exception as e:
    print("[ERROR] Test - Unable to save graph at location: " + str(graph_location) + ". Exception:")
    print(e)
  
  plt.close("all")