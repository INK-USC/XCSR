import json
import collections
from collections import defaultdict
from tqdm import tqdm
import pickle
import numpy as np               
import os


def eval_npy(truth_file, prediction_file):
  """Main fucntion.""" 
  instances = []
  with open(truth_file) as f:
    for line in f.read().splitlines():
      if line:
        instances.append(json.loads(line))
  with open(prediction_file, "rb") as f:
    predictions = pickle.load(f)
  assert len(predictions) == len(instances)

  num_correct = 0
  for ind, (inst, pred) in enumerate(zip(instances, predictions)):
    choices = inst["question"]["choices"]
    correct_choice = choices[ord(inst["answerKey"])-ord("A")]
    assert correct_choice["label"] == inst["answerKey"]
    assert len(choices) == len(pred)
    ranked_list = [(c["text"], float(p)) for c, p in zip(choices, pred)]
    ranked_list.sort(key=lambda x: x[1], reverse=True)

    if ranked_list[0][0] == correct_choice["text"]:
        num_correct += 1
  return num_correct/len(instances)

def output_labels(data_file, prediction_file):
  instances = []
  with open(data_file) as f:
    for line in f.read().splitlines():
      if line:
        instances.append(json.loads(line))
  with open(prediction_file, "rb") as f:
    predictions = pickle.load(f)
  assert len(predictions) == len(instances)

  num_correct = 0
  results = {}
  for ind, (inst, pred) in enumerate(zip(instances, predictions)):
    choices = inst["question"]["choices"]
    assert len(choices) == len(pred)
    ranked_list = [(c["label"], float(p)) for c, p in zip(choices, pred)]
    ranked_list.sort(key=lambda x: x[1], reverse=True)
    results[inst["id"]] = ranked_list[0][0]
  return results

