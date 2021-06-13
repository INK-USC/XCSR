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



def main(dataset="CSQA"):
  # all_langs = ["en", "zh", "de", "es", "fr", "it", "jap", "nl", "pt", "ru", "pl", "ar", "vi", "hi", "sw", "ur"]
  all_langs = ["en", "de", "it", "es", "fr", "nl", "ru", "vi", "zh", "hi", "pl", "ar", "jap", "pt", "sw", "ur"]
  print ("model_name, mode, " + ",".join(all_langs))
  # dataset="CSQA"
  # for model_name in ["mbert", "xlm", "xlmrb", "xlmrl", "mbert+mcp", "xlmrb+mcp", "xlmrl+mcp"]:
  for model_name in ["xlmrb+mcp", "xlmrl+mcp"]:
    for split in ["test"]:
        for mode in ["zero-shot"]:  # 
          accs = []
          for lang in all_langs:
              truth_file = "corpus/"+dataset+"/X-"+dataset+"/%s/%s.jsonl"%(lang, split)
              if mode=="self-train":
                  prediction_npy_file = "corpus/"+dataset+"/X-"+dataset+"/%s/results/self_%s_%s_result.npy"%(lang, split, model_name)
              else: 
                  if model_name.endswith("+mcp"):
                    t = model_name.replace("+mcp", "")
                    prediction_npy_file = "corpus/"+dataset+"/X-"+dataset+"/%s/results/mcp_en-%s_%s_%s_result.npy"%(lang, lang, split, t)
                    # prediction_npy_file = ""
                  else:
                    prediction_npy_file = "corpus/"+dataset+"/X-"+dataset+"/%s/results/en-%s_%s_%s_result.npy"%(lang, lang, split, model_name)
              # print(prediction_npy_file)
              if os.path.exists(prediction_npy_file):
                acc = eval_npy(truth_file, prediction_npy_file)
                accs.append("%.2f"%(acc*100))
              else:
                # print(prediction_npy_file)
                pass
            
          print("%s,%s,%s"%(model_name, mode, ",".join(accs)))


if __name__ == "__main__": 
  main(dataset="CSQA")
  main(dataset="CODAH")
  
  # prediction_npy_file = "/path/to/X-CSQA/en/results/test_robertalarge_result.npy"
  # truth_file = "/path/to/X-CSQA/en/test.jsonl"
  # acc = eval_npy(truth_file, prediction_npy_file)
  # print(acc)

  # prediction_npy_file = "/path/to/X-CODAH/en/results/test_robertalarge_result.npy"
  # truth_file = "/path/to/X-CODAH/en/test.jsonl"
  # acc = eval_npy(truth_file, prediction_npy_file)
  # print(acc)
