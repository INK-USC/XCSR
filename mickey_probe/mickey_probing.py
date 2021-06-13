import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import json
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
from tqdm import tqdm
from tqdm import trange
import argparse
import random
import numpy as np



parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--model_str', default="xlm-roberta-large", type=str)
parser.add_argument('--input_file', default="", type=str) 
parser.add_argument('--result_file', default="", type=str) 
parser.add_argument('--num_shards', type=int, default=0, help='number of shards')
parser.add_argument('--shard_id', type=int, default=-1, help='the current shards')
parser.add_argument('--use_batch', type=int, default=1, help='1 for true; 0 for false')
parser.add_argument('--ignore_badcase', type=int, default=1, help='1 for true; 0 for false')
parser.add_argument('--cut_off', default=None, type=int)
args = parser.parse_args()
print(args) 


ctxs = [mx.gpu()]
 
model, vocab, tokenizer = get_pretrained(ctxs, args.model_str) 

scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
# print(scorer.score_sentences(["Hello world"]))
# print(scorer.score_sentences(["你好"]))
# print(scorer.score_sentences(["Hallo Welt"])) 



with open(args.input_file) as f:
    lines = f.read().splitlines()

assert args.num_shards >= args.shard_id
if args.num_shards > 0 and args.shard_id >= 0:
    lines = list(np.array_split(lines, args.num_shards)[args.shard_id]) 

num_badcases = 0
num_error = 0
num_correct = 0
hit_at_k = [0,0,0]
current_amount = 0
lines = lines[:args.cut_off]
examples = [json.loads(l) for l in lines]


lang = args.input_file.replace(".jsonl", "").split("/")[-1].split("_")[-1] + ":"
desc="Inference Steps " + lang + " (%d/%d)"%(args.shard_id, args.num_shards)
probar = trange(len(examples))
results = []
for example_data in examples:
    # debug:
    truth_probe_len = len(example_data["probes"][example_data["truth_id"]])
    # skip those with too long or too short distractors
    def is_bad_case(p, truth_probe_len):
        if args.ignore_badcase > 0:
            return len(p) > truth_probe_len*1.5 or len(p) < truth_probe_len*0.5
        else:
            return False
    if any([is_bad_case(p, truth_probe_len) for p in example_data["probes"]]):
        num_badcases += 1
        continue
    try:
        if args.use_batch > 0:
            scores = scorer.score_sentences([p for p in example_data["probes"]]) 
        else: 
            scores = [scorer.score_sentences([p])[0] for p in example_data["probes"]] 
    except Exception as e:
        print(e)
        print(example_data["id"]) 
        num_error += 1
        continue

    if np.argmax(scores) == example_data["truth_id"]:
        num_correct += 1
    rank = sorted(scores, reverse=True).index(scores[example_data["truth_id"]])
    example_data["prediction"] = {"scores":scores, "ranks": rank}
    if rank+1 <= len(hit_at_k):
        hit_at_k[rank] += 1
    current_amount += 1
    probar.update(1)
    hit_accs = [hacc/current_amount for hacc in hit_at_k]
    hit_accs = [hit_accs[0]] + [sum(hit_accs[0:i]) for i in range(2,len(hit_accs)+1)]
    report = "; ".join(["%.3f"%(i) for i in hit_accs])
    # probar.set_description( desc + 'top1_acc=%g'%(num_correct/current_amount)) 
    probar.set_description( desc + ' ' + report + " | num_badcases: %d"%num_badcases + " | num_error: %d"%num_error) 
    results.append(example_data)

hit_accs = [hacc/len(examples) for hacc in hit_at_k]
hit_accs = [hit_accs[0]] + [sum(hit_accs[0:i]) for i in range(2,len(hit_accs)+1)]
print(args.input_file)
print("%s\t"%args.model_str + ",".join(["%.4f"%(i) for i in hit_accs]))
with open(args.result_file, "w") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")