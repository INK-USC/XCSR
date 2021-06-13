import random 
import json

random.seed(42)
lang_list = ["en", "es", "nl", "de", "fr", "zh", "it", "ru", "bg", "vi", "hi"]  # all candidate languages
all_data = {}
truth_labels = {}
probes = []
for lang in lang_list:
    with open('/path/to/mickey_'+lang+'.jsonl','r') as f:
        for line in f:
            probes.append(json.loads(line.rstrip('\n|\r')))
            
for probe in probes:
    data = {}
    probe_id = probe["id"]
    truth_labels[probe_id] = probe["truth_id"]
    data[probe["lang"]] = probe["probes"]
    if probe_id not in all_data:
        all_data[probe_id] = data
    else:
        all_data[probe_id][probe["lang"]] = probe["probes"]
  
results = []
for probe_id in all_data.keys():
    new_items = {}    
    truth_id = truth_labels[probe_id]
    random_langs = random.sample(lang_list, 8)  # 8 probes in total.
    random.shuffle(random_langs)
    correct_lang = random.choice(random_langs)
    correct_id = random_langs.index(correct_lang)
    
    multi_probes = []
    for lang in random_langs:
        if lang == correct_lang:
            probe = all_data[probe_id][lang][truth_id]
        else:
            ith = random.choice([i for i in range(0,4) if i != truth_id])
            probe = all_data[probe_id][lang][ith]
        multi_probes.append(probe)
            
    new_items["id"] = probe_id
    new_items["truth_id"] = correct_id
    new_items["langs"] = random_langs
    new_items["probes"] = multi_probes
    results.append(new_items)

with open('mcp_data.jsonl','w') as f:
    for result in results:
        json.dump(result,f, ensure_ascii=False)
        f.write('\n')
    