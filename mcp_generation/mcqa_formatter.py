import json

probes = []
with open("./multilingual_probes.jsonl",'r') as f:
    for line in f:
        probes.append(json.loads(line.rstrip('\n|\r')))

results = []
for probe in probes:
    new_items = {}
    answer_labels = ["A", "B", "C", "D", "E","F","G","H"]
    print(probe["truth_id"])
    answerKey = answer_labels[probe["truth_id"]]
    new_items["id"] = probe["id"]
    new_items["lang"] = probe["langs"]
    new_items["question"] = {"stem": " "}
    new_items["question"]["choices"] = [{"label": l , "text":t} for l,t in zip(answer_labels, probe["probes"])]
    new_items["answerKey"] = answerKey
 
    results.append(new_items)


with open('/path/to/mcp_data/train.jsonl','w') as f:
    for result in results[:-1000]:
        json.dump(result,f, ensure_ascii=False)
        f.write('\n')


with open('/path/to/mcp_data/dev.jsonl','w') as f:
    for result in results[-1000:]:
        json.dump(result,f, ensure_ascii=False)
        f.write('\n')