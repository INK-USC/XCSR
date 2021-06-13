# Multilingual Contrastive Pre-Training for Better X-CSR

## Download the MickeyCorpus

Check it here: 

## Generating the MCP data

Please see the details in our paper. Change the path in `generate_mcp_data.py` and run it to generate the `mcp_data.jsonl`.

Here we show an example in the `mcp_data.jsonl`:

```json
{
  "id": "1472b1b7350f4fcb",
  "truth_id": 1,    # the id of the correct assertion.
  "langs": ["bg", "en", "zh", "ru", "hi", "fr", "vi", "de"], # the lang of each probe in the same order.
  "probes": [
    "Нима ходите по улиците, за да се страхувате от други хора?",
    "You would visit other countries because you want to experience other cultures.",   # only this is correct.
    "你会去其他街道, 因为害怕克服其他文化。",
    "Вы бы посетили другие страны, потому что хотите испытать другие таланты.",
    "आप दूसरे सड़कों पर जाते क्योंकि आप अन्य संस्कृतियों पर विजय पाने का भय रखते हैं ।",
    "Vous visiteriez d'autres personnes parce que vous voulez faire l'expérience d'autres cultures.",
    "Bạn sẽ đi thăm những con đường khác bởi vì bạn sợ để phá vỡ khác chủng tộc.",
    "Sie würden andere Personen besuchen, weil Sie andere Kulturen erleben möchten."
  ]
}
```


## Reformat the MCP data to the MCQA format.

`python mcqa_formatter.py`

## Run the Pre-Training.

We take the XLM-R-L as an example here.

```bash 
run mcp_generation/run_mcp_xlmrl.sh pretrain
run mcp_generation/run_mcp_xlmrl.sh xcsqa-finetune
run mcp_generation/run_mcp_xlmrl.sh xcsqa-infer
```
