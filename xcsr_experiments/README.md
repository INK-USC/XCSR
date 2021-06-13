## Run XCSR Experiments 



### Installation

```bash
conda create -n xcsr python=3.7
conda activate xcsr
pip install transformers==3.4.0
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# adjust according to your env.
```

### Run Experiments 

Note that all scripts are under `/scripts` folder. Here we take the example of training and predicting of `xlmrb` on  `csqa`.
Please remember change the paths in the scripts for your own paths and also the paths to save the models and results.

```bash
bash scripts/run_xcsqa_xlmrb.sh en_train
bash scripts/run_xcsqa_xlmrb.sh all_infer
```

### Evaluation

One can generate the output labels and evaluate the performance on dev set using the functions in `eval_utils.py`. 
We will add more information for guiding you to submit your results to our leaderboard.
