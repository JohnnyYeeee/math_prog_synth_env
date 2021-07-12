# math_prog_synth_env

This repository contains an implementation of math_prog_synth_env as described in <TODO link paper here>. 

![Graph construction video](https://github.com/JohnnyYeeee/math_prog_synth_env/blob/main/READ_ME_assets/graph_construction.gif?raw=true)

The full code used to produce the results reported in the paper can be found here: https://github.com/joepalermo/dm_math_solvers

## Setup:

``` bash
git clone https://github.com/JohnnyYeeee/math_prog_synth_env.git
# optionally create and activate a new environment
conda create -n math_prog_synth_env -y python=3.7
conda activate math_prog_synth_env
# install dependencies
pip install -e math_prog_synth_env
```

```python
import gym
# the first time running this may take awhile (particularly to download the data) 
env = gym.make('math_prog_synth_env:math-env-v0', config_file='params.yaml')
```

Before running the environment several pre-requisites need to be completed:

- The raw data (https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz) needs to be downloaded
- The data needs to be split into train/val/test sets
- A tokenizer needs to be created

Upon running `gym.make('math_prog_synth_env:math-env-v0', config_file='params.yaml')` a check is performed to determine if the last step (tokenizer creation) has been completed. If not then all 3 steps will be automatically completed. 

## Run unit tests

To run the unit tests, change working directory to the root of the project and then run `python -m unittest discover math_prog_synth_env/unit_testing`.
