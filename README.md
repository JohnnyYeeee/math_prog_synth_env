# Installation:

To use this env in your own repo:

``` bash
git clone https://github.com/JohnnyYeeee/dm_math_gym_env.git
pip install -e dm_math_gym_env
cp dm_math_gym_env/params.yaml <path_to_your_repo>
cd <path_to_your_repo>
# To download all necessary assets (data, tokenizer)  in python run:
import gym
#First time running the below command will take up to an hour to download all data
env = gym.make('dm_math_gym_env:math-env-v0', config_file=<path_to_your_repo>/params.yaml)
```

All other code we used for training algorithms can be found at: https://github.com/joepalermo/dm_math_solvers