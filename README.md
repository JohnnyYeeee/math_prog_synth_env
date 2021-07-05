# Setup:

``` bash
git clone https://github.com/JohnnyYeeee/dm_math_gym_env.git
# optionally create and activate a new environment
conda create -n dm_math_gym_env -y python=3.7
conda activate dm_math_gym_env
# install dependencies
pip install -e dm_math_gym_env
```

Before running the environment several pre-requisites need to be completed:
- The raw data (https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz) needs to be downloaded
- The data needs to be split into train/val/test sets
- A tokenizer needs to be created

Upon running `gym.make('dm_math_gym_env:math-env-v0', config_file=params.yaml)` a check is performed to determine if the last step (tokenizer creation) has been completed. If not then all 3 steps will be automatically completed. 

```python
import gym
# First time running the below command will take up to an hour to download all data
env = gym.make('dm_math_gym_env:math-env-v0', config_file='params.yaml')
```

All other code we used for training algorithms can be found at: https://github.com/joepalermo/dm_math_solvers