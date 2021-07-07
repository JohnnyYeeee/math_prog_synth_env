import os
from math_prog_synth_env.utils import load_training_data

n = 20
filename_to_top_lines = dict()
for filename in os.listdir('mathematics_dataset-v1.0/train-easy'):
    filepath = os.path.join('mathematics_dataset-v1.0/train-easy', filename)
    with open(filepath) as f:
        lines = f.read().split('\n')
    top_n_lines = lines[:20]
    filename_to_top_lines[filename] = top_n_lines

for filename, top_n_lines in filename_to_top_lines.items():
    with open(f'environment/unit_testing/artifacts/problems/{filename}', 'w') as f:
        string = "\n".join(top_n_lines)
        f.write(string)
