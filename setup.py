from setuptools import setup

setup(name='dm_math_gym_env',
      version='0.0.1',
      install_requires=['gym', 'sympy', 'numpy', 'scipy', 'sentencepiece', 'torch',  'mathematics_dataset', 'tqdm',
                        'sklearn', "google-cloud-storage", "pyyaml", "multiprocess"]
)