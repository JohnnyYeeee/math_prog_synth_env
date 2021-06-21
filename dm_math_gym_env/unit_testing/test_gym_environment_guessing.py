from hparams import HParams
hparams = HParams('artifacts/.', hparams_filename='hparams_guessing', name='rl_math', ask_before_deletion=False)
from dm_math_gym_env.envs.math_env import MathEnv
from dm_math_gym_env.utils import guess_until_problem_solved, load_question_answer_pairs
import unittest
import os

class Test(unittest.TestCase):

    def test_guess_until_correct(self):
        """this test only terminates when the graph is correctly guessed or timeout is reached"""
        env = MathEnv(hparams.env)
        for filename in [fn for fn in os.listdir('artifacts/problems') if '.txt' in fn]:
            filepath = os.path.join(f'artifacts/problems/{filename}')
            question_answer_pairs = load_question_answer_pairs(filepath)
            for question, answer in question_answer_pairs[:5]:
                guess_until_problem_solved(env, question, answer, verbose=False, max_episode_index=50000)

