from hparams import HParams
hparams = HParams('artifacts/.', hparams_filename='hparams', name='rl_math', ask_before_deletion=False)
import numpy as np
from dm_math_gym_env.utils import extract_formal_elements
from dm_math_gym_env.envs.math_env import MathEnv
from dm_math_gym_env.typed_operators import *
from utils import read_text_file
import unittest


class Test(unittest.TestCase):
    def test_algebra_linear_1d_fail_1(self):
        env = MathEnv(hparams.env)
        # reset - then fail after 1st action
        encoded_question, _ = env.reset_from_text("Solve 0 = 4*b + b + 15 for b.", "-3")
        question = env.decode(encoded_question)
        f = extract_formal_elements(question)  # for use below
        assert f == ["0 = 4*b + b + 15", "b"]
        action = "f0"
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; Equation('0 = 4*b + b + 15')"
        )
        assert reward == 0
        assert done

    def test_algebra_linear_1d_fail_2(self):
        env = MathEnv(hparams.env)
        # reset - then fail after 2nd action
        encoded_question, _ = env.reset_from_text("Solve 0 = 4*b + b + 15 for b.", "-3")
        question = env.decode(encoded_question)
        assert question == "Solve 0 = 4*b + b + 15 for b."
        action = solve_system
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{question}; solve_system('p_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; solve_system(Equation('0 = 4*b + b + 15'))"
        )
        assert reward == 0
        assert done

    def test_algebra_linear_1d_fail_3(self):
        env = MathEnv(hparams.env)
        # reset - then fail after 1st action
        encoded_question, _ = env.reset_from_text("Solve 0 = 4*b + b + 15 for b.", "-3")
        question = env.decode(encoded_question)
        f = extract_formal_elements(question)  # for use below
        assert f == ["0 = 4*b + b + 15", "b"]
        action = "f10"  # indexing out of range
        action_index = env.get_action_index(action)
        observation_, reward, done, info = env.step(action_index)
        assert reward == 0
        assert done

    def test_algebra_linear_1d_success_1(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Solve 0 = 4*b + b + 15 for b.", "-3")
        question = env.decode(encoded_question)
        assert question == "Solve 0 = 4*b + b + 15 for b."
        action = lookup_value
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; lookup_value('p_0','p_1')"
        )
        assert reward == 0
        assert not done
        assert env.compute_graph.current_node == env.compute_graph.root
        # next action
        action = solve_system
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; lookup_value(solve_system('p_0'),'p_1')"
        )
        assert reward == 0
        assert not done
        # current node is still root because it takes 2 arguments and only 1 has been given
        assert env.compute_graph.current_node == env.compute_graph.root
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; lookup_value(solve_system('p_0'),Variable('b'))"
        )
        assert reward == 0
        assert not done
        # current node is now the solve_system node because the lookup_value node has its args set
        assert env.compute_graph.current_node == env.compute_graph.root.args[0]
        # next action
        action = append_to_empty_list
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; lookup_value(solve_system(append_to_empty_list('p_0')),Variable('b'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; lookup_value(solve_system(append_to_empty_list(Equation('0 = 4*b + b + 15'))),Variable('b'))"
        )
        assert reward == 1
        assert done

    def test_calculus_differentiate_success_1_with_masking(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Find the first derivative of 2*d**4 - 35*d**2 - 695 wrt d.",
                                                  "8*d**3 - 70*d")
        question = env.decode(encoded_question)
        assert question == "Find the first derivative of 2*d**4 - 35*d**2 - 695 wrt d."
        # take action
        action = differentiate_wrt
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{question}; differentiate_wrt('p_0','p_1')"
        )
        # take action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        assert (
                info["raw_observation"] == f"{question}; differentiate_wrt(Expression('2*d**4 - 35*d**2 - 695'),'p_1')"
        )
        vector = np.ones(len(env.actions))
        masked_vector = env.mask_invalid_types(vector)
        assert masked_vector[env.get_action_index("f0")] == 0 and \
               masked_vector[env.get_action_index("f1")] == 1
        # take action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done

    def test_calculus_differentiate_success_2_with_masking(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Find the first derivative of 2*d**4 - 35*d**2 - 695 wrt d.",
                                                  "8*d**3 - 70*d")
        question = env.decode(encoded_question)
        assert question == "Find the first derivative of 2*d**4 - 35*d**2 - 695 wrt d."
        # take action
        action = differentiate
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{question}; differentiate('p_0')"
        )
        # take action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done
        assert (
                info["raw_observation"] == f"{question}; differentiate(Expression('2*d**4 - 35*d**2 - 695'))"
        )

    def test_numbers_div_remainder_success(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Calculate the remainder when 93 is divided by 59.", "34")
        question = env.decode(encoded_question)
        assert question == "Calculate the remainder when 93 is divided by 59."
        assert env.compute_graph.formal_elements == [Value("93"), Value("59")]
        # first action
        action = mod
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"] == f"{question}; mod('p_0','p_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; mod(Value('93'),'p_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
            info["raw_observation"]
            == f"{question}; mod(Value('93'),Value('59'))"
        )
        assert reward == 1
        assert done

    def test_numbers_gcd_success(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Calculate the highest common divisor of 1300 and 300.", "100")
        question = env.decode(encoded_question)
        assert question == "Calculate the highest common divisor of 1300 and 300."
        # first action
        action = gcd
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{question}; gcd('p_0','p_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{question}; gcd(Value('1300'),'p_1')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{question}; gcd(Value('1300'),Value('300'))"
        )
        assert reward == 1
        assert done

    def test_is_prime_success_1(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Is 93163 a prime number?", "False")
        question = env.decode(encoded_question)
        assert question == "Is 93163 a prime number?"
        # first action
        action = is_prime
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{question}; is_prime('p_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{question}; is_prime(Value('93163'))"
        )
        assert reward == 1
        assert done

    def test_is_prime_success_2(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Is 66574 a composite number?", "True")
        question = env.decode(encoded_question)
        assert question == "Is 66574 a composite number?"
        # first action
        action = not_op
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"] == f"{question}; not_op('p_0')"
        )
        assert reward == 0
        assert not done
        # next action
        action = is_prime
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{question}; not_op(is_prime('p_0'))"
        )
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert (
                info["raw_observation"]
                == f"{question}; not_op(is_prime(Value('66574')))"
        )
        assert reward == 1
        assert done

    def test_problem_third_diff_success(self):
        env = MathEnv(hparams.env)
        # reset - then succeed after 4th action
        encoded_question, _ = env.reset_from_text("Find the third derivative of -272*j**5 + j**3 - 8234*j**2.",
                                                  "-16320*j**2 + 6")
        question = env.decode(encoded_question)
        assert question == "Find the third derivative of -272*j**5 + j**3 - 8234*j**2."
        # take action
        action_index = env.get_action_index(differentiate)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # take action
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # take action
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # take action
        action_index = env.get_action_index("f0")
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done

    def test_max_nodes_failure(self):
        env = MathEnv(hparams.env)
        encoded_question, _ = env.reset_from_text("Is 66574 a composite number?", "True")
        question = env.decode(encoded_question)
        assert question == "Is 66574 a composite number?"
        nt_action_index = env.get_action_index(not_op)
        for i in range(env.max_num_nodes-1):
            # take action
            observation, reward, done, info = env.step(nt_action_index)
            assert reward == 0
            assert not done
        # take final action
        i += 1
        observation, reward, done, info = env.step(nt_action_index)
        assert reward == 0
        assert done


    def test_lcd1(self):
        env = MathEnv(hparams.env)
        encoded_question, _ = env.reset_from_text("What is the common denominator of -64/1065 and 92/105?", "7455")
        question = env.decode(encoded_question)
        # lcd
        assert question == "What is the common denominator of -64/1065 and 92/105?"
        action_index = env.get_action_index(lcd)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # f0
        action_index = env.get_action_index("f0")
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # f1
        action_index = env.get_action_index("f1")
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done

    def test_lcd2(self):
        env = MathEnv(hparams.env)
        encoded_question, _ = env.reset_from_text("Calculate the common denominator of 1/(3/(-6)) - 402/(-60) and -71/12.", "60")
        question = env.decode(encoded_question)
        # lcd
        assert question == "Calculate the common denominator of 1/(3/(-6)) - 402/(-60) and -71/12."
        action_index = env.get_action_index(lcd)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # f0
        action_index = env.get_action_index("f0")
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # f1
        action_index = env.get_action_index("f1")
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done

    def test_polynomial_roots_1(self):
        question = "What is f in -87616*f**2 - 1776*f - 9 = 0?"
        answer = "-3/296"
        env = MathEnv(hparams.env)
        encoded_question, _ = env.reset_from_text(question, answer)
        action = lookup_value
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = solve_system
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = append_to_empty_list
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 1
        assert done

    def test_polynomial_roots_2(self):
        question = "Solve -3*h**2/2 - 24*h - 45/2 = 0 for h."
        answer = "-15, -1"
        env = MathEnv(hparams.env)
        encoded_question, _ = env.reset_from_text(question, answer)
        action = lookup_value
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = solve_system
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = "f1"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = append_to_empty_list
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        assert reward == 0
        assert not done
        # next action
        action = "f0"
        action_index = env.get_action_index(action)
        observation, reward, done, info = env.step(action_index)
        print(info['raw_observation'])
        assert reward == 1
        assert done
