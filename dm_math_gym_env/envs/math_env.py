
from inspect import signature
import os
from pathlib import Path
from sympy import sympify
from random import sample
import gym
import numpy as np
from gym import spaces
from scipy.special import softmax
import sentencepiece as spm
from dm_math_gym_env.compute_graph import ComputeGraph
from dm_math_gym_env.typed_operators import *
from dm_math_gym_env.utils import load_data, split_validation_data
import torch

class MathEnv(gym.Env):
    def __init__(self, config_file):
        import yaml

        self.compute_graph = None
        self.episode_actions = None
        # load config
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        self.config = config
        self.encode_question = config["encode_question"]
        self.max_num_nodes = self._max_episode_steps = config["max_num_nodes"]
        self.max_formal_elements = config["max_formal_elements"]
        self.max_difficulty = config["max_difficulty"]
        self.question_vocab_size = config["question_vocab_size"]
        self.max_sequence_length = config["max_sequence_length"]
        # define available operator functions
        self.operators = [
            lookup_value,
            solve_system,
            append,
            append_to_empty_list,
            make_equation,
            lookup_value_equation,
            extract_isolated_variable,
            substitution_left_to_right,
            factor,
            differentiate,
            differentiate_wrt,
            simplify,
            make_function,
            replace_arg,
            mod,
            gcd,
            divides,
            is_prime,
            lcm,
            lcd,
            prime_factors,
            evaluate_function,
            not_op
        ]
        # ensure that every operator listed in config["operators"] is present in the above list
        valid_op_names = [op.__name__ for op in self.operators]
        assert all([op in valid_op_names for op in config["operators"]])
        # define action and observation space
        self.operators = [operator for operator in self.operators if (operator.__name__ in config["operators"])]
        self.operator_output_types = [
            signature(operator).return_annotation for operator in self.operators
        ]
        self.actions = self.operators + [
            f"f{i}" for i in range(self.max_formal_elements)
        ]
        self.action_names = [op.__name__ for op in self.operators] + [f"f{i}" for i in range(self.max_formal_elements)]
        self.num_actions = len(self.actions)
        # increment by 2 to account for both the question padding and the answer padding
        self.total_vocab_size = self.question_vocab_size + self.num_actions + 2
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_indices = np.arange(len(self.actions))
        self.observation_space = spaces.MultiDiscrete(
            [self.total_vocab_size for _ in range(config["max_sequence_length"])]
        )

        # Set up if data not downloaded yet
        if not os.path.isfile(self.config["tokenizer_filepath"] + ".model"):
            print("No data/tokenizer found: Redownloading data")
            self.setup()
        # load data
        self.train = load_data(config, train=True)
        self.val = split_validation_data(config, self.train)
        self.test = load_data(config, train=False)
        # load tokenizer
        self.question_padding_token = config["question_vocab_size"]
        # increment config["question_vocab_size"] by 1 to account for padding token
        self.action_padding_token = (config["question_vocab_size"] + 1) + self.num_actions
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.config["tokenizer_filepath"] +  ".model")


    def step(self, action_index):
        """
        :param action_index: index into the action space
        :return: observation, reward, done, info
        An action fills the next element in the compute graph.
        -observation: question + interim compute graph
        -reward: 0 if the compute doesn't evaluate correctly, 1 if it does
        -done: True if the graph is complete, False if it isn't
        -info: None
        """
        action = self.actions[action_index]
        self.compute_graph.n_nodes += 1
        self.compute_graph.add(action)
        self.episode_actions.append(action_index)
        output = self.compute_graph.eval()
        compute_graph = str(self.compute_graph)
        full_raw_observation = f"{self.question}; {compute_graph}"
        if self.encode_question:
            encoded_question = self.encode(self.question)
            # increment by (self.question_vocab_size + 1) to ensure no overlap between question vocab and action vocab
            episode_actions_array = np.array(self.episode_actions) + (self.question_vocab_size + 1)
            episode_actions_padding_array = np.array([self.action_padding_token
                                            for _ in range(self.max_num_nodes - len(self.episode_actions))])
            observation = np.concatenate([encoded_question, episode_actions_array, episode_actions_padding_array])
        else:
            observation = full_raw_observation
        next_mask = self.compute_mask()
        done = (
            self.compute_graph.current_node is None
            or self.compute_graph.n_nodes >= self.max_num_nodes
            or np.array_equal(next_mask, np.zeros(len(next_mask)))
        )
        # get reward
        if done:
            # cleanup output
            sympify_output = None
            sympify_answer = None
            try:
                sympify_output = sympify(str(output))
                sympify_answer = sympify(self.answer)
            except:
                pass
            if sympify_output is not None and sympify_answer is not None and \
                    sympify_output == sympify_answer:
                reward = 1
            elif str(output) == str(self.answer):
                reward = 1
            else:
                reward = 0
        else:
            reward = 0
        info = {"raw_observation": full_raw_observation}
        return observation, reward, done, info


    # tokenization utilities -------------------------------------------------------------------------------------------

    def encode(self, raw_observation):
        encoded_ids = self.tokenizer.encode(raw_observation)
        # pad the encoded ids up to a maximum length
        encoded_ids.extend(
            [self.question_padding_token for _ in range(self.config["max_sequence_length"] - len(encoded_ids))]
        )
        return np.array(encoded_ids)

    def decode(self, encoded_ids):
        # filter out padding tokens before decoding
        encoded_ids = [id_ for id_ in encoded_ids.tolist() if id_ != self.question_padding_token]
        return self.tokenizer.decode(encoded_ids)

    # utilities to reset the environment -------------------------------------------------------------------------------

    def reset(self, mode='train'):
        # randomly sample a module and difficulty level
        module_name = sample(list(self.train.keys()), 1)[0]
        difficulty = sample(list(self.train[module_name].keys()), 1)[0]
        return self.reset_by_module_and_difficulty(module_name, difficulty, mode=mode)

    def reset_from_text(self, question, answer):
        self.module_name = 'N/A'
        self.difficulty = 'N/A'
        self.question = question
        self.answer = answer
        self.module_difficulty_index = 'N/A'
        self.compute_graph = ComputeGraph(self.question)
        self.episode_actions = list()
        obs = np.concatenate([self.encode(self.question),
                              np.array([self.action_padding_token for _ in range(self.max_num_nodes)])])
        return obs, {'raw_observation': self.question}

    def reset_with_same_problem(self):
        self.compute_graph = ComputeGraph(self.question)
        self.episode_actions = list()
        obs = np.concatenate([self.encode(self.question),
                              np.array([self.action_padding_token for _ in range(self.max_num_nodes)])])
        return obs, {'raw_observation': self.question}

    def reset_with_specific_problem(
        self, module_name, difficulty, module_difficulty_index, train=True
    ):
        self.module_name = module_name
        self.difficulty = difficulty
        if train:

            problem_dict = self.train[module_name][difficulty][module_difficulty_index]
        else:
            problem_dict = self.val[module_name][difficulty][module_difficulty_index]
        self.question = problem_dict['question']
        self.answer = problem_dict['answer']
        self.module_difficulty_index = problem_dict['module_difficulty_index']
        self.compute_graph = ComputeGraph(self.question)
        self.episode_actions = list()
        obs = np.concatenate([self.encode(self.question),
                              np.array([self.action_padding_token for _ in range(self.max_num_nodes)])])
        return obs, {'raw_observation': self.question}

    def reset_by_module_and_difficulty(self, module_name, difficulty, mode='train'):
        self.module_name = module_name
        self.difficulty = difficulty
        if mode == 'train':
            problem_dict = sample(
                self.train[module_name][difficulty], 1
            )[0]
        elif mode == 'val':
            problem_dict = sample(
                self.val[module_name][difficulty], 1
            )[0]
        else:
            problem_dict = sample(
                self.test[module_name][difficulty], 1
            )[0]

        self.question = problem_dict['question']
        self.answer = problem_dict['answer']
        self.module_difficulty_index = problem_dict['module_difficulty_index']
        self.compute_graph = ComputeGraph(self.question)
        self.episode_actions = list()
        obs = np.concatenate([self.encode(self.question),
                              np.array([self.action_padding_token for _ in range(self.max_num_nodes)])])
        return obs, {'raw_observation': self.question}

    # utilities to sample actions --------------------------------------------------------------------------------------

    def get_action_index(self, action):
        return self.actions.index(action)

    def sample_action_index(self):
        return self.action_space.sample()

    def sample_masked_action_index(self):
        choices = np.arange(len(self.actions))
        mask = self.compute_mask()
        valid_choices = np.array([x for x, m in zip(choices, mask) if m != 0])
        return np.random.choice(valid_choices)

    def sample_masked_policy_vector(self):
        policy_vector = np.random.uniform(size=len(self.actions))
        masked_policy_vector = self.mask_invalid_types(policy_vector)
        masked_normed_policy_vector = masked_policy_vector / np.sum(
            masked_policy_vector
        )
        return masked_normed_policy_vector

    def sample_masked_action_from_model(self, model, obs):
        policy_vector = softmax(model(obs).detach().numpy()[0])
        masked_policy_vector = self.mask_invalid_types(policy_vector)
        masked_normed_policy_vector = masked_policy_vector / np.sum(
            masked_policy_vector
        )
        choices = np.arange(len(self.actions))
        action_index = np.random.choice(choices, p=masked_normed_policy_vector)
        return action_index

    def compute_mask(self):
        if not self.compute_graph.current_node:
            # first action must be an operator
            mask = np.concatenate(
                [np.ones(len(self.operators)), np.zeros(self.max_formal_elements)]
            )
        else:
            current_arg_index = len(self.compute_graph.current_node.args)
            next_type = self.compute_graph.current_node.types[current_arg_index]
            available_types = (
                self.operator_output_types + self.compute_graph.formal_element_types
            )
            mask = np.array(
                [1 if issubclass(type_, next_type) else 0 for type_ in available_types]
            )
            mask = np.concatenate(
                [
                    mask,
                    np.zeros(
                        self.max_formal_elements
                        - len(self.compute_graph.formal_elements)
                    ),
                ]
            )
        return mask

    def mask_invalid_types(self, model_output):
        mask = self.compute_mask()
        if torch.is_tensor(model_output):
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
        masked_output = mask * model_output
        return masked_output

    def render(self):
        pass

    def close(self):
        pass

    def setup(self):
        """To be ran on first use of the environment.
        Downloads data, splits data and trains tokenizer."""
        print("Downloading Data:")
        self._get_data()
        print("Splitting Data:")
        self._split_data()
        print("Training Tokenizer:")
        self._train_tokenizer()


    def _get_data(self):
        import tarfile
        import requests

        url = 'https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz'
        myfile = requests.get(url)
        open("mathematics_dataset-v1.0.tar.gz", 'wb').write(myfile.content)

        print("Data Downloaded")

        data_tar = tarfile.open(name=self.config["data_download_location"], mode='r:gz')
        data_tar.extractall(path=self.config["data_unpack_dir"])

        print("Data unpacked")


    def _split_data(self):
        import os
        from tqdm import tqdm

        problem_filepaths = [os.path.join(os.path.join(self.config["data_unpack_dir"],self.config["all_data_dirpath"]), filename) for filename in
                             self.config["selected_filenames"]]
        train_problem_filepaths = [os.path.join(self.config["data_dirpath"], filename) for filename in
                                   self.config["selected_filenames"]]
        test_problem_filepaths = [os.path.join(self.config["test_data_dirpath"], filename) for filename in
                                  self.config["selected_filenames"]]

        if os.path.isdir(self.config["data_dirpath"]) or os.path.isdir(self.config["test_data_dirpath"]):
            raise ValueError(f"data directories already exist")
        else:
            os.mkdir(self.config["data_dirpath"])
            os.mkdir(self.config["test_data_dirpath"])

        for filepath, train_filepath, test_filepath in tqdm(
                zip(problem_filepaths, train_problem_filepaths, test_problem_filepaths)):
            # read data
            with open(filepath, "r") as f:
                lines = f.readlines()
            num_pairs = len(lines) // 2
            num_train_pairs = int((1 - self.config["test_percentage"]) * num_pairs)

            # Write data
            with open(train_filepath, "w") as f:
                f.writelines(lines[:2 * num_train_pairs])
            with open(test_filepath, "w") as f:
                f.writelines(lines[2 * num_train_pairs:])
        print("train and test datasets have been created")

    def _get_corpus_for_tokenizer(self):
        from random import shuffle
        from sklearn.model_selection import train_test_split

        filepaths = [
            f"mathematics_dataset-v1.0/train-easy/{filename}" for filename in self.config["selected_filenames"]
        ]
        questions = []

        for filepath in filepaths:
            with open(filepath, "r") as f:
                lines = f.readlines()
            num_pairs = min(len(lines) // 2, self.config["num_problems_per_module_corpus"])
            for i in range(0, 2 * num_pairs, 2):
                question = lines[i].strip()
                answer = lines[i + 1].strip()
                questions.append(question)

        shuffle(questions)
        train_questions, val_questions = train_test_split(questions, test_size=0.4)
        with open(self.config["corpus_path"], "w") as f:
            f.write("\n".join(train_questions))
        print("Downloaded corpus for training tokenizer")

    def _train_tokenizer(self):
        import sentencepiece as spm
        #Get corpus
        self._get_corpus_for_tokenizer()
        # train tokenizer on question corpus
        hardcoded_symbols = ['G']  # why is 'G' needed?
        spm.SentencePieceTrainer.train(input=self.config["corpus_path"],
                                       model_prefix=self.config["tokenizer_filepath"],
                                       vocab_size=250,
                                       user_defined_symbols=hardcoded_symbols)
        print("Tokenizer saved")
