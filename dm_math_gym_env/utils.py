import os
import re
from tqdm import tqdm
from dm_math_gym_env.typed_operators import Equation, Function, Expression, Variable, Value, Rational
import sympy


def is_numeric(string):
    return all([x.isnumeric() or x == "." for x in string] + [string.count(".") <= 1])


def extract_formal_elements_as_annotations(question):
    pattern = "\$f\[(.+?)\]"
    return re.findall(pattern, question)


def extract_formal_elements(question, cast=True):
    # split on punctuation unless it is immediately preceded and followed by a number (indicating it is a decimal)
    split_on_punctuation = "***".join(
        [
            string
            for string in re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)
            if len(string) > 0 and not string.isspace()
        ]
    )
    # TODO: use a more sophisticated mechanism (CFG?) to math expressions, equations, etc... this could account for variables names that have length greater than 1
    split_on_words = [
        string
        for string in re.split("[A-Za-z]\w+|\*\*\*", split_on_punctuation)
        if len(string) > 0 and not string.isspace()
    ]
    # strip trailing or leading whitespace
    formal_elements = [string.strip() for string in split_on_words]
    # filter for the special case where the letter "a" gets included at the end of a formal element
    formal_elements = [
        f if len(re.findall("[0-9A-Za-z\)](\sa)", f)) < 1 else f.split(" a")[0]
        for f in formal_elements
    ]
    # cast types
    if cast:
        formal_elements = [cast_formal_element(f) for f in formal_elements]
    return formal_elements


def cast_formal_element(f):
    try:
        x = sympy.sympify(f)
        if type(x) == sympy.core.numbers.Rational:
            return Rational(str(x))
        elif issubclass(type(x), sympy.core.numbers.Number):
            return Value(str(x))
        elif type(x) == sympy.core.symbol.Symbol:
            return Variable(f)
        else:
            return Expression(f)
    except:
        if "=" in f:
            try:
                return Function(f)
            except:
                return Equation(f)


def guess_until_problem_solved(env, question, answer, verbose=False, max_episode_index=1000):
    episode_i = 0
    graph_guessed_correctly = False
    encoded_question, _ = env.reset_from_text(question, answer)
    print(f"\nquestion: {env.decode(encoded_question)}")
    while not graph_guessed_correctly and episode_i < max_episode_index:
        encoded_question, _ = env.reset_from_text(question, answer)
        done = False
        step_i = 0
        if verbose:
            print(f"episode: {episode_i}")
        while not done:
            action_index = env.sample_masked_action_index()
            observation, reward, done, info = env.step(action_index)
            if verbose:
                if "lookup_value(solve_system(append_to_empty_list('p_0')),Variable('b'))" in info['raw_observation']:
                    print()
                print(f"\t\tS': {info['raw_observation']}, R: {reward}, done: {done}")
            if reward == 1:
                graph_guessed_correctly = True
            step_i += 1
        episode_i += 1
    assert graph_guessed_correctly
    print(f'graph: {info["raw_observation"].split(";")[1]}')
    print(f"{episode_i} trials taken to guess: {question}")


def filter_univariate(examples):
    univariate_examples = []
    for example_dict in examples:
        question = example_dict['question']
        formal_elements = extract_formal_elements(question, cast=False)
        function = formal_elements[0]
        num_vars = len([ch for ch in set(function) if ch.isalpha()])
        if num_vars == 1:
            univariate_examples.append(example_dict)
    return univariate_examples


def get_module_name_from_filepath(fp):
    module_name = fp.split("/")[-1].split(".txt")[0]
    if "compose" in module_name:
        module_name = module_name.split("_compose")[0]
    else:
        module_name = module_name
    return module_name


def load_question_answer_pairs(filepath):
    qa_pairs = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    num_pairs = len(lines) // 2
    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        qa_pairs.append((question, answer))
    return qa_pairs


# load data
def load_data(config, train=True):
    data = {}
    print("loading problems")
    if train:
        problem_filepaths = [os.path.join(config.data_dirpath, filename) for filename in config.selected_filenames]
    else:
        problem_filepaths = [os.path.join(config.test_data_dirpath, filename) for filename in config.selected_filenames]

    problem_counts = {}
    for filepath in tqdm(problem_filepaths):
        with open(filepath, "r") as f:
            lines = f.readlines()
        num_pairs = min(len(lines) // 2, config.num_problems_per_module)
        for i in range(0, 2 * num_pairs, 2):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            # for uncomposed problems set difficulty to 0 to distinguish them
            difficulty = (
                len(re.split("(?<![0-9])[.,;:?]|[.,;:?](?![0-9])", question)) - 1
                if 'compose' in filepath
                else 0
            )
            # don't load problems with difficulty above the maximum
            if difficulty > config.max_difficulty:
                continue
            module_name = get_module_name_from_filepath(filepath)
            # increment problem count for (module_name, difficulty)
            if (module_name, difficulty) in problem_counts:
                problem_counts[(module_name, difficulty)] += 1
            else:
                problem_counts[(module_name, difficulty)] = 1
            # store problem
            problem_dict = {'module_difficulty_index': problem_counts[(module_name, difficulty)],
                            'question': question,
                            'answer': answer}
            if module_name in data:
                if difficulty in data[module_name]:
                    data[module_name][difficulty].append(problem_dict)
                else:
                    data[module_name][difficulty] = [problem_dict]
            else:
                data[module_name] = {difficulty: [problem_dict]}
    if config.univariate_differentiation:
        data['calculus__differentiate'][0] = filter_univariate(data['calculus__differentiate'][0])
    return data

def split_validation_data(config, train):
    val = {}
    for module_name in train:
        val[module_name] = {}
        for difficulty in train[module_name]:
            num_examples = len(train[module_name][difficulty])
            num_val = int(num_examples * config.validation_percentage)
            val[module_name][difficulty] = train[module_name][difficulty][:num_val]
            train[module_name][difficulty] = train[module_name][difficulty][num_val:]
            assert (
                len(train[module_name][difficulty])
                + len(val[module_name][difficulty])
                == num_examples
            )
    return val