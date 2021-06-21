import unittest
from utils import read_json, write_json
from dm_math_gym_env.utils import extract_formal_elements


def test_string_equality(question, formal_elements):
    '''this is a weaker test because types are ignored'''
    extracted_formal_elements = extract_formal_elements(question)
    for efe, fe in zip(extracted_formal_elements, formal_elements):
        print(efe, fe)
        assert str(efe) == fe


def test_type_equality(question, formal_elements):
    '''this is a stronger test because it requires that the formal elements get casted correctly as well'''
    extracted_formal_elements = extract_formal_elements(question)
    for efe, fe in zip(extracted_formal_elements, formal_elements):
        assert efe == fe, (efe, fe)


class Test(unittest.TestCase):

    def test_examples(self):

        # # do weak test
        # question_to_formal_elements = read_json(
        #     "environment/unit_testing/artifacts/extract_formal_elements_examples.json"
        # )
        # for question, formal_elements in question_to_formal_elements.items():
        #     test_string_equality(question, formal_elements)

        # do strong test
        from dm_math_gym_env.unit_testing.artifacts.extract_formal_elements_examples import typed_examples
        for question, formal_elements in typed_examples.items():
            test_type_equality(question, formal_elements)




