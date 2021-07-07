import unittest
from math_prog_synth_env.utils import is_numeric


class Test(unittest.TestCase):
    def test_is_numeric(self):
        assert is_numeric("2")
        assert is_numeric("2.0")
        assert not is_numeric("2.0.")
