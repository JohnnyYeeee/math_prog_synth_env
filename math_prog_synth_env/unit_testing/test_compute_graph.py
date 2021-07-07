import unittest
from math_prog_synth_env.utils import extract_formal_elements
from math_prog_synth_env.typed_operators import *
from math_prog_synth_env.compute_graph import ComputeGraph, Node


class Test(unittest.TestCase):
    def test_easy_algebra__linear_1d(self):
        question = "Solve 0 = 4*b + b + 15 for b."
        f = extract_formal_elements(question)
        cg = ComputeGraph(question)
        lookup_value_node = Node(lookup_value)
        solve_system_node = Node(solve_system)
        append_to_empty_list_node = Node(append_to_empty_list)
        append_to_empty_list_node.set_arg(Node('f0'))
        solve_system_node.set_arg(append_to_empty_list_node)
        lookup_value_node.set_args([solve_system_node, Node('f1')])
        cg.root = lookup_value_node
        assert str(cg) == "lookup_value(solve_system(append_to_empty_list(Equation('0 = 4*b + b + 15'))),Variable('b'))"
        assert cg.eval() == Value(-3)

    def test_incomplete_compute_graph(self):
        question = "Solve 0 = 4*b + b + 15 for b."
        cg = ComputeGraph(question)
        lookup_value_node = Node(lookup_value)
        solve_system_node = Node(solve_system)
        lookup_value_node.set_arg(solve_system_node)
        cg.root = lookup_value_node
        assert str(cg) == "lookup_value(solve_system('p_0'),'p_1')"
        assert cg.eval() == None
