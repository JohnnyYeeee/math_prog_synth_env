from inspect import signature
from dm_math_gym_env.utils import extract_formal_elements
from dm_math_gym_env.typed_operators import *
import signal

class Node:
    def __init__(self, action):
        self.action = action
        self.args = []
        if type(self.action) == str:  # if action is a formal element
            self.num_parameters = 0
            self.types = []
        else:
            self.num_parameters = len(signature(self.action).parameters)
            self.types = [
                type_.annotation
                for name, type_ in signature(self.action).parameters.items()
            ]

    def set_arg(self, node):
        assert len(self.args) < self.num_parameters
        self.args.append(node)

    def set_args(self, nodes):
        assert len(self.args) == 0
        self.args = nodes

    def are_args_set(self):
        return len(self.args) == self.num_parameters


class ComputeGraph:
    def __init__(self, question):
        self.formal_elements = extract_formal_elements(question)
        self.formal_element_types = [type(f) for f in self.formal_elements]
        self.root = None
        self.current_node = None  # reference to the first node (breadth-first) that requires one or more arguments
        self.queue = []
        self.n_nodes = 0

    def lookup_formal_element(self, action):
        """f12 => int(12)"""
        try:
            selected_formal_element = self.formal_elements[int(action[1:])]
        except:
            selected_formal_element = (
                action  # if index is out of range, return dummy value
            )
        return selected_formal_element

    def build_string(self, current_node):
        if type(current_node) == str:  # case: param
            return f"'{current_node}'"
        elif type(current_node.action) == str:  # case: formal element
            assert current_node.action[0] == "f"
            formal_element = self.lookup_formal_element(current_node.action)
            return f"{type(formal_element).__name__}('{formal_element}')"
        elif current_node.action is None:  # case: None (i.e. for an ap)
            return "None"
        else:
            arg_strings = []
            if len(current_node.args) < current_node.num_parameters:
                num_params = current_node.num_parameters
                num_args = len(current_node.args)
                args = current_node.args + [
                    f"p_{i}" for i in range(num_args, num_params)
                ]
            else:
                args = current_node.args
            for arg in args:
                arg_string = self.build_string(arg)
                arg_strings.append(arg_string)
            return f"{current_node.action.__name__}({','.join(['{}'.format(arg_string) for arg_string in arg_strings])})"

    def __str__(self):
        """
        traverse the graph to construct a string representing the compute graph.
        :return:
        """
        return self.build_string(self.root)

    def eval(self):
        """
        evaluate the compute graph
        :return: the output of the compute graph
        """
        try:
            string_to_eval = str(self)
            if "\'p_" in string_to_eval:
                raise Exception("unreplaced params are in arb, e.g. 'p_0'")
            output = eval(string_to_eval)
            # if output is a set, reformat as a sorted string
            if type(output) == set:
                return ", ".join([str(x) for x in sorted(list(output))])
            else:
                return output
        except:
            return None

    def add(self, action):
        """
        Add an action to the compute graph. Elements are added breadth-first order: KNOB.

        :param action: either an operator or a formal element
        """
        if self.root is None:
            self.root = Node(action)
            if not self.root.are_args_set():
                self.current_node = self.root
            else:
                self.current_node = None
        else:
            new_node = Node(action)
            self.current_node.set_arg(new_node)
            if new_node.num_parameters > 0:
                self.queue.append(
                    new_node
                )  # add new node to queue for later processing
            if self.current_node.are_args_set():
                if len(self.queue) > 0:
                    self.current_node = self.queue.pop()
                else:
                    self.current_node = None

    def reset(self):
        self.root = None
        self.current_node = None
        self.queue = []
