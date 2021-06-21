import re
import sympy
from typing import List, Dict, Set
import multiprocess as mp
import time
# from math import log


# type definitions --------------------------------------


class EquationOrExpression:
    def __init__(self, equation_or_expression: str):
        self.equation_or_expression = str(equation_or_expression)

    def __str__(self):
        return self.equation_or_expression


class Equation(object):
    def __init__(self, equation: str):
        assert len(equation.split("=")) == 2
        self.equation = equation

    def __str__(self):
        return self.equation

    def __eq__(self, equation):
        return self.equation == str(equation)

    def split(self, split_on):
        return self.equation.split(split_on)


class Function(Equation):
    def __init__(self, function: str):
        assert len(function.split("=")) == 2
        function_arg_pattern = "([a-zA-Z0-9\s]+)\(([a-zA-Z0-9\s]+)\)"
        # extract parts of function definition
        lhs, rhs = function.split("=")
        match = re.match(function_arg_pattern, lhs)
        assert match is not None
        self.name, self.parameter = match.group(1), match.group(2)
        self.function = function
        self.equation = function

    def __str__(self):
        return str(self.function)

    def __eq__(self, function):
        return self.function == str(function)


class Expression(object):
    def __init__(self, expression: str):
        assert "=" not in expression
        self.expression = str(expression)

    def __str__(self):
        return self.expression

    def __eq__(self, other):
        return self.expression == other.expression

    def __hash__(self):
        return hash(self.expression)


class Variable(Expression):
    def __init__(self, variable: str):
        self.variable = str(variable)
        assert variable.isalpha()

    def __str__(self):
        return self.variable

    def __eq__(self, variable):
        return self.variable == variable

    def __hash__(self):
        return hash(self.variable)


class Value(Expression):
    def __init__(self, value: float):
        self.value = float(value)

    def __str__(self):
        if self.value % 1 == 0:
            return str(int(self.value))
        else:
            return str(self.value)

    def __eq__(self, value):
        return self.value == value.value

    def __hash__(self):
        return hash(str(self.value))

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

class Rational(Expression):
    def __init__(self, rational: str):
        self.rational = str(rational)
        try:
            self.numerator, self.denominator = [Value(x) for x in self.rational.split('/')]
        except:
            self.numerator = Value(self.rational)
            self.denominator = 1

    def __str__(self):
        return self.rational

    def __eq__(self, rational):
        return self.rational == str(rational)

    def __hash__(self):
        return hash(str(self.rational))

    def __lt__(self, other):
        return sympy.Rational(self.rational) < sympy.Rational(other.rational)

# operator definitions --------------------------------------


# solve_system(system: List[Equation]) -> Dict[Variable, Set[Value]]
def solve_system(system: list) -> dict:
    """
    solve a system of linear equations.

    :param system: List[
    :return: Dict[Variable, Value]
    """
    def sympy_solve(system, return_dict):
        # run in try-except to suppress exception logging (must be done here due to use of multiprocess)
        try:
            solutions = sympy.solve(system, rational=True)
            return_dict["solutions"] = solutions
        except:
            pass
    sympy_equations = []
    for equation in system:
        lhs, rhs = str(equation).split("=")
        sympy_eq = sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))
        sympy_equations.append(sympy_eq)

    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=sympy_solve, args=(sympy_equations, return_dict))
    p.start()
    p.join(1)

    if p.is_alive():
        p.terminate()
        p.join()
    solutions = return_dict.get("solutions", [])

    # Convert list to dictionary if no solution found.
    if len(solutions) == 0:
        raise Exception("no solution found")
    elif type(solutions) is dict:
        return {Variable(str(k)): set([Rational(v)]) for k, v in solutions.items()}
    elif type(solutions) is list:
        solutions_dict = {}
        for soln in solutions:
            for k, v in soln.items():
                if str(k) in solutions_dict.keys():
                    solutions_dict[Variable(str(k))].add(Rational(v))
                else:
                    solutions_dict[Variable(str(k))] = set([Rational(v)])
        return solutions_dict


# append(system: List[Equation], equation: Equation) -> List[Equation]
def append(system: list, equation: Equation) -> list:
    if not system:
        return [equation]
    else:
        system.append(equation)
        return system


def append_to_empty_list(equation: Equation) -> list:
    return [equation]


# lookup_value(mapping: Dict[Variable, Set[Value]], key: Variable)
def lookup_value(mapping: dict, key: Variable) -> object:
    # TODO: figure out how to constrain output type in this case (multiple output types)
    assert key in mapping
    corresponding_set = mapping[key]
    if len(corresponding_set) == 1:
        return corresponding_set.pop()
    else:
        return corresponding_set


# lookup_value_equation(mapping: Dict[Variable, Set[Value]], key: Variable) -> Equation:
def lookup_value_equation(mapping: dict, key: Variable) -> Equation:
    assert key in mapping
    corresponding_set = mapping[key]
    value = corresponding_set.pop()
    return Equation(f"{key} = {value}")


def make_equation(expression1: Expression, expression2: Expression) -> Equation:
    return Equation(f"{expression1} = {expression2}")


def make_function(expression1: Expression, expression2: Expression) -> Function:
    return Function(f"{expression1} = {expression2}")


def extract_isolated_variable(equation: Equation) -> Variable:
    lhs, rhs = str(equation).split("=")
    lhs, rhs = lhs.strip(), rhs.strip()
    if len(lhs) == 1 and lhs.isalpha():
        return lhs
    elif len(rhs) == 1 and rhs.isalpha():
        return rhs
    else:
        raise Exception("there is no isolated variable")


def project_lhs(equation: Equation) -> Expression:
    return Expression(str(equation).split("=")[0].strip())


def project_rhs(equation: Equation) -> Expression:
    return Expression(str(equation).split("=")[1].strip())


def substitution_left_to_right(arb: object, eq: Equation) -> object:
    return object(str(arb).replace(str(project_lhs(eq)), str(project_rhs(eq))))


def substitution_right_to_left(arb: object, eq: Equation) -> object:
    """substitution_right_to_left"""
    return object(str(arb).replace(str(project_rhs(eq)), str(project_lhs(eq))))


def factor(inpt: Expression) -> Expression:
    output = Expression(str(sympy.factor(inpt)))
    return output


def simplify(inpt: object) -> object:
    if "=" in str(inpt):
        lhs, rhs = str(inpt).split("=")
        lhs, rhs = lhs.strip(), rhs.strip()
        output = Equation(f"{sympy.simplify(lhs)} = {sympy.simplify(rhs)}".strip())
    else:
        output = Expression(str(sympy.simplify(str(inpt))).strip())
    return output


def differentiate(expression: Expression) -> Expression:
    derivative = sympy.diff(sympy.sympify(str(expression)))
    return Expression(str(derivative))


def differentiate_wrt(expression: Expression, variable: Variable) -> Expression:
    derivative = sympy.diff(sympy.sympify(str(expression)), sympy.sympify(str(variable)))
    return Expression(str(derivative))


def replace_arg(function: Function, var: Variable) -> Function:
    lhs, rhs = function.split(" = ")
    lhs = lhs.replace('(' + str(function.parameter) + ')', '(' + str(var) + ')')
    rhs = rhs.replace(str(function.parameter), str(var))
    return make_function(Expression(lhs), Expression(rhs))


def mod(numerator: Value, denominator: Value) -> Value:
    return Value(numerator.value % denominator.value)


def divides(numerator: Value, denominator: Value) -> bool:
    return numerator.value % denominator.value == 0


def gcd(x: Value, y: Value) -> Value:
    """greatest common divisor"""
    from math import gcd

    return Value(gcd(int(x.value), int(y.value)))


def is_prime(x: Value) -> bool:
    return sympy.isprime(int(x.value))


def lcm(x: Value, y: Value) -> Value:
    """least common multiple"""
    import math
    assert int(x.value) == x.value and int(y.value) == y.value
    x, y = int(x.value), int(y.value)
    return Value(abs(x * y) // math.gcd(x, y))

def lcd(x: Rational, y: Rational) -> Value:
    """least common denominator"""
    return lcm(x.denominator, y.denominator)


def prime_factors(n: Value) -> set:
    # https://stackoverflow.com/questions/16996217/prime-factorization-list
    assert int(n.value) == n.value

    if is_prime(n):
        return n

    n = int(n.value)
    divisors = [d for d in range(2, n // 2 + 1) if n % d == 0]

    return set(
        [Value(d) for d in divisors if all(d % od != 0 for od in divisors if od != d)]
    )


def evaluate_function(function_definition: Function, function_argument: Expression) -> Value:
    """
    :param function_definition: e.g. 'f(x) = x + x**3'
    :param function_argument: e.g. either '2' or 'f(2)'
    :return:
    """
    function_definition_pattern = "([a-zA-Z0-9\s]+)\(([a-zA-Z0-9\s]+)\)"
    function_arg_pattern = "([a-zA-Z0-9\s]+)\((-?[a-zA-Z0-9\s]+)\)"
    # extract parts of function definition
    lhs, rhs = str(function_definition).split("=")
    match = re.match(function_definition_pattern, lhs)
    function_name_from_definition, function_parameter = match.group(1), match.group(2)
    # extract parts of function argument
    function_argument_ = re.match(function_arg_pattern, str(function_argument))
    if function_argument_ is not None:
        function_name_from_argument, function_argument = (
            function_argument_.group(1),
            function_argument_.group(2),
        )
        assert function_name_from_definition == function_name_from_argument
    # evaluate function
    rhs_with_arg = rhs.replace(function_parameter, f'({function_argument})')
    return Value(eval(rhs_with_arg))


def not_op(x: bool) -> bool:
    assert type(x) == bool
    return not x

