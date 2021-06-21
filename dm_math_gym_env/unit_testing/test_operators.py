import unittest
import numpy as np
from dm_math_gym_env.typed_operators import *
from sympy import sympify


class Test(unittest.TestCase):
    def test_value(self):
        assert Value(1) == Value(1.0)
        assert {Value(1)} == {Value(1)}
        assert {Value(1)} == {Value(1.0)}

    def test_solve_system(self):
        system = [Equation("x = 1")]
        assert solve_system(system) == {Variable("x"): {Rational(1)}}

        system = [Equation("x = 1"), Equation("y = 1")]
        assert solve_system(system) == {
            Variable("x"): {Rational(1)},
            Variable("y"): {Rational(1)},
        }

        system = [Equation("x + y = 1"), Equation("x - y = 1")]
        assert solve_system(system) == {
            Variable("x"): {Rational(1)},
            Variable("y"): {Rational(0)},
        }

        system = [Equation("3*x + y = 9"), Equation("x + 2*y = 8")]
        assert solve_system(system) == {
            Variable("x"): {Rational(2)},
            Variable("y"): {Rational(3)},
        }

        # # fails on singular matrix
        system = [
            Equation("x + 2*y - 3*z = 1"),
            Equation("3*x - 2*y + z = 2"),
            Equation("-x + 2*y - 2*z = 3"),
        ]
        self.assertRaises(Exception, solve_system, system)

        # system with floating point coefficients
        system = [Equation("-15 = 3*c + 2.0*c")]
        assert solve_system(system) == {Variable("c"): {Rational(-3)}}

        # quadratic equation
        system = [Equation("-3*h**2/2 - 24*h - 45/2 = 0")]
        assert solve_system(system) == {Variable("h"): {Rational(-15), Rational(-1)}}

        # unsolvable equation / infinite loop without timeout
        system = [Equation('-4*i**3*j**3 - 2272*i**3 - 769*i**2*j - j**3 = 1')]
        self.assertRaises(Exception, solve_system, system)

        system = [Equation('-g**3 - 9*g**2 - g + l(g) - 10 = 0')]
        self.assertRaises(Exception, solve_system, system)

        # unsolvable equation / infinite loop without timeout
        system = [Equation('-4*i**3*j**3 - 2272*i**3 - 769*i**2*j - j**3 = 1')]
        self.assertRaises(Exception, solve_system, system)

        system = [Equation('9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0')]
        assert solve_system(system) == {Variable("s"): {Rational(-1), Rational('1/3'), Rational(997)}}

        system = [Equation('-3*h**2/2 - 24*h - 45/2 = 0')]
        assert solve_system(system) == {Variable("h"): {Rational('-1'), Rational('-15')}}
        # print([(str(k), [str(v) for v in vset]) for k,vset in solve_system(system).items()])

    def test_is_prime(self):
        assert is_prime(Value('3'))
        assert not_op(is_prime(Value('4')))

    def test_prime_factors(self):
        result = prime_factors(Value('7380'))
        assert ", ".join([str(x) for x in sorted(list(result))]) == '2, 3, 5, 41'

    def test_lcd(self):
        assert lcd(Rational('2/3'), Rational('3/5')) == Value('15')
        assert lcd(Rational('2/3'), Rational('3/5')) == Value('15')

    def test_third_derivative(self):
        inpt = Expression('-272*j**5 + j**3 - 8234*j**2')
        third_derivative = differentiate(differentiate(differentiate(inpt)))
        assert sympify(third_derivative) == sympify(Expression('-16320*j**2 + 6'))

    def test_function_evaluation1(self):
        f0 = Function('l(t) = -t**2 - 7*t - 7')
        f1 = Expression('l(-5)')
        output = evaluate_function(f0, f1)
        assert output == Value(3)

    def test_function_evaluation2(self):
        f0 = Function('x(k) = k**3 + k**2 + 6*k + 9')
        f1 = Expression('x(-2)')
        output = evaluate_function(f0, f1)
        assert output == Value(-7)

    def test_diff_distractors(self):
        expression = Expression('442*c**4 + 248')
        output1 = differentiate(factor(expression))
        output2 = factor(differentiate(expression))
        output3 = differentiate(simplify(expression))
        output4 = simplify(differentiate(expression))
        answer = Expression('1768*c**3')
        assert output1 == answer
        assert output2 == answer
        assert output3 == answer
        assert output4 == answer

    def test_lcd(self):
        arg1 = Rational('-64/1065')
        arg2 = Rational('92/105')
        output = lcd(arg1, arg2)
        assert output == Value('7455')

    def test_replace_arg(self):
        f = Function('funk(k) = k**3 + k**2 + 6*k + 9')
        replaced_f = Function('funk(x) = x**3 + x**2 + 6*x + 9')
        assert replace_arg(f, Variable('x')) == replaced_f