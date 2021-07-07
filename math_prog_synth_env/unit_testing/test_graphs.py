import unittest
from math_prog_synth_env.utils import extract_formal_elements, cast_formal_element
from math_prog_synth_env.typed_operators import *


class Test(unittest.TestCase):
    """all test cases are taken from train-easy"""

    def test_easy_algebra__linear_1d(self):
        question = "Solve 0 = 4*b + b + 15 for b."
        fs = extract_formal_elements(question)
        assert fs == [Equation("0 = 4*b + b + 15"), Variable("b")]
        system = append_to_empty_list(fs[0])
        solution = solve_system(system)
        value = lookup_value(solution, fs[1])
        assert value == Value(-3)

    def test_easy_algebra__linear_1d_composed(self):
        question = "Let w be (-1 + 13)*3/(-6). Let b = w - -6. Let i = 2 - b. Solve -15 = 3*c + i*c for c."
        f = extract_formal_elements(question)
        assert f == [
            Variable("w"),
            Value(sympy.sympify("(-1 + 13)*3/(-6)")),
            Equation("b = w - -6"),
            Equation("i = 2 - b"),
            Equation("-15 = 3*c + i*c"),
            Variable("c"),
        ]
        eq1 = make_equation(f[0], f[1])
        system = append(append(append_to_empty_list(eq1), f[2]), f[3])
        soln = solve_system(system)
        i_eq = lookup_value_equation(soln, extract_isolated_variable(f[3]))
        lin_eq = substitution_left_to_right(f[4], i_eq)
        assert lookup_value(solve_system(append_to_empty_list(lin_eq)), f[5]) == Value(-3)

    def test_easy_algebra__linear_2d(self):
        question = "Solve 0 = 4*f - 0*t - 4*t - 4, -4*f + t = -13 for f."
        f = extract_formal_elements(question)
        assert f == [
            Equation("0 = 4*f - 0*t - 4*t - 4"),
            Equation("-4*f + t = -13"),
            Variable("f"),
        ]

        assert lookup_value(
            solve_system(append(append_to_empty_list(f[0]), f[1])), f[2]
        ) == Value(4)

    def test_algebra__linear_2d_composed(self):
        question = "Suppose 2*y + 12 = 6*y. Suppose y = f - 15. Solve -8 = -4*w, -3*d - 4*w + f = -8*d for d."
        f = extract_formal_elements(question)
        assert f == [
            Equation("2*y + 12 = 6*y"),
            Equation("y = f - 15"),
            Equation("-8 = -4*w"),
            Equation("-3*d - 4*w + f = -8*d"),
            Variable("d"),
        ]
        system = append(append(append(append_to_empty_list(f[0]), f[1]), f[2]), f[3])
        assert lookup_value(solve_system(system), f[4]) == Value(-2)

    def test_algebra__polynomial_roots_1(self):
        question = "Solve -3*h**2/2 - 24*h - 45/2 = 0 for h."
        f = extract_formal_elements(question)
        assert f == [Equation("-3*h**2/2 - 24*h - 45/2 = 0"), Variable("h")]
        soln = lookup_value(solve_system(append_to_empty_list(f[0])), f[1])
        assert soln == {Rational(-1), Rational(-15)}

    def test_algebra__polynomial_roots_2(self):
        question = "Factor -n**2/3 - 25*n - 536/3."
        f = extract_formal_elements(question)
        assert f == [Expression("-n**2/3 - 25*n - 536/3")]
        assert factor(f[0]) == Expression("-(n + 8)*(n + 67)/3")

    def test_algebra__polynomial_roots_3(self):
        question = (
            "Find s such that 9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0."
        )
        f = extract_formal_elements(question)
        assert f == [
            Variable("s"),
            Equation("9*s**4 - 8958*s**3 - 14952*s**2 - 2994*s + 2991 = 0"),
        ]
        assert lookup_value(solve_system(append_to_empty_list(f[1])), f[0]) == {
            Rational(-1),
            Rational('1/3'),
            Rational(997),
        }

    def test_algebra__polynomial_roots_composed_1(self):
        question = "Let d = -25019/90 - -278. Let v(j) be the third derivative of 0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j. Suppose v(o) = 0. What is o?"
        f = extract_formal_elements(question)
        assert f == [
            Equation("d = -25019/90 - -278"),
            Expression("v(j)"),
            Expression("0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j"),
            Function("v(o) = 0"),
            Variable("o"),
        ]
        d = simplify(f[0])
        function = substitution_left_to_right(f[2], d)
        v = differentiate(differentiate(differentiate(function)))
        v_eq = make_function(f[1], v)
        v_eq_o = replace_arg(v_eq, f[4])
        equation = substitution_left_to_right(
            f[3], v_eq_o
        )  # e.g. x.subs(sym.sympify('f(x)'), sym.sympify('v'))
        assert lookup_value(solve_system(append_to_empty_list(equation)), f[4]) == {
            Rational('-1/3'),
            Rational(1),
        }

    def test_calculus__differentiate(self):
        question = "What is the second derivative of 2*c*n**2*z**3 + 30*c*n**2 + 2*c*n*z**2 - 2*c + n**2*z**2 - 3*n*z**3 - 2*n*z wrt n?"
        f = extract_formal_elements(question)
        assert f == [Expression('2*c*n**2*z**3 + 30*c*n**2 + 2*c*n*z**2 - 2*c + n**2*z**2 - 3*n*z**3 - 2*n*z'), Variable('n')]
        assert differentiate_wrt(differentiate_wrt(f[0], f[1]), f[1]) == Expression('4*c*z**3 + 60*c + 2*z**2')

    def test_numbers__div_remainder(self):
        question = "Calculate the remainder when 93 is divided by 59."
        f = extract_formal_elements(question)
        assert f == [Value("93"), Value("59")]
        assert mod(f[0], f[1]) == Value("34")

    def test_numbers__gcd(self):
        question = "Calculate the greatest common fac of 11130 and 6."
        f = extract_formal_elements(question)
        assert f == [Value("11130"), Value("6")]
        assert gcd(f[0], f[1]) == Value("6")

    def test_numbers__is_factor(self):
        question = "Is 15 a fac of 720?"
        f = extract_formal_elements(question)
        assert f == [Value("15"), Value("720")]
        assert divides(f[1], f[0]) == True

    def test_numbers__is_prime(self):
        question = "Is 93163 a prime number?"
        f = extract_formal_elements(question)
        assert f == [Value("93163")]
        assert is_prime(f[0]) == False

    def test_numbers__lcm(self):
        question = "Calculate the smallest common multiple of 351 and 141."
        f = extract_formal_elements(question)
        assert f == [Value("351"), Value("141")]
        assert lcm(f[0], f[1]) == Value("16497")

    def test_numbers__list_prime_factors(self):
        question = "What are the prime factors of 329?"
        f = extract_formal_elements(question)
        assert f == [Value("329")]
        assert prime_factors(f[0]) == {Value(7), Value(47)}

    def test_polynomials_evaluate(self):
        question = "Let i(h) = -7*h - 15. Determine i(-2)."
        f = extract_formal_elements(question)
        assert f == [Function("i(h) = -7*h - 15"), Expression("i(-2)")]
        assert evaluate_function(f[0], f[1]) == Value(-1)

    #  requiring new operators --------------------------------------------

    # def test_comparison__closest(self):
    #     question = 'Which is the closest to -1/3?  (a) -8/7  (b) 5  (c) -1.3'
    #     f = extract_formal_elements(question)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'

    # def test_comparison__pair_composed(self):
    #     question = 'Let o = -788/3 - -260. Which is bigger: -0.1 or o?'
    #     f = extract_formal_elements(question)
    #     assert f == [Equation('o = -788/3 - -260'), Value('-0.1'), Variable('o')]
    #     o = sy(f[0])
    #     m = max_arg(f[1], pr(o))
    #     assert srl(m, o) == Value('-0.1')

    # def test_comparison__sort_composed(self):
    #     question = 'Suppose $f[0 = -4*x + 8*x - 40]. Let $f[h(i) = i**2 - 9*i - 14]. Let $f[n] be $f[h(x)]. Sort $f[-1], $f[4], $f[n].'
    #     f = extract_formal_elements(question)
    #     x = lv(ss(f[0]), get

    # def test_arithmetic__add_or_sub_in_base(self):
    #     question = 'In base 13, what is 7a79 - -5?'
    #     f = extract_formal_elements(question)
    #     assert f == ['13', '7a79 - -5']
    #     assert eval_in_base(f[1], f[0]) == '7a81'
    #
    # def test_arithmetic__nearest_integer_root_1(self):
    #     question = 'What is the square root of 664 to the nearest 1?'
    #     f = extract_formal_elements(question)
    #     root_f1 = root(f[1], f[0])
    #     rounded_root_f1 = round_to_int(root_f1, f[2])
    #     assert rounded_root_f1 == '26'
    #
    # def test_arithmetic__nearest_integer_root_2(self):
    #     question = 'What is $f[1699] to the power of $f[1/6], to the nearest $f[1]?'
    #     f = extract_formal_elements(question)
    #     power_f0 = power(f[0], f[1])
    #     rounded_power_f0 = round_to_int(power_f0, f[2])
    #     assert rounded_power_f0 == '3'
