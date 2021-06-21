from dm_math_gym_env.typed_operators import *

typed_examples = {
    "Solve 0 = 4*b + b + 15 for b.": [
        Equation("0 = 4*b + b + 15"),
        Variable("b")
    ],
    "Suppose -3*z + 133 = 4*n - 10, 5*n = 25. Let l = -21 + z. Let r = l + -11. Calculate the least common multiple of 7 and r.": [
        Equation("-3*z + 133 = 4*n - 10"),
        Equation("5*n = 25"),
        Equation("l = -21 + z"),
        Equation("r = l + -11"),
        Value("7"),
        Variable("r")
    ],
    "Calculate the common denominator of 1/(3/(-6)) - 402/(-60) and -71/12.": [
        Rational(sympy.sympify("1/(3/(-6)) - 402/(-60)")),
        Rational("-71/12")
    ],
    "What is the common denominator of -64/1065 and 92/105?": [
        Rational("-64/1065"),
        Rational("92/105")
    ],
    "What is the smallest common multiple of (-4)/12*(-20 - -2) and 4?": [
        Value(sympy.sympify("(-4)/12*(-20 - -2)")),
        Value("4")
    ],
    # "Let q = -54.3 + 54. Suppose 0 = -5*z - 8 - 7. Which is the nearest to -1/5?  (a) 5  (b) z  (c) q": [
    #     Equation("q = -54.3 + 54"),
    #     Equation("0 = -5*z - 8 - 7"),
    #     Rational("-1/5"),
    #     "(a) 5  (b) z  (c) q"
    # ],
    "Let d(j) = -j**3 - 5*j**2 - 4*j + 1. Let n be d(-4). Suppose -5*h = 2*i - 2*h + n, 0 = i + 5*h - 10. What is the nearest to 0 in 1/3, i, -2?": [
        Function("d(j) = -j**3 - 5*j**2 - 4*j + 1"),
        Variable("n"),
        Expression("d(-4)"),
        Equation("-5*h = 2*i - 2*h + n"),
        Equation("0 = i + 5*h - 10"),
        Value("0"),
        Rational("1/3"),
        Variable("i"),
        Value("-2")
    ],
    "Let f = -2.31 + 0.31. What is the nearest to f in 0.3, -2, 0.2?": [
        Equation("f = -2.31 + 0.31"),
        Variable("f"),
        Value("0.3"),
        Value("-2"),
        Value("0.2")
    ],
    "Let o(v) = 77*v + 1. Let b(l) = 155*l + 2. Suppose 4*c - 25 = -c. Let a(u) = c*o(u) - 3*b(u). Is a(-4) composite?": [
        Function("o(v) = 77*v + 1"),
        Function("b(l) = 155*l + 2"),
        Equation("4*c - 25 = -c"),
        Function("a(u) = c*o(u) - 3*b(u)"),
        Expression("a(-4)")
    ],
    "Let j = -5 - 28. Is j/6*(-1 - 13) a composite number?": [
        Equation("j = -5 - 28"),
        Expression("j/6*(-1 - 13)")
    ],
    "Suppose 0 = -j - 4*a + 611, 4*j + a - 1468 = 1051. Is j prime?": [
        Equation("0 = -j - 4*a + 611"),
        Equation("4*j + a - 1468 = 1051"),
        Variable("j")
    ],
    "Let l(b) = -142004*b - 62917*b - 377393*b. Let d be l(-1). Let v = d - 262314. Round v to the nearest 100000.": [
        Function("l(b) = -142004*b - 62917*b - 377393*b"),
        Variable("d"),
        Expression("l(-1)"),
        Equation("v = d - 262314"),
        Variable("v"),
        Value("100000")
    ],
    "Suppose 5*t - 2 = -7. Let z be -612*1 + (-2 - -3). Let c be t/(-4) + z/(-4). What is c rounded to the nearest ten?": [
        Equation("5*t - 2 = -7"),
        Variable("z"),
        Value(sympy.sympify("-612*1 + (-2 - -3)")),
        Variable("c"),
        Expression("t/(-4) + z/(-4)"),
        Variable("c")
    ],
    "Let m = 1.5 - 7.5. Let z = m - -22. Let v = z + -16.00017. Round v to four decimal places.": [
        Equation("m = 1.5 - 7.5"),
        Equation("z = m - -22"),
        Equation("v = z + -16.00017"),
        Variable("v")
    ],
    "Let w(b) = -2*b - 3. Suppose 0*j + 16 = -3*j - o, j + 3*o = 8. Let u = j - -5. What is w(u)?": [
        Function("w(b) = -2*b - 3"),
        Equation("0*j + 16 = -3*j - o"),
        Equation("j + 3*o = 8"),
        Equation("u = j - -5"),
        Expression("w(u)")
    ],
    "Let p(o) = 2*o**3 - 12*o**2 + 6*o - 5. Let i(m) = -m**3 + 6*m**2 - 3*m + 2. Let q be 82/12 - 2/(-12). Let f(s) = q*i(s) + 3*p(s). Determine f(5).": [
        Function("p(o) = 2*o**3 - 12*o**2 + 6*o - 5"),
        Function("i(m) = -m**3 + 6*m**2 - 3*m + 2"),
        Variable("q"),
        Value(sympy.sympify("82/12 - 2/(-12)")),
        Function("f(s) = q*i(s) + 3*p(s)"),
        Expression("f(5)")
    ],
    "Let l(r) be the third derivative of 3*r**6/40 - r**5/60 - 6*r**2. What is l(-1)?": [
        Expression("l(r)"),
        Expression("3*r**6/40 - r**5/60 - 6*r**2"),
        Expression("l(-1)")
    ],
    "Let o = -788/3 - -260. Which is bigger: -0.1 or o?": [
        Equation("o = -788/3 - -260"),
        Value("-0.1"),
        Variable("o")
    ],
    "Let r = 4 + -2. Which is greater: r or 0.09?": [
        Equation("r = 4 + -2"),
        Variable("r"),
        Value("0.09")
    ],
    # "Let q = 17 - 18. Let v be (2 + q)*12/(-16). Is v > -1?": [
    #     Equation("q = 17 - 18"),
    #     Variable("v"),
    #     Expression("(2 + q)*12/(-16)"),
    #     "v > -1"
    # ],
    "Suppose 3*x + 197 = 4*x. Calculate the remainder when x is divided by 33.": [
        Equation("3*x + 197 = 4*x"),
        Variable("x"),
        Value("33")
    ],
    "Suppose -106 = -2*u + s, u - 40 = -5*s + 13. Calculate the remainder when u is divided by 14.": [
        Equation("-106 = -2*u + s"),
        Equation("u - 40 = -5*s + 13"),
        Variable("u"),
        Value("14")
    ],
    "Let x = -41 - -20. Let t = x + 27. Calculate the remainder when t is divided by 4.": [
        Equation("x = -41 - -20"),
        Equation("t = x + 27"),
        Variable("t"),
        Value("4")
    ],
    "Let d = -25019/90 - -278. Let v(j) be the third derivative of 0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j. Suppose v(o) = 0. What is o?": [
        Equation("d = -25019/90 - -278"),
        Expression("v(j)"),
        Expression("0 + 1/27*j**3 - d*j**5 + 1/54*j**4 + 3*j**2 + 0*j"),
        Function("v(o) = 0"),
        Variable("o")
    ],
    "Let g be 2 - (0 - (-1 - -1)). Determine q so that -q**4 - 6*q**2 + 0*q**4 - 3 + g - 4*q - 4*q**3 = 0.": [
        Variable("g"),
        Value(sympy.sympify("2 - (0 - (-1 - -1))")),
        Variable("q"),
        Equation("-q**4 - 6*q**2 + 0*q**4 - 3 + g - 4*q - 4*q**3 = 0")
    ],
    "Let d(k) be the first derivative of -1 - 4/3*k**3 + 0*k + 1/2*k**2. Find z such that d(z) = 0.": [
        Expression("d(k)"),
        Expression("-1 - 4/3*k**3 + 0*k + 1/2*k**2"),
        Variable("z"),
        Function("d(z) = 0")
    ],
    "Suppose -55 = -8*l + 3*l. Let k = l + -7. What is the units digit of k?": [
        Equation("-55 = -8*l + 3*l"),
        Equation("k = l + -7"),
        Variable("k")
    ],
    "Let t(p) = p**3 - 3*p**2 - 4*p + 2. Let a be t(4). Suppose 2*f = a + 2. Let l = f - -12. What is the units digit of l?": [
        Function("t(p) = p**3 - 3*p**2 - 4*p + 2"),
        Variable("a"),
        Expression("t(4)"),
        Equation("2*f = a + 2"),
        Equation("l = f - -12"),
        Variable("l")
    ],
    "Suppose 5*j - 1126 + 331 = 0. What is the tens digit of j?": [
        Equation("5*j - 1126 + 331 = 0"),
        Variable("j")
    ],
    "Suppose 0 = -4*x + 8*x - 40. Let h(i) = i**2 - 9*i - 14. Let n be h(x). Sort -1, 4, n.": [
        Equation("0 = -4*x + 8*x - 40"),
        Function("h(i) = i**2 - 9*i - 14"),
        Variable("n"),
        Expression("h(x)"),
        Value("-1"),
        Value("4"),
        Variable("n")
    ],
    "Let g = 1 + 2. Let a = 0.95 - -0.05. Put a, g, -1 in descending order.": [
        Equation("g = 1 + 2"),
        Equation("a = 0.95 - -0.05"),
        Variable("a"),
        Variable("g"),
        Value("-1")
    ],
    "Let m be (-7)/56 - (-1)/(-8). Sort m, 0, -4 in descending order.": [
        Variable("m"),
        Rational(sympy.sympify("(-7)/56 - (-1)/(-8)")),
        Variable("m"),
        Value("0"),
        Value("-4")
    ],
    "Let w be (-1 + 13)*3/(-6). Let b = w - -6. Let i = 2 - b. Solve -15 = 3*c + i*c for c.": [
        Variable("w"),
        Value(sympy.sympify("(-1 + 13)*3/(-6)")),
        Equation("b = w - -6"),
        Equation("i = 2 - b"),
        Equation("-15 = 3*c + i*c"),
        Variable("c")
    ],
    "Suppose -c + 4*v + 2 = -24, -4*c - 3*v + 9 = 0. Solve 2*b - c = -b for b.": [
        Equation("-c + 4*v + 2 = -24"),
        Equation("-4*c - 3*v + 9 = 0"),
        Equation("2*b - c = -b"),
        Variable("b")
    ],
    "Let v(k) = k**3 + k**2 - k - 3. Let d be v(0). Let a be ((-15)/2)/d*4. Let x = a + -8. Solve -3 + 11 = x*p for p.": [
        Function("v(k) = k**3 + k**2 - k - 3"),
        Variable("d"),
        Expression("v(0)"),
        Variable("a"),
        Expression("((-15)/2)/d*4"),
        Equation("x = a + -8"),
        Equation("-3 + 11 = x*p"),
        Variable("p")
    ],
    "Let h(t) = t**3 + t**2 + 1. Let v(d) = 6*d**3 + 24*d**2 + 4. Let w(j) = 4*h(j) - v(j). What is the third derivative of w(x) wrt x?": [
        Function("h(t) = t**3 + t**2 + 1"),
        Function("v(d) = 6*d**3 + 24*d**2 + 4"),
        Function("w(j) = 4*h(j) - v(j)"),
        Expression("w(x)"),
        Variable("x")
    ],
    "Let v = -7 - -12. Suppose 0 = 2*h - 3*x - 16 - 5, 0 = -v*h + 3*x + 30. What is the first derivative of 5*t - h - t + 0 - 2*t wrt t?": [
        Equation("v = -7 - -12"),
        Equation("0 = 2*h - 3*x - 16 - 5"),
        Equation("0 = -v*h + 3*x + 30"),
        Expression("5*t - h - t + 0 - 2*t"),
        Variable("t")
    ],
    "Let b(y) be the second derivative of -3*y**8/56 - y**4/6 - y. What is the third derivative of b(o) wrt o?": [
        Expression("b(y)"),
        Expression("-3*y**8/56 - y**4/6 - y"),
        Expression("b(o)"),
        Variable("o")
    ],
    "Let p = -3 - -6. Let w(d) = 0*d**2 + p*d**2 - 2*d**2 - 3*d**2. Let t(b) = -3*b. Give t(w(k)).": [
        Equation("p = -3 - -6"),
        Function("w(d) = 0*d**2 + p*d**2 - 2*d**2 - 3*d**2"),
        Function("t(b) = -3*b"),
        Expression("t(w(k))")
    ],
    "Let m(s) = 7*s - 12. Let z(g) = -5*g**2. What is z(m(k))?": [
        Function("m(s) = 7*s - 12"),
        Function("z(g) = -5*g**2"),
        Expression("z(m(k))")
    ],
    "Let w(q) = 2*q**2. Let v(x) be the first derivative of 0*x + 0*x**2 + 4/3*x**3 - 2. Determine v(w(p)).": [
        Function("w(q) = 2*q**2"),
        Expression("v(x)"),
        Expression("0*x + 0*x**2 + 4/3*x**3 - 2"),
        Expression("v(w(p))")
    ],
    # "Let f be 4/22 - 20/(-11). Suppose s = -0*s + 4*n + 12, 0 = -n - f. Which is the second smallest value?  (a) -0.2  (b) s  (c) 2/7": [
    #     Value("f"),
    #     Expression("4/22 - 20/(-11)"),
    #     Equation("s = -0*s + 4*n + 12"),
    #     Equation("0 = -n - f"),
    #     "(a) -0.2  (b) s  (c) 2/7"
    # ],
    # "Let s = 1.5 + -1.5. Suppose 0 = p + p + 8. Which is the third biggest value?  (a) s  (b) -5  (c) p": [
    #     Equation("s = 1.5 + -1.5"),
    #     Equation("0 = p + p + 8"),
    #     "(a) s  (b) -5  (c) p"
    # ],
    # "Let r = 1 - -4. Let u be (-3 - -1)*3/(-2). Suppose -r*s - u = -4*s. Which is the third biggest value?  (a) -0.3  (b) 2/11  (c) s": [
    #     Equation("r = 1 - -4"),
    #     Value("u"),
    #     Expression("(-3 - -1)*3/(-2)"),
    #     Equation("-r*s - u = -4*s"),
    #     "(a) -0.3  (b) 2/11  (c) s"
    # ],
    "Suppose 3*n = -0*x - 3*x + 93, -2*n - 2 = 0. Does 12 divide x?": [
        Equation("3*n = -0*x - 3*x + 93"),
        Equation("-2*n - 2 = 0"),
        Value("12"),
        Variable("x")
    ],
    "Is 1330/(-28)*4/(-2) a multiple of 19?": [
        Value(sympy.sympify("1330/(-28)*4/(-2)")),
        Value("19")
    ],
    "Is 3 - (1344/(-10) + 2/5) a multiple of 36?": [
        Value(sympy.sympify("3 - (1344/(-10) + 2/5)")),
        Value("36")
    ],
    "Suppose 2*y + 12 = 6*y. Suppose y = f - 15. Solve -8 = -4*w, -3*d - 4*w + f = -8*d for d.": [
        Equation("2*y + 12 = 6*y"),
        Equation("y = f - 15"),
        Equation("-8 = -4*w"),
        Equation("-3*d - 4*w + f = -8*d"),
        Variable("d")
    ],
    "Let l(v) = -v**3 + 12*v**2 + 13*v + 2. Let r(q) = -2*q + 5. Let c be r(-4). Let y be l(c). Solve -w + 2 = -3*s - 8, s + 1 = -y*w for s.": [
        Function("l(v) = -v**3 + 12*v**2 + 13*v + 2"),
        Function("r(q) = -2*q + 5"),
        Variable("c"),
        Expression("r(-4)"),
        Variable("y"),
        Expression("l(c)"),
        Equation("-w + 2 = -3*s - 8"),
        Equation("s + 1 = -y*w"),
        Variable("s")
    ],
    "Suppose 0 = f + 1, 17*f = 5*w + 12*f - 15. Suppose -3*s + 2 = -13. Suppose s*c + 3*j = 36, 3*c + 0*j - 18 = -3*j. Solve 20 = a - 5*z, -2*a + c = -w*z - 7 for a.": [
        Equation("0 = f + 1"),
        Equation("17*f = 5*w + 12*f - 15"),
        Equation("-3*s + 2 = -13"),
        Equation("s*c + 3*j = 36"),
        Equation("3*c + 0*j - 18 = -3*j"),
        Equation("20 = a - 5*z"),
        Equation("-2*a + c = -w*z - 7"),
        Variable("a")
    ],
    "Let q be (25 + 1)/2 - (5 + -3). What is the highest common divisor of q and 99?": [
        Variable("q"),
        Value(sympy.sympify("(25 + 1)/2 - (5 + -3)")),
        Variable("q"),
        Value("99")
    ],
    "Let n(j) = 5*j**3 - j**2 + 2*j - 1. Let u be 7/9 - 6/(-27). Let v be n(u). Calculate the greatest common factor of 1 and v.": [
        Function("n(j) = 5*j**3 - j**2 + 2*j - 1"),
        Variable("u"),
        Value(sympy.sympify("7/9 - 6/(-27)")),
        Variable("v"),
        Expression("n(u)"),
        Value("1"),
        Variable("v")
    ],
    "Let f be (-6)/5*(-360)/(-27). Suppose -5*k - 5*a = -335, 0*k = 4*k + 3*a - 271. Let p = k + f. Calculate the greatest common factor of p and 6.": [
        Variable("f"),
        Value(sympy.sympify("(-6)/5*(-360)/(-27)")),
        Equation("-5*k - 5*a = -335"),
        Equation("0*k = 4*k + 3*a - 271"),
        Equation("p = k + f"),
        Variable("p"),
        Value("6")
    ],
    "Let k(w) = -w**2 + 13*w - 4. What are the prime factors of k(6)?": [
        Function("k(w) = -w**2 + 13*w - 4"),
        Expression("k(6)")
    ],
    "Let w(x) = x**2 + 10*x + 24. List the prime factors of w(-11).": [
        Function("w(x) = x**2 + 10*x + 24"),
        Expression("w(-11)")
    ],
    "Let x(m) = m**3 + 6*m**2 - 7*m + 4. Let k be x(-7). Suppose g + 4*u = -12, 3*u = -0*g - k*g + 4. What are the prime factors of g?": [
        Function("x(m) = m**3 + 6*m**2 - 7*m + 4"),
        Variable("k"),
        Expression("x(-7)"),
        Equation("g + 4*u = -12"),
        Equation("3*u = -0*g - k*g + 4"),
        Variable("g"),
    ]
}
