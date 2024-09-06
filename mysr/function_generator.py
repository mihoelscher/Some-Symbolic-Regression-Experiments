import numpy as np
import sympy as sp
import re


def generate_random_polynomial(max_degree: int, min_value: float = -5, max_value: float = 5):
    coefficients = np.random.uniform(min_value, max_value, max_degree + 1)  # random values in [-min_value,max_value]
    _x = sp.Symbol('x')
    return sp.sympify(sum(c.round(4) * _x ** i for i, c in enumerate(coefficients)))


def generate_random_function(max_degree_p: int, max_degree_q: int, min_value: float = -5, max_value: float = 5):
    return (generate_random_polynomial(max_degree_p, min_value, max_value)
            / generate_random_polynomial(max_degree_q, min_value, max_value))


def generate_random_function_list(n: int, max_degree_p: int, max_degree_q: int, min_value: float = -5,
                                  max_value: float = 5):
    return [generate_random_function(max_degree_p, max_degree_q, min_value, max_value) for _ in range(n)]


def get_functions_for_degrees(_degrees, n: int = 1):
    _functions = []
    for _degrees in _degrees:
        _functions += generate_random_function_list(n, _degrees[0], _degrees[1])
    return _functions


def get_random_degrees(n, max_degree):
    _degrees = []
    for _ in range(n):
        _degrees.append([np.random.randint(1, max_degree + 1), np.random.randint(1, max_degree + 1)])
    return _degrees

def get_last_coefficient(rational_function):
    # Extract the denominator part
    try:
        denominator = re.search(r'/\((.*)\)', rational_function).group(1)
    except AttributeError:
        return 1
    # Find the last number without 'x' in the denominator
    last_constant_number = re.findall(r'[-+]?\d*\.?\d+(?!\*)', denominator)[-1]
    return last_constant_number

if __name__ == '__main__':
    print('Starting function_generator.py test...')
    degrees = get_random_degrees(1, 5)
    function = get_functions_for_degrees(degrees)[0]
    result = sp.lambdify('x', function)
    print(result(2))

