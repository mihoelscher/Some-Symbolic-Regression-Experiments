from mysr.RationalSR import RationalFunction
import torch
import sympy as sp
from mysr.function_generator import get_random_degrees, get_functions_for_degrees, get_last_coefficient

degrees = get_random_degrees(10, 5)
functions = get_functions_for_degrees(degrees)

recovered_functions = []
x_train = torch.linspace(-5, 5, 1001).to('cpu')
for i, function in enumerate(functions):
    target_function = sp.lambdify('x', function)
    y_train = target_function(x_train)
    my_model = RationalFunction(5)
    my_model.fit(x_train, y_train, num_epochs=1000, regularization_parameter=0.1, verbose=0, regularization_order=None)
    print("Functions trained: {}".format(i+1))
    print(function)
    factor = float(get_last_coefficient(function))
    recovered_function = my_model.get_function(factor)
    recovered_functions.append(recovered_function)
    print(recovered_function)
