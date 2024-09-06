from ginnlp import GINNLP
import torch
import sympy as sp

functions = [f'(2*x**2 + {torch.pi} * x + 3)/(x+7)',
             f'(x**2 + 2)/(x**2+1)']

recovered_functions = []
x_train_base = torch.linspace(-5, 5, 101).to('cpu').unsqueeze(0).T
for i, function in enumerate(functions):
    target_function = sp.lambdify('x', function)
    y_train = target_function(x_train_base)

    mask = ~torch.isnan(y_train)
    x_train = x_train_base[mask].unsqueeze(1)
    y_train = y_train[mask]

    ginnLP = GINNLP(num_epochs=500, round_digits=3, start_ln_blocks=1, growth_steps=3,
                    l1_reg=1e-4, l2_reg=1e-4, init_lr=0.01, decay_steps=1000, reg_change=0.5)
    ginnLP.fit(x_train, y_train.squeeze())
    print("Functions learned: {0}".format(i+1))
    print(function)
    print(ginnLP.recovered_eq)