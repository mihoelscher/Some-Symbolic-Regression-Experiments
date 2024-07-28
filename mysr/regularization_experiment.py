import sympy
import torch
from matplotlib import pyplot as plt
from torch import optim

import data_utility
from RationalSR import RationalFunction

if __name__ == '__main__':
    device = 'cpu'
    x_train = torch.linspace(-3, 3, 1000).to(device)
    y_train = (2 * x_train ** 2 + 3.141 * x_train + 3)
    target_function = sympy.lambdify('x', sympy.sympify(f'(2*x**2 + 3.141 * x + 3)'))

    # Model without reg
    model_none = RationalFunction(2, 0).to(device)
    loss_history_none = model_none.fit(x_train, y_train, num_epochs=1000, regularization_parameter=0.1, regularization_order=None,
                                  optimizer=optim.Adam(model_none.parameters(), lr=0.01))
    model_none.eval()

    recovered_function = sympy.lambdify('x', model_none.get_function())
    plt.show()


def plot_multiple_losses(models, losses):
    with torch.no_grad():
        # print("Learned coefficients for P(x):", model_none.coeffs_p)
        # print("Learned coefficients for Q(x):", model_none.coeffs_q)
        print("Recovered function: ", model_none.get_function())
    # data_utility.function_to_plot(target_function, recovered_function)
    fig, ax = data_utility.get_loss_plot(loss_history_none)
