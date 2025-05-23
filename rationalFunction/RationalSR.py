import matplotlib.pyplot as plt
import sympy
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

import data_utility


class RationalFunction(nn.Module):
    """
    A PyTorch module that represents and learns a rational function.

    The rational function is of the form P(x) / Q(x), where P(x) and Q(x) are polynomials.
    The coefficients of these polynomials are learnable parameters.

    Attributes:
    coeffs_p (nn.Parameter): The coefficients of the numerator polynomial P(x).
    coeffs_q (nn.Parameter): The coefficients of the denominator polynomial Q(x).

    Methods:
    forward(x):
        Computes the forward pass through the rational function.
    fit(x_train, y_train, num_epochs=100, learning_rate=0.01):
        Trains the RationalFunction model using the provided training data.
    get_function():
        Returns the rational function.
    """

    def __init__(self, degree_p=1, degree_q=1):
        """
        Initializes the RationalFunction module.

        Parameters:
        num_coeffs_p (int): The number of coefficients in the numerator polynomial P(x).
        num_coeffs_q (int): The number of coefficients in the denominator polynomial Q(x).
        """
        super(RationalFunction, self).__init__()
        self.degree_p = degree_p
        self.degree_q = degree_q
        self.coeffs_p = nn.Parameter(torch.rand(degree_p + 1))
        self.coeffs_q = nn.Parameter(torch.rand(degree_q + 1))
        self.losses = []

    def forward(self, x) -> Tensor:
        # Compute the numerator P(x) and denominator Q(x)
        numerator = sum(self.coeffs_p[-i - 1] * x ** i for i in range(len(self.coeffs_p)))
        denominator = sum(self.coeffs_q[-i - 1] * x ** i for i in range(len(self.coeffs_q)))
        return torch.Tensor(numerator / denominator)

    def fit_once(self, x_input, target, num_epochs=1000, regularization_parameter=0.1, verbose=1,
            regularization_order: int | float = None):
        """
        Fit the RationalFunction model to the training data.

        Parameters:
        x_train (torch.Tensor): The input training data.
        y_train (torch.Tensor): The target training data.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.

        Returns:
        losses (list): A list of losses computed during training.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = None
        self.train()
        self.losses = []
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x_input)
            loss = criterion(y_pred, target)
            self.losses.append(loss.item())
            # Add regularization for sparsity
            if regularization_order is not None:
                loss += (sum([torch.linalg.vector_norm(p, ord=regularization_order) for p in self.parameters()])
                         * regularization_parameter)
            loss.backward()
            optimizer.step()

            # adjust learning rate with an optional scheduler
            if scheduler is not None:
                scheduler.step(loss)

            # prune coefficients
            with torch.no_grad():
                self.coeffs_p[torch.abs(self.coeffs_p) < 0.001] = 0.0
                self.coeffs_q[torch.abs(self.coeffs_q) < 0.001] = 0.0

            if verbose == 1 and (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.9f}')
                print(self.get_function())
        # round if near to integer (diff less than 0.01)
        with torch.no_grad():
            mask = torch.abs(self.coeffs_p - torch.round(self.coeffs_p)) < 0.01
            self.coeffs_p[mask] = torch.round(self.coeffs_p[mask])
            mask = torch.abs(self.coeffs_q - torch.round(self.coeffs_q)) < 0.01
            self.coeffs_q[mask] = torch.round(self.coeffs_q[mask])

    def fit(self, x_input, target, num_epochs=1000, regularization_parameter=0.1, verbose=1,
                 regularization_order: int | float = None):
        # repetitively tries to fit the model until our loss is small enough, or we exceed given number of iterations

        self.fit_once(x_input, target, num_epochs=num_epochs, regularization_parameter=regularization_parameter,
                      verbose=verbose, regularization_order=regularization_order)



    def get_function(self):
        """
            Constructs a symbolic rational function from the stored numerator and denominator coefficients.

            Returns:
                sympy.core.expr.Expr: A simplified SymPy expression representing the rational function.
            """
        _x = sympy.Symbol('x')
        numerator = sum(round(coeff.item(), 4) * _x ** i for i, coeff in enumerate(reversed(self.coeffs_p)))
        # + 1 has to be outside since in the case Q = 1, we have no coeffs so sum will be 0 -> we get nan/zoo
        denominator = sum(round(coeff.item(), 4) * _x ** i for i, coeff in enumerate(reversed(self.coeffs_q)))
        return sympy.sympify(numerator / denominator)


if __name__ == '__main__':
    device = 'cpu'
    target_function_string = f'(2*x**2 + 1)/(x + 1)' # f'(2*x**2 + x + 1)/(x + 1)'
    reg = 2
    success = 0
    for seed in range(100):
        torch.manual_seed(seed)
        model = RationalFunction(2, 1).to(device)
        x_train = torch.linspace(-0.8, 5, 10000).to(device)
        target_function = sympy.lambdify('x', sympy.sympify(target_function_string))
        y_train = target_function(x_train)

        # Train the model
        model.fit(x_train, y_train, num_epochs=4000, regularization_parameter=0.0001, verbose=0,
                                 regularization_order=reg)
        model.eval()
        with torch.no_grad():
            recovered_function = model.get_function()
            lambda_function = sympy.lambdify('x', recovered_function)
            print("Target function     : ", target_function_string)
            print("Recovered function  : ", recovered_function, "Final loss: ", model.losses[-1])

        if model.losses[-1] < 1e-4:
            success += 1

        #data_utility.function_to_plot(target_function, lambda_function, -3, 5)
        # fig, ax = data_utility.get_loss_plot(loss_history)
        # plt.show()
    print("Success: ", success)