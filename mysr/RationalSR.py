import matplotlib.pyplot as plt
import sympy
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

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

    def __init__(self, degree_p, degree_q):
        """
        Initializes the RationalFunction module.

        Parameters:
        num_coeffs_p (int): The number of coefficients in the numerator polynomial P(x).
        num_coeffs_q (int): The number of coefficients in the denominator polynomial Q(x).
        """
        super(RationalFunction, self).__init__()
        self.coeffs_p = nn.Parameter(torch.rand(degree_p + 1))
        self.coeffs_q = nn.Parameter(torch.rand(degree_q))

    def forward(self, x) -> Tensor:
        """
        Forward pass through the RationalFunction module.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after applying the rational function.
        """
        # Compute the numerator P(x) and denominator Q(x)
        numerator = sum(self.coeffs_p[-i - 1] * x ** i for i in range(len(self.coeffs_p)))
        denominator = sum(self.coeffs_q[-i - 1] * x ** (i+1) for i in range(len(self.coeffs_q))) + 1
        return torch.Tensor(numerator / denominator)

    def fit(self, x_input, target, num_epochs=1000, regularization_parameter=0.1, verbose=1,
            criterion=None, optimizer=None, scheduler=None, regularization_order: int | float = None):
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
        criterion = criterion if criterion else nn.MSELoss()
        optimizer = optimizer if optimizer else optim.Adam(self.parameters(), lr=0.01)
        scheduler = scheduler if scheduler else optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        self.train()
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x_input)
            loss = criterion(y_pred, target)
            losses.append(loss.item())
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
                self.coeffs_q[:-1][torch.abs(self.coeffs_q[:-1]) < 0.001] = 0.0

            if verbose == 1 and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.9f}')
                print(self.get_function())
        # round if near to integer (diff less than 0.01)
        with torch.no_grad():
            mask = torch.abs(self.coeffs_p - torch.round(self.coeffs_p)) < 0.01
            self.coeffs_p[mask] = torch.round(self.coeffs_p[mask])
            mask = torch.abs(self.coeffs_q[:-1] - torch.round(self.coeffs_q[:-1])) < 0.01
            self.coeffs_q[:-1][mask] = torch.round(self.coeffs_q[:-1][mask])
        return losses

    def get_function(self):
        """
            Constructs a symbolic rational function from the stored numerator and denominator coefficients.

            Returns:
                sympy.core.expr.Expr: A simplified SymPy expression representing the rational function.
            """
        _x = sympy.Symbol('x')
        numerator = sum(coeff.round(decimals=5) * _x ** i for i, coeff in enumerate(reversed(self.coeffs_p)))
        # + 1 has to be outside since in the case Q = 1, we have no coeffs so sum will be 0 -> we get nan/zoo
        denominator = sum(coeff.round(decimals=5) * _x ** (i+1) for i, coeff in enumerate(reversed(self.coeffs_q))) + 1
        return sympy.sympify(numerator / denominator)


if __name__ == '__main__':
    device = 'cpu'
    model = RationalFunction(2, 1).to(device)
    x_train = torch.linspace(-3, 3, 101).to(device)
    target_function = sympy.lambdify('x', sympy.sympify(f'(2*x**2 + {torch.pi} * x + 3)/(x+1)'))
    y_train = target_function(x_train)

    # Train the model
    loss_history = model.fit(x_train, y_train, num_epochs=1000, regularization_parameter=0.1, regularization_order=None,
                             optimizer=optim.Adam(model.parameters(), lr=0.01))
    model.eval()
    recovered_function = sympy.lambdify('x', model.get_function())
    with torch.no_grad():
        print("Learned coefficients for P(x):", model.coeffs_p)
        print("Learned coefficients for Q(x):", model.coeffs_q)
        print("Recovered function: ", model.get_function())
    data_utility.function_to_plot(target_function, recovered_function)
    fig, ax = data_utility.get_loss_plot(loss_history)
    plt.show()
