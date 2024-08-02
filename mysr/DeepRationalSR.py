import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim

import data_utility


class DeepRationalSR(nn.Module):
    def __init__(self, p_degree, q_degree):
        super(DeepRationalSR, self).__init__()
        self.p_coeff_count = p_degree + 1
        self.q_coeff_count = q_degree  # we standardize the free coefficient of q as 1
        self.input_layer = nn.Linear(1, 1000)
        self.linear_layers = nn.ModuleList([nn.Linear(1000, 1000) for _ in range(10)])
        self.coefficients_layer = nn.Linear(1000, p_degree + q_degree)
        self.coeffs = torch.rand(p_degree + q_degree)

    def forward(self, x):
        c = F.relu(self.input_layer(x))
        for layer in self.linear_layers:
            c = F.relu(layer(c))
        c = F.relu(self.coefficients_layer(c))
        self.coeffs = c
        numerator = sum(coeff * x ** (self.p_coeff_count - i - 1) for i, coeff in enumerate(c[:self.p_coeff_count]))
        denominator = sum(coeff * x ** (self.q_coeff_count - i) for i, coeff in enumerate(c[self.p_coeff_count:])) + 1
        return torch.div(numerator, denominator)

    def get_function(self):
        """
            Constructs a symbolic rational function from the stored numerator and denominator coefficients.

            Returns:
                sympy.core.expr.Expr: A simplified SymPy expression representing the rational function.
            """
        _x = sympy.Symbol('x')
        numerator = sum(coeff * _x ** (self.p_coeff_count - i - 1) for i, coeff in
                        enumerate(self.coefficients_layer[:self.p_coeff_count]))
        # + 1 has to be outside since in the case Q = 1, we have no coeffs so sum will be 0 -> we get nan/zoo
        denominator = sum(coeff * _x ** (self.q_coeff_count - i) for i, coeff in
                          enumerate(self.coefficients_layer[self.q_coeff_count:])) + 1
        return sympy.sympify(numerator / denominator)

    def fit(self, x_input, target, num_epochs=1000, regularization_parameter=0.1, verbose=1,
            criterion=None, optimizer=None, scheduler=None, regularization_order: int | float = None):
        criterion = criterion if criterion else nn.MSELoss()
        optimizer = optimizer if optimizer else optim.Adam(self.parameters(), lr=0.01)
        scheduler = scheduler if scheduler else optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        self.train()
        losses = []
        for epoch in range(num_epochs):
            for x, y in zip(x_input, target):
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = criterion(y_pred, y)
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


if __name__ == '__main__':
    device = 'cpu'
    model = DeepRationalSR(2, 1).to(device)
    x_train = torch.linspace(-3, 3, 101).to(device)
    target_function = sympy.lambdify('x', sympy.sympify(f'(2*x**2 + {torch.pi} * x + 3)/(x+7)'))
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
    data_utility.function_to_plot(target_function, recovered_function, -4, 4)
    fig, ax = data_utility.get_loss_plot(loss_history)
    plt.show()
