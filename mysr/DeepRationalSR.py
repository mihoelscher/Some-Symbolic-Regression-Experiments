import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim

import data_utility


class DeepRationalSR(nn.Module):
    def __init__(self, p_degree, q_degree, num_hidden_layers=3, hidden_size=101):
        super(DeepRationalSR, self).__init__()
        self.coeffs = None
        self.p_coeff_count = p_degree + 1
        self.q_coeff_count = q_degree  # we standardize the free coefficient of q as 1
        self.hidden_size = hidden_size

        layers = [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]  # input layer
        for _ in range(num_hidden_layers - 1):  # Add hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_size, self.p_coeff_count + self.q_coeff_count))  # output layer

        # Initialize the weights using Xavier uniform initialization
        #for layer in layers:
        #    if isinstance(layer, nn.Linear):
        #        nn.init.xavier_normal_(layer.weight)
        #        nn.init.zeros_(layer.bias)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        dummy = torch.randn(101)
        self.coeffs = self.network(dummy).squeeze()
        numerator = sum(
            coeff * x ** (self.p_coeff_count - i - 1) for i, coeff in enumerate(self.coeffs[:self.p_coeff_count]))
        denominator = sum(
            coeff * x ** (self.q_coeff_count - i) for i, coeff in enumerate(self.coeffs[self.p_coeff_count:])) + 1
        return torch.div(numerator, denominator)

    def get_function(self):
        """
            Constructs a symbolic rational function from the stored numerator and denominator coefficients.

            Returns:
                sympy.core.expr.Expr: A simplified SymPy expression representing the rational function.
            """
        _x = sympy.Symbol('x')
        numerator = sum(coeff * (_x ** (self.p_coeff_count - i - 1)) for i, coeff in
                        enumerate(self.coeffs[:self.p_coeff_count]))
        # + 1 has to be outside since in the case Q = 1, we have no coeffs so sum will be 0 -> we get nan/zoo
        denominator = sum(coeff * _x ** (self.q_coeff_count - i) for i, coeff in
                          enumerate(self.coeffs[self.p_coeff_count:])) + 1
        return sympy.sympify(numerator / denominator)

    def fit(self, x_input, target, num_epochs=1000, regularization_parameter=0.1, verbose=1,
            criterion=None, optimizer=None, scheduler=None, regularization_order: int | float = None):
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
                self.coeffs[torch.abs(self.coeffs) < 0.001] = 0.0

            if verbose == 1 and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.9f}')
                print(self.get_function())

            if loss.item() < 1:
                if verbose == 1:
                    print(f'Finished after {epoch + 1} epochs, loss: {loss.item():.9f}')
                break
        # round if near to integer (diff less than 0.01)
        with torch.no_grad():
            mask = torch.abs(self.coeffs - torch.round(self.coeffs)) < 0.01
            self.coeffs[mask] = torch.round(self.coeffs[mask])
        return losses



if __name__ == '__main__':
    device = 'cpu'
    torch.manual_seed(42)
    model = DeepRationalSR(2, 1).to(device)
    x_train = torch.linspace(-0.8, 5, 101).to(device).unsqueeze(1)
    target_function = sympy.lambdify('x', sympy.sympify(f'(2*x**2 + 3)/(x+1)'))
    y_train = target_function(x_train)

    # Train the model
    loss_history = model.fit(x_train, y_train, num_epochs=1000, regularization_parameter=1, regularization_order=1,
                             optimizer=optim.Adam(model.parameters(), lr=0.01))
    model.eval()
    recovered_function = sympy.lambdify('x', model.get_function())
    with torch.no_grad():
        print("Recovered function: ", model.get_function())
    data_utility.function_to_plot(target_function, recovered_function, -4, 4)
    # fig, ax = data_utility.get_loss_plot(loss_history)
    plt.show()
