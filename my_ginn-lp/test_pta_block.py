import concurrent.futures

import pandas as pd
import sympy as sp
import torch

from pta_module import PTABlock, PTABlockWithLog


def train_model(model, inputs_x, y, num_epochs=5000, lr=0.01):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_threshold = 1e-5
    first_zero_epoch = None

    for epoch in range(num_epochs):
        model.train()
        predictions = model(inputs_x)
        loss = criterion(predictions, y)
        model.losses.append(loss.item())

        # Check if the loss has gone below the threshold for the first time
        if first_zero_epoch is None and loss.item() <= loss_threshold:
            first_zero_epoch = epoch + 1  # Store the epoch (1-indexed)

        optimizer.zero_grad()  # Zero out the gradients from the previous step
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update the parameters

        if (epoch + 1) % 100 == 0 or epoch == 0:
            if epoch == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print(f"Learned exponents for {model.__class__.__name__}:", model.exponents.detach().numpy())
    print(f"Final loss for {model.__class__.__name__}: {loss.item():.4f}")

    if first_zero_epoch is not None:
        print(
            f"Loss reached zero (or below threshold {loss_threshold}) for the first time at epoch {first_zero_epoch}.")
        return first_zero_epoch
    else:
        print("Loss did not reach zero during training.")
        return 10e10


def train_and_evaluate(formula_str, num_samples, seed):
    torch.manual_seed(seed)

    x1, x2 = sp.symbols('x1 x2')
    formula = sp.sympify(formula_str)  # Convert string to sympy expression

    x1_vals = torch.rand(num_samples, 1) * 5
    x2_vals = torch.rand(num_samples, 1) * 5

    formula_func = sp.lambdify((x1, x2), formula, 'numpy')
    y_vals = torch.tensor(formula_func(x1_vals.numpy(), x2_vals.numpy()), dtype=torch.float32)

    inputs_x = torch.cat([x1_vals, x2_vals], dim=1)

    # Initialize models
    model_pta = PTABlock(num_inputs=2, block_number=1)
    model_pta_log = PTABlockWithLog(num_inputs=2, block_number=1)

    with torch.no_grad():
        learned_exponents = model_pta.exponents.clone()
        model_pta_log.ln_dense.weight = torch.nn.Parameter(learned_exponents.view_as(model_pta_log.ln_dense.weight))

    torch.manual_seed(seed)
    print(f"Training PTABlock for formula: {formula_str}, seed: {seed}")
    res1 = train_model(model_pta, inputs_x, y_vals)

    torch.manual_seed(seed)
    print(f"\nTraining PTABlockWithLog for formula: {formula_str}, seed: {seed}")
    res2 = train_model(model_pta_log, inputs_x, y_vals)

    if res1 < res2:
        winner = "PTA"
    elif res1 > res2:
        winner = "PTA with log"
    else:
        winner = "Fail"

    return [winner, res1, res2]

def train_and_evaluate_parallel(t, num_samples=1000):
    formula, seed = t
    return [formula, seed] + train_and_evaluate(formula, num_samples, seed)

if __name__ == '__main__':
    formulas = ["x1**2 * x2", "x1 / x2", "x1**3/x2"]  # Example formulas
    seeds = range(0,1000)
    columns = ['Target Formula', 'Seed', 'Winner', 'Last Epoch PTA', 'Last Epoch PTA with Log']

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(train_and_evaluate_parallel, [(f,s) for f in formulas for s in seeds]))

    result_table = pd.DataFrame(results, columns=columns)
    result_table.to_csv("pta_block.csv", index=False)