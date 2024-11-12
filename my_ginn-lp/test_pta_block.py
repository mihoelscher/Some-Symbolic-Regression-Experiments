import torch
from matplotlib import pyplot as plt

from experiments.regularization_experiment import plot_multiple_losses
from pta_module import PTABlock, PTABlockWithLog

def train_model(model, inputs_x, y, num_epochs=5000, lr=0.001):
    # Define a mean squared error loss function
    criterion = torch.nn.MSELoss()

    # Use an optimizer (e.g., Adam) to update model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Forward pass: Compute the model's prediction
        predictions = model(inputs_x)

        # Compute the loss
        loss = criterion(predictions, y)
        model.losses.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update the parameters

        # Print the loss every 100 epochs for monitoring
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Print final results after training
    print(f"Learned exponents for {model.__class__.__name__}:", model.exponents.detach().numpy())
    print(f"Final loss for {model.__class__.__name__}: {loss.item():.4f}")

if __name__ == '__main__':
    # Generate random training data
    num_samples = 1000
    x1 = torch.rand(num_samples, 1) * 5  # Random values for x1
    x2 = torch.rand(num_samples, 1) * 5  # Random values for x2

    # The target output is x1 * x2^2
    y = x1 / x2

    # Combine x1 and x2 into a single input tensor
    inputs_x = torch.cat([x1, x2], dim=1)  # Shape: (num_samples, 2)

    # Initialize the models
    #torch.manual_seed(1)
    model_pta = PTABlock(num_inputs=2, block_number=1)
    #torch.manual_seed(1)
    model_pta_log = PTABlockWithLog(num_inputs=2, block_number=1)

    print("Training PTABlock...")
    train_model(model_pta, inputs_x, y)

    print("\nTraining PTABlockWithLog...")
    train_model(model_pta_log, inputs_x, y)

    # Plot the losses for both models
    plt.figure(figsize=(10, 6))
    plt.plot(model_pta.losses, label='PTABlock Loss')
    plt.plot(model_pta_log.losses, label='PTABlockWithLog Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.show()
