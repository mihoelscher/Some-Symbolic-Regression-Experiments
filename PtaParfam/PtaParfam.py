import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from PtaGrowthModule import PtaGrowth


class PtaParfam(torch.nn.Module):
    input_dim = None
    # outer_pta_growth_block
    # inner_pta_growth_blocks_dict = None
    functions = [lambda x: x]
    function_names = ['id']
    losses = []

    def __init__(self, functions=None, max_number_of_growth_blocks=3, max_repetitions=3, *args, **kwargs):
        super(PtaParfam, self).__init__(*args, **kwargs)
        if functions:
            self.functions += functions
            self.function_names += [function.__name__ for function in functions]
        self.max_number_of_growth_blocks = max_number_of_growth_blocks
        self.max_repetitions = max_repetitions

    def _init_pta_blocks(self):
        self.outer_pta_growth_block = PtaGrowth(len(self.functions), 1)
        self.inner_pta_growth_blocks_dict = nn.ModuleDict(
            {function_name: PtaGrowth(self.input_dim, self.max_number_of_growth_blocks) for function_name in
             self.function_names})

    def forward(self, input_x):
        function_outputs = []
        for i, function in enumerate(self.functions):
            inner_pta_output = self.inner_pta_growth_blocks_dict[self.function_names[i]](input_x)
            function_outputs.append(function(inner_pta_output))
        return self.outer_pta_growth_block(torch.cat(function_outputs, dim=1))

    def fit(self, X, y, num_epochs=1000, lr=0.001, batch_size=1024):
        self.input_dim = X.shape[1]
        self._init_pta_blocks()
        y = y.unsqueeze(1)

        # Create a dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.5)
        loss_threshold = 1e-5
        first_zero_epoch = None

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0  # Track loss per epoch

            for batch_X, batch_y in dataloader:
                predictions = self(batch_X)  # Forward pass
                loss = criterion(predictions, batch_y)
                epoch_loss += loss.item()

                optimizer.zero_grad()  # Zero gradients
                loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters

            self.losses.append(epoch_loss / len(dataloader))  # Average loss over batches

            self.prune_exponents()
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {self.losses[-1]:.4f}")

            # Check if the loss has gone below the threshold for the first time
            if first_zero_epoch is None and epoch_loss / len(dataloader) <= loss_threshold:
                first_zero_epoch = epoch + 1  # Store the epoch (1-indexed)

    def to_formula(self):
        if not self.input_dim:
            print('Model not yet fitted to the data')

        inner_formulas = []
        for i, function in enumerate(self.functions):
            inner_formulas.append(
                f'{self.function_names[i]}({self.inner_pta_growth_blocks_dict[self.function_names[i]].to_formula()})')

        return self.outer_pta_growth_block.to_formula(inner_formulas).replace('id','')

    def prune_exponents(self):
        for pta_growth_block in self.inner_pta_growth_blocks_dict.values():
            pta_growth_block.prune_exponents()
        self.outer_pta_growth_block.prune_exponents()


if __name__ == '__main__':
    df = pd.read_csv('I.8.14', sep=" ", header=None, nrows=10000).iloc[:,:-1]
    train_x = torch.tensor(df.iloc[:, :-1].values.astype(float))
    train_y = torch.tensor(df.iloc[:, -1].values)
    pp = PtaParfam(functions=[torch.sqrt])
    pp.fit(train_x, train_y)
    print(pp.to_formula())
