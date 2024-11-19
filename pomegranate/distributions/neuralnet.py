# _distribution.py
# Jacob Schreiber <jmschreiber91@gmail.com>

import torch

import torch.nn as nn
import torch.nn.functional as F

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _cast_as_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from ._distribution import Distribution


class CategoricalNN(nn.Module):
    def __init__(
            self,
            num_categories_list,
            embedding_dim_list,
            hidden_dim,
            output_dim):
        """
        Args:
            num_categories_list (list of int): Number of categories for each categorical variable.
            embedding_dim_list (list of int): Dimension of the embedding for each categorical variable.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output layer.
        """
        super(CategoricalNN, self).__init__()

        # Embedding layers for each categorical variable
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in zip(num_categories_list, embedding_dim_list)
        ])

        # FC layers
        total_embedding_dim = sum(embedding_dim_list)
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split input into separate categorical variables
        embedded = [self.embeddings[i](x[:, i])
                    for i in range(len(self.embeddings))]

        # concatenate embeddings
        x = torch.cat(embedded, dim=1)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        output = self.sigmoid(x)
        return output


class NeuralDistribution(Distribution):
    # TODO: define a probs parameter for the class
    """A base distribution object.

    This distribution is inherited by all the other distributions.
    input num_categories_list incluedes the categaries of all variables. The first variable is the output of the neural network
    This is saved into categories attribute of the class. meanwhile the num_categories_list attribute includes only the categories of the output variables
    """

    def __init__(
            self,
            num_categories_list,
            embedding_dim_list,
            hidden_dim,
            output_dim,
            inertia=0.0,
            frozen=False,
            check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self._device = _cast_as_parameter([0.0])

        _check_parameter(inertia, "inertia", min_value=0, max_value=1, ndim=0)
        _check_parameter(frozen, "frozen", value_set=[True, False], ndim=0)
        _check_parameter(check_data, "check_data", value_set=[True, False],
                         ndim=0)

        self.register_buffer("inertia", _cast_as_tensor(inertia))
        self.register_buffer("frozen", _cast_as_tensor(frozen))
        self.register_buffer("check_data", _cast_as_tensor(check_data))

        self._initialized = False

        if (num_categories_list[0] != 2) or (output_dim != 2): # TODO can this be changed to multiple categories if we change the network to be use softmax instead of sigmoid
            raise ValueError("The output variable should have two categories both in output_dim and in number of categories")

        self.num_categories_list = num_categories_list[1:]
        self.categories = num_categories_list
        self.embedding_dim_list = embedding_dim_list
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        

        self.model = CategoricalNN(
            self.num_categories_list,
            self.embedding_dim_list,
            hidden_dim,
            output_dim)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except BaseException:
            return 'cpu'

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def freeze(self):
        self.register_buffer("frozen", _cast_as_tensor(True))
        return self

    def unfreeze(self):
        self.register_buffer("frozen", _cast_as_tensor(False))
        return self

    def forward(self, X):
        self.summarize(X)
        return self.log_probability(X)

    def backward(self, X):
        self.from_summaries()
        return X

    def _initialize(self, d):
        self.d = d
        self._reset_cache()

    def _reset_cache(self):  # Replaced this with just redefininh the model
        if not self._initialized:
            return

        self.model = CategoricalNN(
            self.num_categories_list,
            self.embedding_dim_list,
            self.hidden_dim,
            self.output_dim)


    def probability(self, X):
        """returns the probability fo X happening.
        X is the array of all variables connected to the factor, including the 0 index
        """

        model_input = torch.tensor([X[1:]])
        expected_output = torch.tensor(X[0])

        output = self.model(model_input).item()
        if expected_output == 1:
            return output
        else:
            return 1 - output


    # TODO - This funciton can be just the log of probability function. left unimplemented for now because it is not used
    def log_probability(self, X):
        raise NotImplementedError

    def fit(self, X, sample_weight=None):
        self.summarize(X, sample_weight=sample_weight)
        self.from_summaries()
        return self

    def summarize(self, X, sample_weight=None, num_epochs=100):
        """Instead of extracting the sufficient statistics, we will train the model on the data."""
        if not self._initialized:
            self._initialize(len(X[0]))

        # X = _cast_as_tensor(X)
        # _check_parameter(X, "X", ndim=2, shape=(-1, self.d),
        #                  check_parameter=self.check_data)

        X_train = torch.stack([i[1:] for i in X[:-100]], dim=0)
        Y_train = torch.stack([i[0] for i in X[:-100]], dim=0)

        X_val = torch.stack([i[1:] for i in X[-100:]], dim=0)
        Y_val = torch.stack([i[0] for i in X[-100:]], dim=0)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), Y_train.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), Y_val.float())
                    predicted = (val_outputs.squeeze() > 0.5).float()
                    accuracy = (predicted == Y_val.float()
                                ).sum().item() / Y_val.size(0)
                    print(f'Validation Accuracy: {accuracy:.4f}')

                    print(f'Validation Loss: {val_loss.item():.4f}')

        return X, sample_weight

    def from_summaries(self):   # We dont need this when replacing summary statistics with neural network
        return
        # raise NotImplementedError


class ConditionalDistribution(Distribution):
    def __init__(self, inertia, frozen, check_data):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)

    def marginal(self, dim):
        raise NotImplementedError
