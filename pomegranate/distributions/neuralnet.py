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
    def __init__(self, num_categories_list, embedding_dim_list, hidden_dim, output_dim):
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
        
        # Fully connected layers
        total_embedding_dim = sum(embedding_dim_list)
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split input into separate categorical variables
        embedded = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]
        
        # Concatenate all embedded representations
        x = torch.cat(embedded, dim=1)
        
        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax to output (for classification)
        output = self.sigmoid(x)
        return output


class NeuralDistribution(Distribution):
	"""A base distribution object.

	This distribution is inherited by all the other distributions.
	"""

	def __init__(self,num_categories_list, embedding_dim_list, hidden_dim, output_dim, inertia, frozen, check_data):
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
		
		self.model = CategoricalNN(num_categories_list, embedding_dim_list, hidden_dim, output_dim)

	@property
	def device(self):
		try:
			return next(self.parameters()).device
		except:
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

	def _reset_cache(self):  # TODO - check this
		raise NotImplementedError

	def probability(self, X):
		return torch.exp(self.log_probability(X))

	def log_probability(self, X):   # TODO - this will probably be an inferense step on the model
		raise NotImplementedError

	def fit(self, X, sample_weight=None):
		self.summarize(X, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weight=None):
		if not self._initialized:
			self._initialize(len(X[0]))

		X = _cast_as_tensor(X)
		_check_parameter(X, "X", ndim=2, shape=(-1, self.d), 
			check_parameter=self.check_data)

		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight), 
			device=self.device)

		return X, sample_weight

	def from_summaries(self):   # TODO - this we be the training step on the model
		raise NotImplementedError


class ConditionalDistribution(Distribution):
	def __init__(self, inertia, frozen, check_data):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)

	def marginal(self, dim):
		raise NotImplementedError