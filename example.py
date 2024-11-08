import torch

import numpy as np

from pomegranate.distributions import Categorical
from pomegranate.distributions import JointCategorical
from pomegranate.factor_graph import FactorGraph
from pomegranate.distributions import NeuralDistribution

r1 = Categorical([[0.5, 0.5]])
r2 = Categorical([[0.5, 0.5]])
r3 = Categorical([[0.5, 0.5]])
r4 = Categorical([[0.5, 0.5]])
r5 = Categorical([[0.5, 0.5]])
r6 = Categorical([[0.5, 0.5]])

def random_normalized_array(shape):
    # array = np.random.rand(*shape)
    # changing this to ones 
    array = np.ones(shape)
    return array / np.sum(array)


num_categories_list = [2,2,2,2,2]
embedding_dim_list = [2,2,2,2,2]  # Specify embedding dimensions for each variable
# num_categories_list = [2,2,2]
# embedding_dim_list = [2,2,2]  
hidden_dim = 64
output_dim = 1

f1 = JointCategorical(random_normalized_array((2,2,2,2,2,2))) #,2,2,2)))
# f1 = NeuralDistribution(num_categories_list, embedding_dim_list, hidden_dim, output_dim)

model = FactorGraph()

model.add_factor(f1)

model.add_marginal(r1)
model.add_marginal(r2)
model.add_marginal(r3)
model.add_marginal(r4)
model.add_marginal(r5)
model.add_marginal(r6)

model.add_edge(r1, f1)
model.add_edge(r2, f1)
model.add_edge(r3, f1)
model.add_edge(r4, f1)
model.add_edge(r5, f1)
model.add_edge(r6, f1)

data = np.random.randint(2, size=(1000,6))
# data = torch.tensor(data, dtype=torch.int32)
# print(data)
# print(data)
model.fit(data)


X_torch = torch.tensor([[0,1,0,1,0,1]]) #, [0,1,0,1,0,1], [0,1,0,1,0,1]])
mask = torch.tensor([([False,]*1)+([True,]*5)]) #, ([False,]*1)+([True,]*5), ([False,]*1)+([True,]*5)])
# X_torch = torch.tensor([[0,1,0]])
# mask = torch.tensor([([False,]*1)+([True,]*2)])
# print(mask)
X_masked = torch.masked.MaskedTensor(X_torch, mask=mask)
print("X_masked", X_masked)
print("pred" , model.predict(X_masked))
# print("prob" , model.predict_proba(X_masked))