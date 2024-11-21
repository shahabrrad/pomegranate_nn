Pomegranate is a library for probabilistic modeling defined by its modular implementation and treatment of all models as the probability distributions they are. The modular implementation allows one to easily drop normal distributions into a mixture model to create a Gaussian mixture model just as easily as dropping a gamma and a Poisson distribution into a mixture model to create a heterogeneous mixture. But that's not all! Because each model is treated as a probability distribution, Bayesian networks can be dropped into a mixture just as easily as a normal distribution, and hidden Markov models can be dropped into Bayes classifiers to make a classifier over sequences. Together, these two design choices enable a flexibility not seen in any other probabilistic modeling package.

This is a modified version of Pomegranate that allows the use of neural network estimation instead of probability tables in the definition of factor nodes in factor graph models. The networks cannot be used for other graphical models and are only available for factor graphs.

### Installation
You can either install the main package and change the files in the installed package, or you can just clone this repository and use it.
To intall the main package, run
`pip install pomegranate`
then use the `pomegranate/factor_graph.py` , `pomegranate/__init__.py`, `pomegranate/distributions/neuralnet.py`, and `pomegranate/distributions/__init__.py` from this repository

### How to
To learn about how pomegranate works generally you can just read the documents for the main package. After that, you can read the `example.py` file to see an example of how it can be used with the neural net as a factor function. I believe it is self explanitory. The order of the num_categories_list and embedding_dim_list should be the same as the the order of adding marginals and connecting them to the factor node. 
