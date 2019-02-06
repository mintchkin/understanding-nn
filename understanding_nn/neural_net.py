import numpy as np
from typing import Tuple


class FeedForward:
    """
    A feed forward neural network consists of multiple neuron 'layers'
    in which the output of one layer feeds to the input of the next layer

    Weights from one layer to the next are represented as matrices.
    """

    def __init__(self, *shape: Tuple[int]):
        # Matrix shapes are (r, c), where
        #     r = number of nodes in a layer
        #     c = number of inputs to that layer
        shapes = zip(shape[1:], shape)

        # We need to introduce an extra input in each layer for the bias
        shapes = ((r, c + 1) for r, c in shapes)

        self.layer_weights = [np.random.random(s) for s in shapes]

        # Arbitrary learning rate and bias constants
        self.learning_rate = 0.01
        self.bias = 1

    def guess(self, inputs: Tuple[float]):
        """
        The feed-forward algorithm!
        """
        data = np.array(inputs)
        for layer in self.layer_weights:
            # np.matmul(m, v) magically converts the vector argument to a matrix
            # with the appropriate shape
            data = np.matmul(layer, np.append(data, self.bias))

        return data
