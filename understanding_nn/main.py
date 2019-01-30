import math
import numpy as np
from random import random
from typing import Callable, List, Tuple
from itertools import product


# Sample training sets
train_logical_and = [((1, 1), 1), ((1, 0), 0), ((0, 1), 0), ((0, 0), 0)]
train_logical_or = [((1, 1), 1), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0)]

train_middle_3 = [(x, x[1]) for x in product((0, 1), (0, 1), (0, 1))]
train_odd_20 = [(x, sum(x) % 2) for x in product(*((0, 1) for x in range(20)))]


class Perceptron:
    """
    A binary perceptron is a type of rudamentary "neural network" with only a single node.
    """

    def __init__(self, inputs: int, threshold: Callable):
        # Initialize random weights for now
        self.weights = [random() for x in range(inputs)]
        self.bias_weight = random()

        # Technically this isn't required for a binary perceptron
        self.learning_rate = 0.01

        self.threshold = threshold
        self.bias = 1

    def guess(self, xs: Tuple[float]):
        """
        Perform a prediction with the current weights.
        """
        if len(xs) != len(self.weights):
            raise IndexError(f"Invalid inputs size: {len(xs)} != {len(self.weights)}")

        return self.threshold(
            sum(x * w for x, w in zip(xs, self.weights)) + self.bias * self.bias_weight
        )

    def learn(self, xs: Tuple[float], expect: float):
        """
        Correct the weights based on the error
        """
        error = expect - self.guess(xs)

        # Adjust the weights
        for i, x in enumerate(xs):
            self.weights[i] += self.learning_rate * error * x

        self.bias_weight += self.learning_rate * error * self.bias

    def train(self, data: List[Tuple], epochs: int = 3):
        """
        Run the learn function over a collection of training data.

        data = [((x1, x2, ...), expect), ...]
        """
        print(f"Initial Error: {self.test(data)}")
        for x in range(epochs):
            print(f"Epoch {x+1}/{epochs}: ", end="")
            for d in data:
                self.learn(*d)
            print(f"complete, error: {self.test(data)}")

    def test(self, data: List[Tuple]):
        """
        Return the average error over a list of test data

        data = [((x1, x2, ...), expect), ...]
        """
        return sum((expect - self.guess(xs)) for (xs, expect) in data) / len(data)

    def status(self):
        """
        Convenience function for printing some internal values
        """
        print(
            f"Activation: {self.threshold.__name__}",
            f"Wbias: {self.bias_weight}",
            f"Weights: {self.weights}",
            sep="\n",
        )


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def heaviside(x: float) -> int:
    return int(x >= 0)
