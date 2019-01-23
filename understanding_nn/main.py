import math
import numpy as np
import random
from typing import Callable, List, Tuple


# Sample training sets
train_logical_and = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 0)]
train_logical_or = [(1, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)]


class Perceptron:
    """
    A binary perceptron is a type of rudamentary "neural network" with only a single node.
    """

    def __init__(self, threshold: Callable):
        # Initialize random weights for now
        self.w1 = random.random()
        self.w2 = random.random()
        self.wb = random.random()

        self.threshold = threshold

        # Technically this isn't required for a binary perceptron
        self.learning_rate = 0.01
        self.bias = 1

    def guess(self, x1: float, x2: float):
        """
        Perform a prediction with the current weights.
        """
        return self.threshold((self.w1 * x1) + (self.w2 * x2) + (self.wb * self.bias))

    def learn(self, x1: float, x2: float, expect: float):
        """
        Correct the weights based on the error
        """
        error = expect - self.guess(x1, x2)

        # Adjust the weights
        self.w1 += self.learning_rate * error * x1
        self.w2 += self.learning_rate * error * x2
        self.wb += self.learning_rate * error * self.bias

    def train(self, data: List[Tuple], epochs: int = 3):
        """
        Run the learn function over a collection of training data.

        data = [(x1, x2, expect), ...]
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

        data = [(x1, x2, expect), ...]
        """
        return sum((expect - self.guess(x1, x2)) for (x1, x2, expect) in data) / len(
            data
        )

    def status(self):
        """
        Convenience function for printing some internal values
        """
        print(
            f"W1: {self.w1}",
            f"W2: {self.w2}",
            f"Wb: {self.wb}",
            f"Activation: {self.threshold.__name__}",
            sep="\n",
        )


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def heaviside(x: float) -> int:
    return int(x >= 0)
