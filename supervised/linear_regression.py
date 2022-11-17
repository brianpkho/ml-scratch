from typing import Union, List


class LinearRegression:
    def __init__(self, weights: Union[tuple, List], x: Union[tuple, List]):
        self.weights = weights
        self.x = x
