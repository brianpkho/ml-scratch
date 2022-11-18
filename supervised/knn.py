from typing import List, Union
import math
from heapq import heappush, heappop
import statistics
import random


class KNearestNeighbors:
    def __init__(self, k: int = 3):
        self.k = k
        self._fit_data = []

    def fit(self, x: Union[tuple, List], y: Union[tuple, List]):
        """Train KNN model with x data"""
        self._fit_data = [(coordinates, label) for coordinates, label in zip(x, y)]

    def predict(self, x: Union[List, tuple]):
        predicts = []
        for predict_point in x:

            distances = []
            for data_point, data_label in self._fit_data:
                heappush(
                    distances,
                    (self.euclidean_distance(predict_point, data_point), data_label),
                )

            predicts.append(
                statistics.mode([heappop(distances)[1] for _ in range(self.k)])
            )
        return predicts

    def euclidean_distance(self, p1: Union[List, tuple], p2: Union[List, tuple]):
        assert len(p1) == len(p2), f"Different dimension {len(p1)} and {len(p2)}"
        differences = []
        for x1, x2 in list(zip(p1, p2)):
            differences.append(x2 - x1)

        differences = list(map(lambda d: d ** 2, differences))
        return math.sqrt(sum(differences))


class PointGenerator:
    def __init__(self, n: int, dimension: int = 2):
        self._dimension = dimension
        self.n = n
        random.seed(42)

    def nd_point(self, lower: int, upper: int):
        return [random.randint(lower, upper) for _ in range(self._dimension)]

    def generate_points(self, lower: int, upper: int):
        return [self.nd_point(lower, upper) for _ in range(self.n)]

    def generate_labels(self, k):
        return [random.randint(1, k) for _ in range(self.n)]


class ClassifierTester:
    @staticmethod
    def accuracy(y_true, y_pred):
        assert len(y_true) == len(
            y_pred
        ), f"Labelled predictions dimension must be the same; instead got {len(y_true)}, {len(y_pred)}"
        accuracy = 0
        for ytrue_i, ypred_i in zip(y_true, y_pred):
            accuracy += ytrue_i == ypred_i

        return accuracy / len(y_true)


if __name__ == "__main__":
    generator = PointGenerator(n=10, dimension=3)
    k = 2
    # fit training data
    train_x = generator.generate_points(-1000, 1000)
    train_y = generator.generate_labels(k=k)
    print(train_x, train_y)

    # Define KNN model
    knn = KNearestNeighbors(k=k)
    knn.fit(train_x, train_y)

    # Test model
    train_x = generator.generate_points(-1000, 1000)
    predicted_y = knn.predict(train_x)
    print(predicted_y)

    # TestClassifier
    print(f"Accuracy: {ClassifierTester.accuracy(y_true=train_y, y_pred=predicted_y)}")
