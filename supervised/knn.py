from typing import List, Union
import math
from heapq import heappush, heappop
import statistics
import random


class Point:
    def __init__(self, coordinate: Union[List, tuple]):
        self.coordinate = coordinate

    def euclidean_distance(self, other: "Point"):
        assert len(self.coordinate) == len(
            other.coordinate
        ), f"Different dimension {len(self.coordinate)} and {len(other.coordinate)}"

        differences = []
        for x1, x2 in list(zip(self.coordinate, other.coordinate)):
            differences.append(x2 - x1)

        differences = list(map(lambda d: d ** 2, differences))
        return math.sqrt(sum(differences))


class KNearestNeighbors:
    def __init__(self, k: int = 3):
        self.k = k
        self._fit_data = []

    def fit(self, x: Union[tuple, List], y: Union[tuple, List]):
        """Train KNN model with x data"""
        self._fit_data = [
            (Point(coordinates), label) for coordinates, label in zip(x, y)
        ]

    def predict(self, x: Union[List, tuple]):
        predicts = []
        for coordinates in x:
            predict_point = Point(coordinates)

            distances = []
            for data_point, data_label in self._fit_data:
                heappush(
                    distances,
                    (predict_point.euclidean_distance(data_point), data_label),
                )

            predicts.append(
                statistics.mode([heappop(distances)[1] for _ in range(self.k)])
            )
        return predicts


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


# if __name__ == "__main__":
#     generator = PointGenerator(n=10, dimension=3)
#     k = 2
#     # fit training data
#     train_x = generator.generate_points(-1000, 1000)
#     train_y = generator.generate_labels(k=k)
#     print(train_x, train_y)

#     # Define KNN model
#     knn = KNearestNeighbors(k=k)
#     knn.fit(train_x, train_y)

#     # Test model
#     test_x = generator.generate_points(-1000, 1000)
#     print(test_x, knn.predict(test_x))
