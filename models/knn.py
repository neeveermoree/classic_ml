import numpy as np


class KNN:
    def __init__(self, neighbors_num, p=2, normalize_features=True):
        assert neighbors_num > 1
        self._neighbors_num = neighbors_num
        self._p = p
        self._normalize_features = normalize_features


    def fit(self, features, target):
        # 0 - 1 normalization
        if self._normalize_features:
            self._min_value = np.min(features, axis=0)
            features = features - self._min_value
            self._max_value = np.max(features, axis=0)
            features = features / self._max_value

        self._features = features
        self._target = target
    

    def _get_top_neighbors(self, features):
        if self._normalize_features:
            features -= self._min_value
            features = features / self._max_value
        
        distances = self._minkowski_distance(features)

        # Sort training examples by distance
        sorted_args = np.argsort(distances, axis=1)

        # Select top neighbors
        top_neighbors = sorted_args[:, :self._neighbors_num]

        # Select top distance
        top_distances = np.sort(distances, axis=1)[:, :self._neighbors_num]
        return top_neighbors, top_distances
    

    def _minkowski_distance(self, features):
        # Same result as sklearn.metrics.pairwise_distance
        # Row-wise matrix subtraction
        subtraction = features[:, np.newaxis] - self._features
        absolute = np.abs(subtraction)
        power_p = absolute ** self._p
        row_sum = np.sum(power_p, axis=2)
        distance = row_sum ** (1/self._p)
        return distance


    def predict(self, features):
        raise NotImplementedError


class KNNClassifier(KNN):
    def __init__(self, neighbors_num=3, p=2, normalize_features=True, weighted=True):
        super().__init__(neighbors_num, p, normalize_features)
        self._weighted = weighted
    

    def __most_occurent_class(self, top_classes):
        occurence_count = np.bincount(top_classes)
        most_frequent_class_id = np.argmax(occurence_count)
        return most_frequent_class_id


    def __weighted_prediction(self, targets, distance):
        unique_classes, counts = np.unique(targets, return_counts=True)



    def predict(self, features):
        top_neighbors, distance = self._get_top_neighbors(features)
        targets = self._target[top_neighbors]
        print(targets)
        print(distance)
        if self._weighted:
            pass
        else:
            prediction = np.apply_along_axis(self.__most_occurent_class, axis=1, arr=targets)
        return prediction



a = np.array([
    [1, 2, 3],
    [4, 5, 7],
    [1, 0, 3],
    [12, 12, 10]
])


b = np.array([
    [1, 2, 3],
    [10, 10, 12]
])

knn = KNNClassifier()
knn.fit(a, np.array([1,2, 1,2]))
print(knn.predict(b))
