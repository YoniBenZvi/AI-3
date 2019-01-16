from hw3_utils import *
import operator


class knn_classifier(abstract_classifier):
    def __init__(self, data, labels, k=5):
        self.k = k
        self.data = data
        self.labels = labels

    def calculate_final_classification(self, k_nearest_neighbours) -> int:
        neighbour_classifications_counter = [0, 0]
        for example in k_nearest_neighbours:
            if self.labels[example[0]] == 0:
                neighbour_classifications_counter[0] = neighbour_classifications_counter[0] + 1
            else:
                neighbour_classifications_counter[1] = neighbour_classifications_counter[1] + 1
        return 0 if neighbour_classifications_counter[0] >= neighbour_classifications_counter[1] else 1

    def classify(self, example_to_classify) -> int:
        neighbours = {}
        i = 0
        for example in self.data:
            neighbours[i] = distance_euclidean(example_to_classify, example)
            i = i + 1
        examples_sorted_by_distance = sorted(neighbours.items(), key=operator.itemgetter(1))
        k_nearest_neighbours = examples_sorted_by_distance[:self.k]
        return self.calculate_final_classification(k_nearest_neighbours)


def distance_euclidean(list1, list2):
    accumulator = 0
    for x, y in zip(list1, list2):
        accumulator = accumulator + (x - y) ** 2
    return accumulator ** .5


class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels) -> knn_classifier:
        return knn_classifier(data, labels, self.k)


def main():
    data = load_data()
    labels = data[1]
    test_set = data[2]
    data = data[0]
    factory = knn_factory(k=5)
    classifier = factory.train(data=data, labels=labels)
    result = classifier.classify(test_set[0])
    print(result)


if __name__ == '__main__':
    main()
