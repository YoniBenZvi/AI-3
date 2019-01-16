from hw3_utils import *
import operator


class knn_classifier(abstract_classifier):
    def __init__(self, k=5):
        self.k = k
        self.data = load_data()

    def calculate_final_classification(self, k_nearest_neighbours):
        neighbour_classifications_counter = [0, 0]
        for example in k_nearest_neighbours:
            if self.data[1][example[0]] == 0:
                neighbour_classifications_counter[0] = neighbour_classifications_counter[0] + 1
            else:
                neighbour_classifications_counter[1] = neighbour_classifications_counter[1] + 1
        return 0 if neighbour_classifications_counter[0] >= neighbour_classifications_counter[1] else 1

    def classify(self, example_to_classify) -> int:
        neighbours = {}
        i = 0
        for row in self.data[0]:
            neighbours[i] = distance_euclidean(example_to_classify, row)
            i = i + 1
        examples_sorted_by_distance = sorted(neighbours.items(), key=operator.itemgetter(1))
        k_nearest_neighbours = examples_sorted_by_distance[:self.k]
        return self.calculate_final_classification(k_nearest_neighbours)


def distance_euclidean(list1, list2):
    accumulator = 0
    for x, y in zip(list1, list2):
        accumulator = accumulator + (x - y) ** 2
    return accumulator ** .5


def main():
    c = knn_classifier()
    result = c.classify(c.data[2][0])
    print(result)


if __name__ == '__main__':
    main()
