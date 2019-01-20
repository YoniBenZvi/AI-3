from hw3_utils import *
import numpy as np
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


def split_to_pos_and_neg_groups(data) -> (np.ndarray, np.ndarray):
    pos_examples = np.ndarray((0, np.size(data, 1)))
    neg_examples = np.ndarray((0, np.size(data, 1)))
    # print(f'initial shape of pos_examples and neg_examples: {np.shape(pos_examples)},{np.shape(neg_examples)}')
    for example in data:
        if example[-1] == 1:  # recall that the last (rightmost) column is the label
            pos_examples = np.vstack([pos_examples, example])
        else:
            neg_examples = np.vstack([neg_examples, example])

    # the following lines make sure we've used all the examples in the data input variable correctly
    # print(f'final shape of pos_examples and neg_examples: {np.shape(pos_examples)},{np.shape(neg_examples)}')
    assert np.shape(pos_examples)[0] + np.shape(neg_examples)[0] == np.size(data, 0)  # used all examples in data
    assert np.shape(pos_examples)[1] == np.shape(neg_examples)[1] == np.shape(data)[1]  # columns did not get messed up
    #

    return pos_examples, neg_examples


def split_groups_to_folds(pos_examples, neg_examples, num_folds) -> list:
    pos_size = round(len(pos_examples) / num_folds)
    neg_size = round(len(neg_examples) / num_folds)
    # print(pos_size,neg_size)
    output_groups = []
    for i in range(num_folds - 1):
        group_to_append = pos_examples[i * pos_size:(i + 1) * pos_size]
        group_to_append = np.vstack([group_to_append, neg_examples[i * neg_size:(i + 1) * neg_size]])
        assert len(group_to_append) == pos_size + neg_size
        np.random.shuffle(group_to_append)
        output_groups.append(group_to_append)

    last_group_to_append = pos_examples[(num_folds - 1) * pos_size:]
    last_group_to_append = np.vstack([last_group_to_append, neg_examples[(num_folds - 1) * neg_size:]])
    np.random.shuffle(last_group_to_append)
    output_groups.append(last_group_to_append)
    assert len(output_groups) == num_folds
    return output_groups


def export_to_pickle_file(groups: list):
    for i in range(len(groups)):
        path = f'./data/ecg_fold_{i + 1}.data'
        with open(path, 'wb') as file:
            pickle.dump(groups[i], file)


def did_use_all_dataset(groups) -> bool:
    accumulator = 0
    for group in groups:
        accumulator += np.shape(group)[0]
    return accumulator == 1000


def split_crosscheck_groups(dataset: (np.ndarray, list, np.ndarray), num_folds: int = 2):
    data = dataset[0]
    labels = np.asarray(dataset[1])
    labels = np.expand_dims(labels, axis=1)
    data = np.hstack([data, labels])  # adding the labels list as
    # the last (rightmost) column of the data matrix
    np.random.shuffle(data)  # randomly shuffling the order of the examples and their labels
    pos_examples, neg_examples = split_to_pos_and_neg_groups(data)
    groups = split_groups_to_folds(pos_examples, neg_examples, num_folds)
    assert did_use_all_dataset(groups) is True
    export_to_pickle_file(groups)


def load_k_fold_data(i: int) -> (np.ndarray, list):
    path = rf'data/ecg_fold_{i}.data'
    with open(path, 'rb') as file:
        examples = pickle.load(file)
    train_features = examples[:, :-1]
    train_labels = list(examples[:, -1])
    return train_features, train_labels


def evaluate(classifier_factory: abstract_classifier_factory, k: int) -> (float, float):
    num_folds = k
    for fold in range(num_folds):


def main():
    # dataset is a 3-tuple consisting of:
    # (2D ndarray of training features, list of labels,2D ndarray of testing features)
    dataset = load_data()
    data = dataset[0]
    labels = dataset[1]
    test_set = dataset[2]
    # factory = knn_factory(k=5)
    # classifier = factory.train(data=data, labels=labels)
    # result = classifier.classify(test_set[0])
    # print(result)
    # split_crosscheck_groups(dataset, 2)
    # load_k_fold_data(2)

    patients, labels, test = load_data()
    split_crosscheck_groups(patients, labels, 2)
    knn3 = knn_factory(3)
    accuracy, error = evaluate(knn3, 2)


if __name__ == '__main__':
    main()
