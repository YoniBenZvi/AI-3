from hw3_utils import *
import numpy as np
import operator
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB


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
        # examples_sorted_by_distance = sorted(neighbours.items(), key=operator.itemgetter(1))
        examples_sorted_by_distance = sorted(neighbours.items(), key=lambda tup: tup[1])
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
    """
    Debugging purposes only
    """
    accumulator = 0
    for group in groups:
        accumulator += np.shape(group)[0]
    return accumulator == 1000


def split_crosscheck_groups(dataset: (np.ndarray, list, np.ndarray), num_folds: int = 2, is_testing=False):
    data = dataset[0]
    labels = np.asarray(dataset[1])
    labels = np.expand_dims(labels, axis=1)
    data = np.hstack([data, labels])  # adding the labels list as
    # the last (rightmost) column of the data matrix
    # TODO: assert it shuffles rows and not columns
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


def calculate_evaluation_accuracy_error(data, factory) -> (float, float):
    error_count = 0
    experiment_count = 0
    for i in range(len(data)):
        training_set = None
        test_set = data[i]
        for j in range(len(data)):
            if i == j:
                continue
            else:
                if training_set is None:
                    training_set = (data[j][0], data[j][1])
                else:
                    np.vstack((training_set[0], data[j][0]))
                    np.vstack((training_set[1], data[j][1]))
        classifier = factory.train(training_set[0], training_set[1])
        if isinstance(factory, knn_factory):
            for example, n in zip(test_set[0], range(len(test_set[0]))):
                experiment_count += 1
                if classifier.classify(example) != test_set[1][n]:
                    error_count += 1
        else:
            classifier_labels = classifier.classify(test_set[0])
            for label, n in zip(classifier_labels, range(len(classifier_labels))):
                experiment_count += 1
                if label != test_set[1][n]:
                    error_count += 1
    avg_error = error_count / experiment_count
    return 1 - avg_error, avg_error


def evaluate(classifier_factory: abstract_classifier_factory, k: int) -> (float, float):
    num_folds = k
    data = {}
    for i in range(1, num_folds + 1):
        data[i - 1] = load_k_fold_data(i)
    # data is now a dictionary of the form {i:(features, labels)} where i is the key
    # and (features, labels) is the value, which is a tuple of the features matrix
    # and the labels vector
    return calculate_evaluation_accuracy_error(data, classifier_factory)


def export_to_csv(results, filename):
    with open(filename, 'w', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        for row in results:
            wr.writerow(list(row))


def question5():
    k_list = [1, 3, 5, 7, 13]
    results = []
    for k in k_list:
        print(f'k={k}')
        factory = knn_factory(k)
        accuracy, error = evaluate(factory, 2)
        results.append((k, accuracy, error))
    export_to_csv(results, 'experiments6.csv')


class ID3_classifier(abstract_classifier):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        This constructor also trains the classifier by creating the decision tree.
        :param data: the unlabeled data of the dataset
        :param labels: the correct labels of the data
        """

        self.inner_classifier = DecisionTreeClassifier(criterion="entropy")
        self.inner_classifier.fit(data, labels)

    def classify(self, features) -> int:
        return self.inner_classifier.predict(features)


class ID3_factory(abstract_classifier_factory):
    def train(self, data, labels) -> ID3_classifier:
        return ID3_classifier(data, labels)


class MultinomialNB_classifier(abstract_classifier):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        This constructor also trains the classifier by creating the decision tree.
        :param data: the unlabeled data of the dataset
        :param labels: the correct labels of the data
        """

        self.inner_classifier = MultinomialNB()
        self.inner_classifier.fit(data, labels)

    def classify(self, features) -> int:
        return self.inner_classifier.predict(features)


class MultinomialNB_factory(abstract_classifier_factory):
    def train(self, data, labels) -> MultinomialNB_classifier:
        return MultinomialNB_classifier(data, labels)


class Contest_classifier(abstract_classifier):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        This constructor also trains the classifier by creating the decision tree.
        :param data: the unlabeled data of the dataset
        :param labels: the correct labels of the data
        """
        self.inner_factories = [perceptron_factory(), ID3_factory(), knn_factory(1), knn_factory(5), knn_factory(13)]
        self.inner_classifiers = []
        for factory in self.inner_factories:
            self.inner_classifiers.append(factory.train(data, labels))
        for c in self.inner_classifiers:
            if not isinstance(c, knn_classifier):
                c.inner_classifier.fit(data, labels)

    def classify(self, features) -> int:
        classification_votes = []
        for c in self.inner_classifiers:
            if isinstance(c, knn_classifier):
                classification_votes.append(c.classify(features))
            else:
                classification_votes.append(c.inner_classifier.predict(features))

        return 0 if classification_votes.count(0) > classification_votes.count(1) else 1


class Contest_factory(abstract_classifier_factory):
    def train(self, data, labels) -> Contest_classifier:
        return Contest_classifier(data, labels)


class perceptron_classifier(abstract_classifier):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        This constructor also trains the classifier by creating the decision tree.
        :param data: the unlabeled data of the dataset
        :param labels: the correct labels of the data
        """

        self.inner_classifier = Perceptron(tol=1e-3)
        self.inner_classifier.fit(data, labels)

    def classify(self, features) -> int:
        return self.inner_classifier.predict(features)


class perceptron_factory(abstract_classifier_factory):
    def train(self, data, labels) -> perceptron_classifier:
        return perceptron_classifier(data, labels)


def question7():
    # ID3
    id3_results = []
    factory = ID3_factory()
    accuracy, error = evaluate(factory, 2)
    id3_results.append((1, accuracy, error))
    export_to_csv(id3_results, 'experiments12ID3.csv')

    # Perceptron
    perceptron_results = []
    factory = perceptron_factory()
    accuracy, error = evaluate(factory, 2)
    perceptron_results.append((2, accuracy, error))
    export_to_csv(perceptron_results, 'experiments12perceptron.csv')


def evaluate_without_known_bad_features(classifier_factory: abstract_classifier_factory, k: int) -> (float, float):
    num_folds = k
    data = {}
    known_bad_features = [33]
    for i in range(1, num_folds + 1):
        data[i - 1] = load_k_fold_data(i)
        for bad_feature in known_bad_features:
            data[i - 1] = (np.delete(data[i - 1][0], bad_feature, 1), data[i - 1][1])
    for i in range(1, num_folds + 1):
        data[i - 1] = load_k_fold_data(i)
    # data is now a dictionary of the form {i:(features, labels)} where i is the key
    # and (features, labels) is the value, which is a tuple of the features matrix
    # and the labels vector
    return calculate_evaluation_accuracy_error(data, classifier_factory)


def evaluate_without_bad_features(classifier_factory: abstract_classifier_factory, k: int):
    num_folds = k
    data = {}
    known_bad_features = [33]
    for i in range(1, num_folds + 1):
        data[i - 1] = load_k_fold_data(i)
        for bad_feature in known_bad_features:
            data[i - 1] = (np.delete(data[i - 1][0], bad_feature, 1), data[i - 1][1])

    accuracies = {}
    for bad_feature in range(np.shape(data[1][0])[1]):
        data_without_feature = data.copy()
        for fold in range(num_folds):
            data_without_feature[fold] = (np.delete(data_without_feature[fold][0], bad_feature, 1),
                                          data_without_feature[fold][1])
        accuracies[bad_feature] = calculate_evaluation_accuracy_error(data_without_feature, classifier_factory)[0]
    sorted_accuracies = sorted(accuracies.items(), key=lambda tup: tup[1], reverse=True)
    print(f'{classifier_factory} classifier: {sorted_accuracies[:20]}')

    # data is now a dictionary of the form {i:(features, labels)} where i is the key
    # and (features, labels) is the value, which is a tuple of the features matrix
    # and the labels vector


def checking_bad_features(dataset):
    # creating folds
    num_folds = 2
    # split_crosscheck_groups(dataset, num_folds=num_folds, is_testing=True)
    # print('splitting to folds completed.')

    # evaluate_without_bad_features(ID3_factory(), num_folds)
    # evaluate_without_bad_features(perceptron_factory(), num_folds)

    # Naive Bayesian
    # factory = MultinomialNB_factory()
    # evaluate_without_bad_features(factory, num_folds)

    # KNN
    k_list = [1, 3, 5, 7, 13]
    results = []
    for k in k_list:
        print(f'k={k}')
        accuracy, error = evaluate_without_known_bad_features(knn_factory(k), num_folds)
        results.append((k, accuracy, error))
    print(results)


def contest(num_folds=2):
    accuracy, error = evaluate(Contest_factory(), num_folds)
    print(accuracy, error)


def main():
    # dataset is a 3-tuple consisting of:
    # (2D ndarray of training features, list of labels,2D ndarray of testing features)
    # dataset = load_data()
    # data = dataset[0]
    # labels = dataset[1]
    # test_set = dataset[2]
    # factory = knn_factory(k=5)
    # classifier = factory.train(data=data, labels=labels)
    # result = classifier.classify(test_set[0])
    # print(result)
    # load_k_fold_data(2)
    # patients, labels, test = load_data()
    # split_crosscheck_groups(patients, labels, 2)
    # question5()
    # question7()
    # dataset = load_data()
    # checking_bad_features(dataset)
    contest(2)

if __name__ == '__main__':
    main()
