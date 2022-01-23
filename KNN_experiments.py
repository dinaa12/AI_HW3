import subprocess
import random

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, y_test, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :return: indices of top 'b' features, sorted.
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    top_b_features_indices = []

    # ====== YOUR CODE: ======

    # calc cov per feature #
    x_t = x.T
    cov_list = []
    for feature_idx, row in enumerate(x_t):  # = for col in x
        array = (row, y)
        cov_matrix = np.cov(array)
        cov = abs(cov_matrix[0][1])
        cov_list.append((cov, feature_idx))
    cov_list.sort(key=lambda tup:tup[0], reverse=True)

    # calc acc per feature #
    acc_list = []
    for feature_idx in range(8):
        x_train_new = x_train[:, feature_idx]
        x_test_test = x_test[:, feature_idx]
        # run_knn #
        neigh = KNNClassifier(k=best_k)
        neigh.train(x_train_new, y)
        y_pred = neigh.predict(x_test_test)
        acc = accuracy(y_test, y_pred)
        acc_list.append((acc, feature_idx))
    acc_list.sort(key=lambda tup:tup[0], reverse=True)

    # calc weighted sum of acc and cov per feature #
    acc_and_cov_list = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        acc_and_cov_list[acc_list[i][1]] += 0.8*i
        acc_and_cov_list[cov_list[i][1]] += 0.2*i

    acc_and_cov_list_for_sort = []
    for i in range(8):
        acc_and_cov_list_for_sort.append((acc_and_cov_list[i], i))

    acc_and_cov_list_for_sort.sort(key=lambda tup: tup[0])
    top_b = acc_and_cov_list_for_sort[0:b]
    top_b_features_indices = [x[1] for x in top_b]

    # ========================
    return top_b_features_indices


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    #run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 4

    # # ========================================================================

    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, y_test, b=b, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
