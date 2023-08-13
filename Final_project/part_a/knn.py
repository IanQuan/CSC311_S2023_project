from sklearn.impute import KNNImputer
from matplotlib import pyplot as plt

from CSC311_S2023_project.Final_project.utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("For k = {}, Validation Accuracy on user-based collaborative filtering: {}".format(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("For k = {}, Validation Accuracy on item-based collaborative filtering: {}".format(k, acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Q1(a)
    k_values = [1, 6, 11, 16, 21, 26]
    accuracies_by_user = []
    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracies_by_user.append(acc)

    plt.plot(k_values, accuracies_by_user)
    plt.xlabel('k')
    plt.ylabel('accuracy on the validation data')
    plt.title("Accuracy on the validation data imputed by user as a function of k")
    plt.savefig("Q1(a)_acc_by_user")
    plt.show()

    # Q1(b)
    # Find k* which has the highest performance on validation data
    highest_acc = max(accuracies_by_user)
    k_star = k_values[accuracies_by_user.index(highest_acc)]

    # Calculate the test accuracy on the chosen k*
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print(f"The final test accuracy when k_star = {k_star} on user-based collaborative filtering is {test_acc}")

    # Q1(c)
    accuracies_by_item = []
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracies_by_item.append(acc)

    plt.plot(k_values, accuracies_by_item)
    plt.xlabel('k')
    plt.ylabel('accuracy on the validation data')
    plt.title("Accuracy on the validation data imputed by item as a function of k")
    plt.savefig("Q1(c)_acc_by_item")
    plt.show()
    # Q1(b)
    # Find k* which has the highest performance on validation data
    highest_acc = max(accuracies_by_item)
    k_star = k_values[accuracies_by_item.index(highest_acc)]

    # Calculate the test accuracy on the chosen k*
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_star)
    print(f"The final test accuracy when k_star = {k_star} on item-based collaborative filtering is {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
