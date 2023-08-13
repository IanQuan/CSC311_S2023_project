from matplotlib import pyplot as plt

from CSC311_S2023_project.Final_project.utils import *


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data['user_id'])):
        user_i = data['user_id'][i]
        question_j = data['question_id'][i]
        z = theta[user_i] - beta[question_j]
        c_ij = data['is_correct'][i]

        log_lklihood += c_ij * z - np.log(1 + np.exp(z))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_update = np.zeros(theta.shape)
    beta_update = np.zeros(beta.shape)

    for i in range(len(data['user_id'])):
        user_i = data['user_id'][i]
        question_j = data['question_id'][i]
        z = theta[user_i] - beta[question_j]
        c_ij = data['is_correct'][i]

        theta_update[user_i] += (c_ij - sigmoid(z))
        beta_update[question_j] += (sigmoid(z) - c_ij)

    # update the gradient
    theta += lr * theta_update
    beta += lr * beta_update
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(data["user_id"]))
    beta = np.zeros(len(data["question_id"]))

    train_accuracies = []
    validation_accuracies = []
    train_log_likelihoods = []
    validation_log_likelihoods = []

    for i in range(iterations):
        # Training data:
        train_accuracies.append(evaluate(data, theta, beta))

        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_log_likelihoods.append(-neg_lld)

        # Evaluation data:
        accuracy = evaluate(data=val_data, theta=theta, beta=beta)
        validation_accuracies.append(accuracy)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        validation_log_likelihoods.append(-val_neg_lld)

        # Print Result:
        print("Iteration: {} \t Training log-likelihood: {} \t Validation Accuracy: {}".format(i + 1, neg_lld, accuracy))

        # update theta and beta using gradient descent
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_accuracies, validation_accuracies, train_log_likelihoods, validation_log_likelihoods


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iterations = 25
    lr = 0.005
    iter_range = range(1, iterations + 1)
    (theta, beta, train_accuracies, validation_accuracies, train_log_likelihoods,
     validation_log_likelihoods) = irt(train_data, val_data, lr, iterations)

    plt.scatter(iter_range, train_accuracies)
    plt.scatter(iter_range, validation_accuracies)
    plt.plot(iter_range, train_accuracies)
    plt.plot(iter_range, validation_accuracies)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy as a function of iterations")
    plt.legend(["Train", "Validation"])
    plt.savefig("Q2(b)_acc_vs_iteration")
    plt.show()

    # Plot train log-likelihood on the primary y-axis (left y-axis)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(iter_range, train_log_likelihoods, label="Train", color="blue")
    ax1.plot(iter_range, train_log_likelihoods, color="blue")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train Log-Likelihood", color="blue")

    # Plot validation log-likelihood on the secondary y-axis (right y-axis)
    ax2 = ax1.twinx()
    ax2.scatter(iter_range, validation_log_likelihoods, label="Validation", color="orange")
    ax2.plot(iter_range, validation_log_likelihoods, color="red")
    ax2.set_ylabel("Validation Log-Likelihood", color="red")
    lines_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    lines = lines_ax1 + lines_ax2
    labels = labels_ax1 + labels_ax2
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Log-Likelihood as a function of iterations")
    plt.tight_layout()
    plt.savefig("Q2(b)_lld_vs_iteration")
    plt.show()

    final_validation_accuracy = evaluate(val_data, theta, beta)
    final_test_accuracy = evaluate(test_data, theta, beta)
    print("Final Validation Accuracy: {}".format(final_validation_accuracy))
    print("Final Test Accuracy: {}".format(final_test_accuracy))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    j_1 = beta.argmax()  # highest difficulty
    j_2 = beta.argmin()  # lowest difficulty
    j_3 = np.abs(beta - np.median(beta)).argmin()  # average difficulty

    # Evenly space the range of theta
    theta_range = np.linspace(-max(theta)-0.2, max(theta)+0.2, 1000)
    # Get the probability with the selected beta value
    probs_j1 = sigmoid(theta_range - beta[j_1])
    probs_j2 = sigmoid(theta_range - beta[j_2])
    probs_j3 = sigmoid(theta_range - beta[j_3])

    # x_axis is range of theta, y_variable is probability
    plt.scatter(theta_range, probs_j1, color='red', label='Q1 (difficult)')
    plt.scatter(theta_range, probs_j2, color='green', label='Q2 (easy)')
    plt.scatter(theta_range, probs_j3, color='blue', label='Q3 (average)')
    plt.xlabel(r'Student\'s Ability $(\theta)$')
    plt.ylabel(r'Probability of Correct Response $p(c_{ij}=1)$')
    plt.title('Probability of Correct Response vs Student\'s Ability')
    plt.legend()
    plt.savefig("Q2(d)_prob_vs_theta")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
