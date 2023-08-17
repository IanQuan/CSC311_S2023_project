from matplotlib import pyplot as plt
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torch

from CSC311_S2023_project.Final_project.utils import *


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        temp = self.g(inputs)
        temp = F.sigmoid(temp)
        temp = self.h(temp)
        out = F.sigmoid(temp)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: training_loss, validation_accuracies
    """
    # TODO: Add a regularizer to the cost function. 

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # Record the training loss and validation accuracy.
    training_loss = []
    validation_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss = loss + lamb / 2 * model.get_weight_norm()  # Add the regularizer.
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        training_loss.append(train_loss)
        # Evaluate the model on the validation set
        valid_acc = evaluate(model, zero_train_data, valid_data)
        validation_accuracies.append(valid_acc)

        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #       "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return training_loss, validation_accuracies
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################0


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k_values = [10, 50, 100]
    # Set optimization hyperparameters.
    lr_values = [0.005, 0.001, 0.02]
    num_epoch_values = [25, 35, 45]
    lamb_values = [0, 0.001, 0.01, 0.1, 1]

    # # Q3(c)
    max_accuracy, k_star, optimal_epoch, optimal_lr = 0, 0, 0, 0
    # Find the optimal hyperparamters
    for k in k_values:
        for lr in lr_values:
            valid_accuracy = 0
            for num_epoch in num_epoch_values:
                # Train the autoencoder using latent dimensions of k
                model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
                lamb = 0.0
                train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
                valid_accuracy = evaluate(model, zero_train_matrix, valid_data)

                if valid_accuracy > max_accuracy:
                    max_accuracy = valid_accuracy
                    k_star = k
                    optimal_epoch = num_epoch
                    optimal_lr = lr
                print(f"k: {k}, learning rate: {lr}, epoch: {num_epoch}, Validation Accuracy: {valid_accuracy}")
    print(f"Optimal k*: {k_star}")
    print(f"Optimal learning rate: {optimal_lr}")
    print(f"Optimal epoch: {optimal_epoch}")
    print(f"Optimal validation accuracy: {max_accuracy}")

    # Q3(d)
    # Choose k = 50, lr = 0.02, lamb = 0.0, num_epoch = 25
    model = AutoEncoder(train_matrix.shape[1], k=50)
    training_loss, validation_accuracies = train(model, 0.02, 0.0, train_matrix, zero_train_matrix, valid_data, 25)

    data_range = range(len(training_loss))

    plt.plot(data_range, training_loss, label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training loss")
    plt.legend()
    plt.savefig("Q3(d)_train_loss_vs_epoch")
    plt.show()

    data_range = range(len(validation_accuracies))
    plt.plot(data_range, validation_accuracies, label='valid accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Validation accuracy")
    plt.legend()
    plt.savefig("Q3(d)_valid_acc_vs_epoch")
    plt.show()

    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print("Test Accuracy: {}".format(test_accuracy))

    # # Q3(e)
    lamb_values = [0.001, 0.01, 0.1, 1]
    max_accuracy, optimal_lamb = 0, 0

    # Continue using k = 100, lr = 0.005, num_epoch = 50, find the optimal value of lambda
    for lamb in lamb_values:
        model = AutoEncoder(num_question=train_matrix.shape[1], k=50)
        train(model, 0.02, lamb, train_matrix, zero_train_matrix, valid_data, 50)

        valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            optimal_lamb = lamb

        print(f"lambda: {lamb}, Validation Accuracy = {max_accuracy}")
    print(f"Optimal value of lambda: {optimal_lamb}")

    # Find the validation and test accuracy with the optimal lambda
    model = AutoEncoder(num_question=train_matrix.shape[1], k=50)
    train(model, 0.02, optimal_lamb, train_matrix, zero_train_matrix, valid_data, 50)
    valid_accuracy = evaluate(model, zero_train_matrix, valid_data)
    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print("Validation Accuracy: {}".format(valid_accuracy))
    print("Test Accuracy: {}".format(test_accuracy))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
