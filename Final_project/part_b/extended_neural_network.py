from sklearn.model_selection import KFold
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torch
import matplotlib.pyplot as plt

from utils_checkpoint import *
from CSC311_S2023_project.Final_project.part_a.item_response import *
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
    def __init__(self, num_question, student_ability_dict, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question + len(student_ability_dict[0]), k)
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
        x = self.g(inputs)
        x = F.sigmoid(x)
        x = self.h(x)
        out = F.sigmoid(x)
        #####################################################################
        return out


def concatenate_theta(zero_train_data, student_ability_dict, user_index):

    student_ability_vec = torch.tensor(student_ability_dict[user_index], dtype=torch.float32)
    return torch.cat((zero_train_data[user_index], student_ability_vec))


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, student_ability_dict):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: (lst_train_loss, lst_val_acc)
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Record the training loss and validation accuracy.
    lst_train_loss = []
    lst_val_acc = []

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = concatenate_theta(zero_train_data, student_ability_dict, user_id).unsqueeze(0)

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            offset = len(student_ability_dict[0])
            target = inputs.clone()[:, :-offset]
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            ce = cross_entropy(train_data, target, output, user_id)
            # Add the regularizer.
            # loss = torch.sum((output - target) ** 2.) + lamb / 2 * model.get_weight_norm()
            ce_loss = ce + lamb / 2 * model.get_weight_norm()
            ce_loss.backward()

            train_loss += ce_loss.item()
            optimizer.step()

        # Evaluate the model on the validation set.
        valid_acc = evaluate(model, zero_train_data, valid_data, student_ability_dict)
        lst_train_loss.append(train_loss)
        lst_val_acc.append(valid_acc)

        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #       "Valid Acc: {}".format(epoch + 1, train_loss, valid_acc))
    return lst_train_loss, lst_val_acc


def evaluate(model, zero_train_data, valid_data, student_ability_dict):
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

    for i, user_id in enumerate(valid_data["user_id"]):
        # inputs = Variable(train_data[user_id]).unsqueeze(0)
        # inputs = inputs.to(torch.float32)

        inputs = concatenate_theta(zero_train_data, student_ability_dict, user_id).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def cross_entropy(train_data, target, output, user_index):
    output = output[:, :train_data.shape[1]]
    target = target[:, :train_data.shape[1]]

    valid_mask = ~torch.isnan(train_data[user_index])
    ce = F.binary_cross_entropy(output[0][valid_mask], target[0][valid_mask],  reduction='sum')
    return ce

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # get theta, i.e. i-th studnets' ability
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    theta, _, _, _, _, _ = irt(train_data, val_data, 0.005, 25)
    student_ability_dict = {}
    for i in range(len(theta)):
        student_ability_dict[i] = [theta[i]]

    # Set model hyperparameters.
    k_values = [10, 50, 100]
    # Set optimization hyperparameters.
    lr_values = [0.0005, 0.001]
    num_epoch_values = [20, 30, 40, 50, 60]
    lamb_values = [0, 0.001, 0.01, 0.1, 1]

    # testing with specific parameters
    k_star = 10
    optimal_lr = 0.0005
    optimal_epoch = 50
    best_lamb = 0.001
    best_split = 5

    # max_accuracy, k_star, optimal_epoch, optimal_lr = 0, 0, 0, 0
    # # Find the optimal hyperparamters
    # for k in k_values:
    #     for lr in lr_values:
    #         valid_accuracy = 0
    #         for num_epoch in num_epoch_values:
    #             # Train the autoencoder using latent dimensions of k
    #             model = AutoEncoder(num_question=train_matrix.shape[1], student_ability_dict=student_ability_dict, k=k)
    #             lamb = 0.0
    #             train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, student_ability_dict)
    #             valid_accuracy = evaluate(model, zero_train_matrix, valid_data, student_ability_dict)
    #
    #             if valid_accuracy > max_accuracy:
    #                 max_accuracy = valid_accuracy
    #                 k_star = k
    #                 optimal_epoch = num_epoch
    #                 optimal_lr = lr
    #             print(f"k: {k}, learning rate: {lr}, epoch: {num_epoch}, Validation Accuracy: {valid_accuracy}")
    # print(f"Optimal k*: {k_star}")
    # print(f"Optimal learning rate: {optimal_lr}")
    # print(f"Optimal epoch: {optimal_epoch}")
    # print(f"Optimal validation accuracy: {max_accuracy}")

    # finding optimal epoch
    # k = 10
    # lr = 0.001
    # lamb = 0.0
    # max_accuracy, optimal_epoch = 0, 0
    #
    # for num_epoch in num_epoch_values:
    #     valid_accuracy = 0
    #     model = AutoEncoder(num_question=train_matrix.shape[1], student_ability_dict=student_ability_dict, k=k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, student_ability_dict)
    #     valid_accuracy = evaluate(model, zero_train_matrix, valid_data, student_ability_dict)
    #
    #     if valid_accuracy > max_accuracy:
    #         max_accuracy = valid_accuracy
    #         optimal_epoch = num_epoch
    #     print(f"k: {k}, learning rate: {lr}, epoch: {num_epoch}, Validation Accuracy: {valid_accuracy}")
    # print(f"Optimal epoch: {optimal_epoch}")

    model = AutoEncoder(num_question=train_matrix.shape[1], student_ability_dict=student_ability_dict, k=k_star)
    training_loss, validation_accuracies = train(model, optimal_lr, 0.001, train_matrix, zero_train_matrix, valid_data, optimal_epoch, student_ability_dict)
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
    # valid_accuracy = evaluate(model, zero_train_matrix, valid_data, student_ability_dict)
    test_accuracy = evaluate(model, zero_train_matrix, test_data, student_ability_dict)
    # print("Validation Accuracy: {}".format(valid_accuracy))
    print("Test Accuracy: {}".format(test_accuracy))

# SA
# k: 10, learning rate: 0.001, epoch: 20, Validation Accuracy: 0.691927744848998
# k: 10, learning rate: 0.001, epoch: 30, Validation Accuracy: 0.6817668642393452
# k: 10, learning rate: 0.001, epoch: 40, Validation Accuracy: 0.6824724809483489
# k: 10, learning rate: 0.001, epoch: 50, Validation Accuracy: 0.6838837143663562
# k: 10, learning rate: 0.001, epoch: 60, Validation Accuracy: 0.68077900084674
# k: 10, learning rate: 0.01, epoch: 20, Validation Accuracy: 0.6517075924357889
# k: 10, learning rate: 0.01, epoch: 30, Validation Accuracy: 0.645780412080158
# k: 10, learning rate: 0.01, epoch: 40, Validation Accuracy: 0.6491673722833756
# k: 10, learning rate: 0.01, epoch: 50, Validation Accuracy: 0.6432401919277448
# k: 10, learning rate: 0.01, epoch: 60, Validation Accuracy: 0.639994355066328
# k: 10, learning rate: 0.1, epoch: 20, Validation Accuracy: 0.6299745977984759
# k: 10, learning rate: 0.1, epoch: 30, Validation Accuracy: 0.6178379904036128
# k: 10, learning rate: 0.1, epoch: 40, Validation Accuracy: 0.611346316680779
# k: 10, learning rate: 0.1, epoch: 50, Validation Accuracy: 0.615015523567598
# k: 10, learning rate: 0.1, epoch: 60, Validation Accuracy: 0.6075359864521592
# k: 50, learning rate: 0.001, epoch: 20, Validation Accuracy: 0.6759808072255151
# k: 50, learning rate: 0.001, epoch: 30, Validation Accuracy: 0.6658199266158623
# k: 50, learning rate: 0.001, epoch: 40, Validation Accuracy: 0.6548123059554051


# ce + SA
# k: 10, learning rate: 0.0005, epoch: 20, Validation Accuracy: 0.6560824160316117
# k: 10, learning rate: 0.0005, epoch: 30, Validation Accuracy: 0.6762630539091166
# k: 10, learning rate: 0.0005, epoch: 40, Validation Accuracy: 0.6855771944679651
# k: 10, learning rate: 0.0005, epoch: 50, Validation Accuracy: 0.6987016652554332
# k: 10, learning rate: 0.0005, epoch: 60, Validation Accuracy: 0.6953147050522156
# k: 10, learning rate: 0.001, epoch: 20, Validation Accuracy: 0.6872706745695738
# k: 10, learning rate: 0.001, epoch: 30, Validation Accuracy: 0.6857183178097658
# k: 10, learning rate: 0.001, epoch: 40, Validation Accuracy: 0.6852949477843635
# k: 10, learning rate: 0.001, epoch: 50, Validation Accuracy: 0.6896697713801863
# k: 10, learning rate: 0.001, epoch: 60, Validation Accuracy: 0.676545300592718
# k: 50, learning rate: 0.0005, epoch: 20, Validation Accuracy: 0.69037538808919
# k: 50, learning rate: 0.0005, epoch: 30, Validation Accuracy: 0.6865650578605701
# k: 50, learning rate: 0.0005, epoch: 40, Validation Accuracy: 0.6857183178097658
# k: 50, learning rate: 0.0005, epoch: 50, Validation Accuracy: 0.6751340671747107
# k: 50, learning rate: 0.0005, epoch: 60, Validation Accuracy: 0.6711826136042901
# k: 50, learning rate: 0.001, epoch: 20, Validation Accuracy: 0.682895850973751
# k: 50, learning rate: 0.001, epoch: 30, Validation Accuracy: 0.6727349703640982
# k: 50, learning rate: 0.001, epoch: 40, Validation Accuracy: 0.6598927462602314
# k: 50, learning rate: 0.001, epoch: 50, Validation Accuracy: 0.6615862263618403
# k: 50, learning rate: 0.001, epoch: 60, Validation Accuracy: 0.6519898391193903
# k: 100, learning rate: 0.0005, epoch: 20, Validation Accuracy: 0.6920688681907987
# k: 100, learning rate: 0.0005, epoch: 30, Validation Accuracy: 0.6793677674287327
# k: 100, learning rate: 0.0005, epoch: 40, Validation Accuracy: 0.6762630539091166
# k: 100, learning rate: 0.0005, epoch: 50, Validation Accuracy: 0.6711826136042901
# k: 100, learning rate: 0.0005, epoch: 60, Validation Accuracy: 0.6649731865650579
# k: 100, learning rate: 0.001, epoch: 20, Validation Accuracy: 0.6696302568444821
# k: 100, learning rate: 0.001, epoch: 30, Validation Accuracy: 0.6639853231724527
# k: 100, learning rate: 0.001, epoch: 40, Validation Accuracy: 0.6624329664126446
# k: 100, learning rate: 0.001, epoch: 50, Validation Accuracy: 0.6608806096528366
# k: 100, learning rate: 0.001, epoch: 60, Validation Accuracy: 0.65961049957663
# Optimal k*: 10
# Optimal learning rate: 0.0005
# Optimal epoch: 50
# Optimal validation accuracy: 0.6987016652554332



if __name__ == "__main__":
    main()
