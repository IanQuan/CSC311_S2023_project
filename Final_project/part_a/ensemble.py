from item_response import *


def generate_new_dataset(data) -> dict:
    """
    Bootstrap the training set by randomly sample data with replacement.
    The returned new data set should have the same number of samples
    """
    new_dataset = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    i = 0
    size = len(data['user_id'])
    for i in range(size):
        index = np.random.randint(0, size)
        new_dataset['user_id'].append(data['user_id'][index])
        new_dataset['question_id'].append(data['question_id'][index])
        new_dataset['is_correct'].append(data['is_correct'][index])
    return new_dataset


def evaluate(data, theta_list, beta_list):
    """
    Evaluate the accuracy of the model.
    """
    predictions = []
    for i, q in enumerate(data["question_id"]):
        question_pred = []
        for k in range(len(theta_list)):
            u = data["user_id"][i]
            x = (theta_list[k][u] - beta_list[k][q]).sum()
            p_a = sigmoid(x)
            if p_a >= 0.5:
                question_pred.append(1)
            else:
                question_pred.append(0)
        # For the current question, add the average prediction of all students into the list
        predictions.append(np.mean(question_pred) >= 0.5)
    predictions = np.array(predictions)
    # Return teh average accuracy of the model
    return np.sum((data["is_correct"] == predictions)) / len(data["is_correct"])


def ensemble(train_data, val_data, lr, iterations):
    """
    Train IRT model with ensemble.
    (We train multiple IRT models and use the majority vote to predict the answer.)
    """
    theta_list = []
    beta_list = []

    for i in range(3):
        print(f"IRT Model {i + 1} training:")
        train_data = generate_new_dataset(train_data)
        theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst = irt(train_data, val_data, lr, iterations)
        theta_list.append(theta)
        beta_list.append(beta)
    return theta_list, beta_list


def main():
    data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Initialize the hyperparameters
    lr = 0.005
    iterations = 20

    # Ensemble method 1
    theta_list, beta_list = ensemble(data, val_data, lr, iterations)

    # Evaluate the performance on the validation and test data
    val_acc = evaluate(val_data, theta_list, beta_list)
    print(f"Final Validation Accuracy after ensemble: {val_acc}")
    test_acc = evaluate(test_data, theta_list, beta_list)
    print(f"Test Accuracy after ensemble: {test_acc}")


if __name__ == "__main__":
    main()

