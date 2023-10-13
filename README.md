# CSC311_S2023_project

## Online Education Assessment with Machine Learning
This repository contains the code and documentation for a machine learning project focused on improving the assessment of student understanding in online education platforms. The project's main objectives include predicting the correctness of students' answers to diagnostic questions, analyzing various machine learning algorithms, and proposing modifications for enhanced accuracy.

## Project Overview
**Objective:** 
Predict student correctness on diagnostic questions based on previous answers and other students' responses.
Dataset: Utilize a subsampled dataset of 542 students, 1774 diagnostic questions from the Eedi online education platform.
**Tools: Python, NumPy, Scipy, Pandas, PyTorch.**
The project is divided into two parts:

## Part A - Baseline Models
**k-Nearest Neighbor (kNN):**

- Implement kNN for user-based and item-based collaborative filtering.
- Experiment with different k values and evaluate their performance.
- Compare user-based and item-based collaborative filtering.

**Item Response Theory (IRT):**

- Implement IRT to predict student correctness.
- Derive the log-likelihood and perform alternating gradient descent.
- Report training and validation log-likelihoods.

**Matrix Factorization or Neural Networks:**

- Option 1: Implement matrix factorization with SVD and ALS.
- Option 2: Implement an autoencoder neural network.
Compare the two methods and apply L2 regularization.

**Ensemble:**

- Implement a bagging ensemble of base models.
- Generate ensemble predictions by averaging base model predictions.
- Evaluate the impact of ensemble learning.

## Part B - Custom Algorithm Modification
We introduce feature enrichment by considering students' ability and experiment with a new loss function for improved model performance.

### Feature Enrichment: Students' Ability
**Motivation:**

To enhance the model's understanding of student performance, we augment the feature space by appending a vector of students' ability (θ) to the input data. This enrichment aims to provide the neural network with more nuanced information, enabling it to capture complex patterns and make informed predictions based on students' abilities.

**Hypothesis:**

The inclusion of students' ability information in the input data will improve the overall model performance. Both training and validation accuracy are expected to increase compared to the base model.

**Algorithm Box:**
1. Concatenate the original input data with the students’ ability tensor according to the userID.
2. Increase the dimension of the linear function g.
3. Train the model with the new input data.
4. Tune the hyper-parameters and evaluate the model.

### Binary Cross Entropy Loss
**Motivation:**

To further enhance the Student Ability Model, we adopt the Binary Cross Entropy Loss. This loss function is well-suited for binary classification problems, where the model predicts correctness with class labels 0 and 1. It is particularly beneficial when handling class imbalance, as it penalizes misclassifications more heavily, making the model sensitive to the minority class.

**Hypothesis:** 

With the Binary Cross Entropy Loss, we anticipate improved training and validation accuracy compared to the base model.

**Algorithm Box:**
1. Change the loss function from Mean Squared Error (MSE) to Binary Cross Entropy (BCE).
2. Implement feature enrichment by including students' ability in the input data.
3. Train the model with the new loss function.
4. Fine-tune hyperparameters and evaluate model performance.


## Report
A detailed report explaining the project's methodology, findings, and insights is included in the repository. It provides a comprehensive understanding of the project's goals and outcomes.

## Usage
Follow the instructions in the project code to run and evaluate different machine learning algorithms.
Modify and experiment with the algorithms to test your own hypotheses.
