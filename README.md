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
Propose a modification to one of the algorithms implemented in Part A to improve performance.

Provide a formal description of the modification, figures or diagrams, and a detailed comparison with baseline models.
Highlight limitations and potential future work.

## Report
A detailed report explaining the project's methodology, findings, and insights is included in the repository. It provides a comprehensive understanding of the project's goals and outcomes.

## Usage
Follow the instructions in the project code to run and evaluate different machine learning algorithms.
Modify and experiment with the algorithms to test your own hypotheses.
