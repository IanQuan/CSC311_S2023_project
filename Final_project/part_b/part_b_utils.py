import ast
import csv
import datetime as dt
import os

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def dict_to_matrix(data: dict) -> np.ndarray:
    # convert dict into matrix
    user_ids = data['user_id']
    question_ids = data['question_id']
    is_correct = data['is_correct']
    # create a Ns * Nq matrix
    matrix = np.empty((np.max(user_ids) + 1, np.max(question_ids) + 1))
    matrix[:] = np.nan
    matrix[user_ids, question_ids] = is_correct
    return matrix


def load_csv(path) -> dict:
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_student_meta_data(root_dir="../data") -> dict:
    path = os.path.join(root_dir, "student_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "gender": [],
        "dob": [],
        "premium_pupil": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            data["user_id"].append(int(row[0]))
            data["gender"].append(int(row[1]))
            if row[2] != '':
                actual_date = dt.datetime.fromisoformat(row[2])
                start_date = dt.datetime.fromtimestamp(0)
                days = (actual_date - start_date).days / 365
                data['dob'].append(days)
            else:
                data['dob'].append(float('nan'))
            if row[3] != '':
                data["premium_pupil"].append(float(row[3]))
            else:
                data["premium_pupil"].append(0.)
    return data


def process_student_meta_data(data: dict) -> dict:
    result = {}
    dobs = np.array(data['dob'], dtype='float32')
    mean_dob = np.nanmean(dobs)
    dobs[np.isnan(dobs)] = mean_dob
    dob_factors = sigmoid(-(mean_dob - dobs))
    user_ids = data['user_id']
    genders = data['gender']
    pps = data['premium_pupil']
    for i in range(len(user_ids)):
        result[user_ids[i]] = [genders[i], dob_factors[i], pps[i]]
    return result


def load_question_meta_data(root_dir="../data") -> dict[int: list[int]]:
    path = os.path.join(root_dir, "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            question_id = int(row[0])
            data[question_id] = np.array([0] * 388, dtype='float32')
            subjects = ast.literal_eval(row[1])
            data[question_id][subjects] = 1
    return data


def get_average_correct(data: dict):
    is_correct = np.array(data['is_correct'])
    return is_correct.sum() / len(is_correct)
