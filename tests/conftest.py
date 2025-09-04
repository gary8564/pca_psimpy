import os
import numpy as np
import pytest


@pytest.fixture(scope="session")
def humanity_data():
    curr_path = os.path.dirname(__file__)
    data_path = os.path.join(curr_path, "data")

    X_train = np.genfromtxt(os.path.join(data_path, "humanityX.csv"),  delimiter=",")
    Y_train = np.genfromtxt(os.path.join(data_path, "humanityY.csv"),  delimiter=",")
    X_test  = np.genfromtxt(os.path.join(data_path, "humanityXt.csv"), delimiter=",")
    Y_test  = np.genfromtxt(os.path.join(data_path, "humanityYt.csv"), delimiter=",")

    return X_train, Y_train, X_test, Y_test