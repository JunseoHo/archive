import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # x의 최대 값을 제하여 오버플로우 방지
    return exp_x / np.sum(exp_x, axis=0)
