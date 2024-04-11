import sys, os

from common.activate_functions import *
from common.error_functions import *

sys.path.append(os.pardir)


# 1층 (입력층 -> 은닉층)
# 2층 (은닉층 -> 출력층)

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad
class TwoLayerNeuralNetwork:
    # weight_init_std : randn 함수에 적용할 표준편차 (randn 함수의 기본 표준 편차는 1)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # W1 : 1층의 가중치, b1 : 1층의 편향, W2 : 2층의 가중치, n2 : 2층의 편향
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def predict(self, x):
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1  # 입력에 가중치 적용
        z1 = sigmoid(a1)  # 활성화 함수 적용
        a2 = np.dot(z1, W2) + b2  # 입력에 가중치 적용
        y = softmax(a2)  # 활성화 함수 적용

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):  # x는 예측을 위한 입력, t는 테스트 데이터
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {
            'W1': numerical_gradient(loss_w, self.params['W1']),
            'b1': numerical_gradient(loss_w, self.params['b1']),
            'W2': numerical_gradient(loss_w, self.params['W2']),
            'b2': numerical_gradient(loss_w, self.params['b2'])
        }

        return grads
