import os
import sys
import pickle
import numpy as np

from common.activate_functions import sigmoid, softmax
from dataset.mnist import load_mnist

sys.path.append(os.pardir)  # 기존 시스템 경로뿐만 아니라 부모 디렉토리에서도 파일을 탐색하기 위함


# x : 이미지
# 각 이미지는 784(28 * 28)개의 원소를 가지는 벡터로 변환된다.
# 원소의 값은 명도이다. 즉 0이면 검정, 255면 하양이다.

# t : 레이블
# 각 이미지가 표현하는 숫자를 값으로 가진다.

# 예를 들어 x_train[0]은 5를 표현하는 이미지의 픽셀 값이 784 크기의 벡터로 표현된 것이다.
# x_train[0]이 표현하는 값이 5이므로 t_train[0]은 5이다.

# flatten이 True이면 이미지가 1 * 28 * 28의 3차원 벡터로 반환되며
# False이면 784개의 원소로 구성되는 1차원 벡터로 반환된다.

# normalize는 픽셀의 값을 [0.0, 1.0]으로 정규화할지 결정한다.

# one_hot_label은 레이블을 원-핫 인코딩으로 저장할지 결정한다.
# 예를 들어 레이블 5이면 one_hot_label의 값에 따라 다음과 같이 저장된다.
# True  : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# False : 5

# 처음 실행하면 파이선에서 제공하는 기능인 pickle 파일이 생성된다.

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    return x_test, t_test


# 이미지 하나는 784개의 픽셀 값으로 구성된다.
# 그리고 우리는 각 입력이 10개의 숫자 중 어떤 숫자에 해당되는지 분류하고자 한다.
# 따라서 입력층 뉴런은 784개, 출력층 뉴런은 10개로 구성되어야 한다.
# 아래 예시에서 은닉층은 2개이며 뉴런의 개수는 임의로 각각 50개, 100개씩 배치되어 있다.

# 미리 학습된 매개변수를 담고 있는 sample_weight.pkl 파일을 network 객체로 변환

def init_net():
    with open("sample_weight.pkl", 'rb') as f:
        net = pickle.load(f)
    return net


# 매개변수를 담고 있는 신경망과 입력을 받아 결과를 예측하는 함수

def predict(net, x):
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# main

x, t = get_data()
net = init_net()

batch_size = 100  # 한번에 여러 장을 처리할 수 있도록 배치 크기를 정의한다.
correct_count = 0
for i in range(0, len(x), batch_size):  # 0부터 len(x)까지 batch_size만큼의 보폭으로 반복한다.
    x_batch = x[i:i + batch_size]
    y_batch = predict(net, x_batch)
    p = np.argmax(y_batch, axis=1)
    correct_count += np.sum(p == t[i:i + batch_size])

print("Accuracy: ", str(float(correct_count / len(x))))
