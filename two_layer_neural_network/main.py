from dataset.mnist import load_mnist
from two_layer_neural_network import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

# 초매개변수
iteration = 1
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

net = TwoLayerNeuralNetwork(input_size=784, hidden_size=100, output_size=10)

for i in range(iteration):
    # train_size 중에서 barch_size만큼의 무작위 인덱스를 선정하여 리스트로 반환한다.
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = net.numerical_gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

print(train_loss_list)