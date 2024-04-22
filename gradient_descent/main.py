"""
 이 예제는 함수 f(x, y) = x^2 + y^2 + 10 의 극솟값을 경사하강법으로 찾는 예제입니다.
"""
import random

# 초매개변수
# 초매개변수를 조정하여 학습 결과의 차이를 확인할 수 있다, 아래는 바람직한 에폭의 수와 학습률 예시
EPOCH = 10000
LEARNING_RATE = 0.001

# 에폭의 수가 너무 적어 학습이 되지 않는 예시
# EPOCH = 100
# LEARNING_RATE = 0.001

# 학습률이 너무 적어 학습이 거의 되지 않는 예시
# EPOCH = 10000
# LEARNING_RATE = 0.00000001

# 학습률이 너무 커 학습이 거의 되지 않는 예시
# EPOCH = 10000
# LEARNING_RATE = 1


# x에 대한 편미분 = 2x
def partial_derivative_x(x):
    return 2 * x


# y에 대한 편미분 = 20y
def partial_derivative_y(y):
    return 2 * y


x = random.randint(100, 500)  # 100 ~ 500 사이의 임의의 수를 초기 x 값으로 선정
y = random.randint(100, 500)  # 100 ~ 500 사이의 임의의 수를 초기 y 값으로 선정

print(f'초기 x 값 = {x}, 초기 y 값 = {y}, 초기 f(x + y) 값 = {x**2 + y**2 + 10}')

for i in range(EPOCH):
    gradient_x = partial_derivative_x(x)    # 현재 x 좌표에서 x에 대한 기울기
    gradient_y = partial_derivative_y(y)    # 현재 y 좌표에서 y에 대한 기울기
    # 방향을 바꾸기 위해 -1을 곱하여 부호 변환
    difference_x = gradient_x * LEARNING_RATE * -1
    difference_y = gradient_y * LEARNING_RATE * -1
    # 좌표 갱신
    x = x + difference_x
    y = y + difference_y

# 본 예제의 정답 f(x + y) 값은 10이다
print(f'결과 x 값 = {x}, 결과 y 값 = {y}, 결과 f(x + y) 값 = {x**2 + y**2 + 10}')


