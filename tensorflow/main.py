# 아래와 같은 텐서 플로우 공식 문서를 참고하였음.
# https://www.tensorflow.org/tutorials/images/segmentation?hl=ko
import sys

import numpy as np

from utils import *
from keras import metrics
import pix2pix
import tensorflow as tf  # 텐서플로우 라이브러리
import tensorflow_datasets as tfds  # 텐서플로우에서 자주 사용되는 데이터 세트를 쉽게 활용할 수 있도록 만들어진 라이브러리

import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리

# venv 가상 환경에서 GraphViz를 사용하기 위해서는 아래와 같이 PATH 환경변수에 경로를 직접 추가해야 한다.
import os

os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.realpath(__file__)) + "Graphviz\\bin"

# with_info는 데이터 세트의 메타데이터(데이터 세트 설명, 클래스 개수, 이미지 크기 등)의 반환 여부.
# 본 예제에서는 True로 지정하여 메타데이터를 info 변수에 함께 반환 받는다.
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples  # 훈련 데이터 세트의 샘플 개수, 본 예제에서는 3669이다.
BATCH_SIZE = 64  # 배치 당 샘플의 개수.
SHUFFLE_BUFFER_SIZE = 4000  # shuffle 함수로 데이터를 무작위로 섞을 때 사용될 변수.
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE  # '3669 // 64 = 57' 이므로 매 에폭마다 57번의 역전파 학습을 수행한다.

# dataset의 각 원소에 load_image 함수를 적용한다.
# num_parallel_calls는 map 함수의 병렬 처리 여부이며 기본 값은 None이다.
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# 훈련 배치 데이터를 정의한다.
train_batches = (
    train_images
    .cache()  # train_images를 캐시한다. 따라서 이 스크립트의 첫 실행에만 작동한다
    # shffle 함수는 아래와 같은 절차로 수행된다
    # 1. 전체 데이터 세트에서 첫 BUFFER_SIZE만큼의 데이터를 버퍼에 적재한다.
    # 2. 버퍼의 데이터를 무작위로 섞는다.
    # 3. 전체 데이터 세트의 다음 BUFFER_SIZE만큼의 데이터들과 버퍼의 데이터를 교환한다.
    #
    # 유의사항 : 위와 같은 알고리즘으로 인해, 전체 데이터 세트가 클래스 순서로 정렬되어 있다면
    # BUFFER_SIZE에 따라 이미지가 섞이지 않고 클래스의 순서만 바뀌는 문제가 발생할 수 있다.
    # 따라서 대체로 성능이 떨어지지 않는 선에서 전체 데이터 세트의 크기보다 버퍼의 크기를 크게 잡는다.
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .batch(BATCH_SIZE)  # 함수의 반환 객체는 BatchDataset 클래스이다
    .repeat()
    .map(Augment())  # 내부에서 tf.keras.layers.Layer 클래스의 call 함수가 모든 배치의 개별 데이터에 적용된다.
    .prefetch(buffer_size=tf.data.AUTOTUNE))

# 테스트 이미지 세트에서 미니배치를 추출
# 함수의 반환 객체는 BatchDataset 클래스이다
test_batches = test_images.batch(BATCH_SIZE)

# 첫번째, 두번째 미니배치로부터 첫번째 데이터(이미지 & 마스크)를 출력한다.
# 아래 코드는 디버그 전용으로 사용되므로 기본적으로는 주석처리
"""
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])
"""

# MobileNetV2은 TensorFlow에서 제공하는 사전 학습된 인코더이다.
# include_top은 신경망의 입력층에 있는 전연결 레이어 포함 여부이다.
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# 아래 레이어 이름은 base_model이 포함하고 있는 각 레이어의 기본 이름이다.
# 따라서 수정하면 예외가 발생한다.
# 각 레이어마다 이미지의 크기에 대한 차원축소가 이루어지지만 채널의 수는 증가한다.
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]

# 위에서 정의한 layer_names에 해당하는 레이어의 출력들을 list로 저장한다.
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# MobileNetV2을 이용한 인코딩 모델 객체를 생성한다.
# down_stack에서는 수축 경로의 작업이 수행된다.
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False  # trainable 속성은 훈련 중에 모델의 매개변수 갱신 여부이다.

# up_stack에서는 확장 경로의 작업이 수행된다.
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


# 위에서 구현된 함수들을 활용하여 U-Net 모델을 반환한다.
def unet_model(output_channels: int):
    # (128, 128, 3) 차원의 데이터를 입력으로 받는다.
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # 다운 샘플링
    skips = down_stack(inputs)  # 각 레이어에서 반환된 피쳐 맵을 리스트로 반환한다.
    x = skips[-1]  # 그 중 마지막을 변수 x에 저장한다
    skips = reversed(skips[:-1])  # 마지막으로 전체 피쳐 맵 리스트를 반전한다, 이는 아래에서 skip connection 구축에 사용된다.

    # 업 샘플링 및 스킵 연결 구축
    for up, skip in zip(up_stack, skips):
        x = up(x)  # 현재 피쳐맵을 업샘플링한다.
        concat = tf.keras.layers.Concatenate()  # 연결 레이어를 정의한다.
        x = concat([x, skip])  # 대응되는 다운샘플링 레이어의 피쳐맵과 현재 레이어에서 업샘플링된 피쳐맵을 연결한다.

    # 출력 레이어, 64x64까지 업샘플링된 이미지를 완전히 복원한다.
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  # 64x64 -> 128x128
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def my_precision(y_true, y_pred):
    print(y_true, y_pred)
    # 모델의 예측값을 이진 분류로 변환합니다.
    y_pred_binary = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
    # True Positive 계산: 모델이 True로 예측했고 실제 값도 True인 경우
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)), tf.float32))
    # False Positive 계산: 모델이 True로 예측했지만 실제 값은 False인 경우
    false_positives = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)), tf.float32))

    # 정밀도 계산
    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())

    return precision


OUTPUT_CLASSES = 3  # 분류할 클래스의 개수
model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[my_precision]
)
tf.keras.utils.plot_model(model, show_shapes=True)

# 초매개변수 선언
EPOCHS = 1  # 에폭 수
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

# 이제 fit 함수를 통해 모델을 학습한다.
# fit 함수의 callback 속성을 이용하면 매 학습마다 실행될 콜백 함수를 지정할 수 있다.
# (본 스크립트에서는 제외되어 있다.)

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches)

loss = model_history.history['loss']  # 매 에폭이 끝날 때마다 측정된 학습 데이터에 대한 손실 함수 값
val_loss = model_history.history['val_loss']  # 매 에폭이 끝날 때마다 측정된 검증 데이터에 대한 손실 함수 값

# 데이터 시각화
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_batches, 3, model)
