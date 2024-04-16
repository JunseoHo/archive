# 아래와 같은 텐서 플로우 공식 문서를 참고하였음.
# https://www.tensorflow.org/tutorials/images/segmentation?hl=ko

import tensorflow as tf
import tensorflow_datasets as tfds  # 텐서플로우에서 자주 사용되는 데이터 세트를 쉽게 활용할 수 있도록 만들어진 라이브러리
import pix2pix
# from IPython.display import clear_output
import matplotlib.pyplot as plt
# venv 가상 환경에서 GraphViz를 사용하기 위해서는 아래와 같이 경로를 직접 추가해야 한다.
import os
os.environ["PATH"] += os.pathsep + "C:\\Users\\Work\Desktop\\Workspace\\archive\\tensorflow\\Graphviz\\bin"
# with_info는 데이터 세트의 메타데이터(데이터 세트 설명, 클래스 개수, 이미지 크기 등)의 로드 여부
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


TRAIN_LENGTH = info.splits['train'].num_examples  # 데이터 세트의 샘플 개수
BATCH_SIZE = 64  # 미니배치의 크기
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE  # 3669 // 64 = 57 이므로 57번의 에폭을 수행한다

# dataset의 각 원소에 load_image 함수를 적용한다
# num_parallel_calls는 map 함수의 병렬 처리 여부이며 기본 값은 None이다
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


# 데이터 증강은 과적합을 방지하기 위해 데이터를 임의로 변형하는 작업이다
# 이 예제에서는 단순히 RandomFlip 함수를 통해 수평 반전을 수행한다.
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # 두 함수는 동일한 시드(42)를 사용하므로 무작위 선택 결과가 동일하다.
        # 두 함수는 이미지를 반전하는 레이어를 반환하며 이밎 데이터를 입력으로 받는다.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


train_batches = (
    train_images
    .cache()  # train_images를 캐시한다. 따라서 이 스크립트의 첫 실행에만 작동한다
    # shffle 함수는 아래와 같은 절차로 수행된다
    # 1. 전체 데이터 세트에서 첫 BUFFER_SIZE만큼의 데이터를 버퍼에 적재한다.
    # 2. 버퍼의 데이터를 무작위로 섞는다.
    # 3. 전체 데이터 세트의 다음 BUFFER_SIZE만큼의 데이터들과 버퍼의 데이터를 교환한다.
    #
    # 유의사항 : 위와 같은 알고리즘으로 인해, 전체 데이터 세트가 클래스 순서로 정렬되어 있다면
    # BUFFER_SIZE에 따라 클래스의 순서만 바뀌는 문제가 발생할 수 있다.
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)  # 함수의 반환 객체는 BatchDataset 클래스이다
    .repeat()
    .map(Augment())  # 내부에서 tf.keras.layers.Layer 클래스의 call 함수가 모든 배치의 개별 데이터에 적용된다.
    .prefetch(buffer_size=tf.data.AUTOTUNE))

# 테스트 이미지 세트에서 미니배치를 추출
# 함수의 반환 객체는 BatchDataset 클래스이다
test_batches = test_images.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# 첫번째, 두번째 미니배치로부터 첫번째 데이터(이미지 & 마스크)를 출력한다.
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
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]

# 각 레이어의 출력을 list로 저장한다.
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# 피쳐 추출 모델을 생성한다.
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False  # trainable 속성은 훈련 중에 모델의 매개변수 갱신 여부이다.

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


# 위에서 구현된 함수들을 활용하여 U-Net 모델을 반환한다.
def unet_model(output_channels: int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        print('No dataset provided')


# class DisplayCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     clear_output(wait=True)
#     show_predictions()
#     print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 3
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches)
                          # callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_batches, 3)

