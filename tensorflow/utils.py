import tensorflow as tf  # 텐서플로우 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리


def normalize(input_image, input_mask):
    # 모든 원소를 255로 나누어 색상 값을 [0.0, 1.0] 범위로 정규화한다.
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # 마스크에는 색상 값이 아니라 픽셀에 대응되는 클래스의 레이블 번호가 있다.
    # 이 때 마스크 상의 레이블 번호가 1번부터 시작하는데 우리가 정의하는 레이블은 0번부터 시작하므로
    # 모든 픽셀에서 1을 빼준다.
    input_mask -= 1

    return input_image, input_mask


def load_image(datapoint):
    # datapoint['image']는 SymbolicTensor 타입의 데이터.
    # 샘플 이미지와 정답 마스크의 크기를 128 * 128로 변경한다.
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    # 이 시점에 input_image와 input_mask는 (128, 128, 3)의 형태이다.
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


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


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1, model=None):
    if model is None:
        print("Model is not provided.")
        return
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        print('No dataset provided')
