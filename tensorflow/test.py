

import tensorflow as tf
# 리스트를 텐서로 변환
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# 심볼릭 텐서로 변환
symbolic_tensor = tf.convert_to_tensor(tensor)

# 출력
print(type(symbolic_tensor))
print(symbolic_tensor)

