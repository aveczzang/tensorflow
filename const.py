print(2+3)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
sess = tf.Session()  # sess 하나 정의해서 향후 모두 사용

with tf.Session() as sess:
    print(sess.run(x))

# a = tf.constant([[[2, 2], [2, 2], [2,2], [2,2]])
#a = tf.constant([2, 2])  # 1차원 tensor
a = tf.constant([[2, 2]])  # 1차원 tensor
b = tf.constant([[0, 1], [2, 3]]) # 2차원 tensor
x = tf.add(a, b)
y = tf.multiply(a, b)
z = tf.matmul(a, b)  # tesor를 행렬로 변경
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z))
