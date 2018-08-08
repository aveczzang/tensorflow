import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

input = tf.constant([[1., 1., 1., 0., 0.],
                     [0., 1., 1., 1., 0.],
                     [0., 0., 1., 1., 1.],
                     [0., 0., 1., 1., 0.],
                     [0., 1., 1., 0., 0.]])
input = tf.reshape(input, [1, 5, 5, 1])
filter = tf.constant([[1., 0., 1.],
                      [0., 1., 0.],
                      [1., 0., 1.]])
filter = tf.reshape(filter, [3, 3, 1, 1])

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding = 'VALID')
with tf.Session() as sess:
    result = sess.run(op)
    print("convolution result:")
    print(result)

input = tf.constant([[1., 1., 2., 4.],
                     [5., 6., 7., 8.],
                     [3., 2., 1., 0.],
                     [1., 2., 3., 4.]])
input = tf.reshape(input, [1, 4, 4, 1])

op = tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='VALID')
with tf.Session() as sess:
    result = sess.run(op)
    print("pooling result:")
    print(result)
