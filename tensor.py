import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 3]) # input size 구성, 아래 부분에서 input 2개 설정


#W = tf.Variable([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]) # weight를 초기값 설정
#b = tf.Variable([0.2, 0.2]) # bias 초기값 설정

# random_normal을 활용해서 weight와 bias 초기값 설정
W = tf.Variable(tf.random_normal([3, 2], 0, 10))
b = tf.Variable(tf.random_normal([2, 1], 0, 10))

output = tf.matmul(x, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input = [[1, 2, 3], [4, 5, 6]] # input 2개 설정

print("input : ", input) # 단순한 python 리스트 이므로 그대로 사용
print("W : ", sess.run(W)) # 항상 session.run을 통해서 출력
print("b : ", sess.run(b)) # 항상 session.run을 통해서 출력

print("output : ", sess.run(output, feed_dict={x: input}))
# ANN을 구동하는 방법
# output을 지정하고, ANN에 input을 지정(feed_dict) placeholde를 찾아서 input을 지정해서
# placeholde x에 지정해서 ANN을 구성한다.

print("shape of W: ", W.get_shape())
print("shape of b: ", b.get_shape())
print("shape of x: ", x.get_shape())
print("shape of output: ", output.get_shape())

# tensor가 적용되지 않아서 에러가 발생한다.
# python array이므로 numpy를 활용
#print("shape of input: ", input.get_shape())
import numpy as np
print("shape of input: ", np.shape(input))