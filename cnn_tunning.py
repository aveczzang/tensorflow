#기본 코드 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#MNIST 데이터 가져오기
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

# weight initialization with truncated normal distribution
def weight_variable(shape):
     # 이 부분은 random_normal, random_uniform과 같이 변경해서 값이 잘 나오는 것으로 선택
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# initialize bias with 0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # shape이 들어 오는 것 만큼 0.1만큼 tensor를 만들고 Variable로 전환
    #initial = tf.Variable(tf.zeros(shape) + 0.1, shpae=shape)
    return tf.Variable(initial)

# convolution with common setting
def conv2d(x, W):
    # stride는 하나 오른쪽 하나 아래.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling with common setting
def max_pool_2x2(x):
    # input x, winodws size 2 * 2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
      #strides=[1, 2, 2, 1], padding='SAME') # SAME : padding 사용
      strides=[1, 2, 2, 1], padding='VALID') # VALID : padding 사용 안함.

# input image reshaping to 4D tensor for CNN
# format: [batch, height, width, channels]
x = tf.placeholder(tf.float32, shape=[None, 784]) # input neuron : 784
x_image = tf.reshape(x, [-1, 28, 28, 1]) # convolution시 reshape 필수

# Convolutional layer 1
# first CNN layer: CONV -> RELU -> POOL
# We use 5x5 patch, accept 1 channel, and produce 32.
# input image : 1개. 생성 이미지 32개. 5*5 필터가 32개 생성.
# CNN에서는 필터가 가중치가 있다.(variable) 필터값들이 학습이 된다.
W_conv1 = weight_variable([5, 5, 1, 2]) # 필터. 5 * 5, 1 : input channel, 32 : output channel
b_conv1 = bias_variable([2]) # 32개 features
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
# second CNN layer: CONV -> RELU -> POOL
# We use 5x5 patch, accept 32 channel, and produce 64.
W_conv2 = weight_variable([5, 5, 2, 64]) # 필터, 5*5, 32개 input, 64개 필터
b_conv2 = bias_variable([64]) # 64개 features
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 3단 perceptron
# fc(fully connected). fully connected layer with 1024 neurons: FC -> RELU
# hidden layer : 1024 neurons
# Images are reduced to 7x7 and reshaped.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # 1차원 배열로 reshape. dimention은 -1로 하나 줄임.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # hidden neurons의 output

# neuron dropout to avoid overfitting
# 'keep' contains keep rate
keep = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep)
# overfitting(과적합) fitting을 너무 심하게 하거나, 러닝을 심하게 함.
# h_fc1 은 neuron의 수, keep 은 keep할 %
# dropout : neuron 몇개를 버린다.

# readout layer using softmax regression
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 예를 들면, 10개중에 큰 거 하나를 취함.

# loss function and optimizer
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 정답 output
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), 1)) #error function. cross 엔트로피
opt = tf.train.AdamOptimizer(0.001).minimize(loss)

# accuracy calculation for printing
right = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) # argmax : 1번째 축으로 몇번째가 제일 큰 가?
acc = tf.reduce_mean(tf.cast(right, tf.float32))

# session run
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    sess.run(opt, feed_dict={x: batch[0], y_: batch[1], keep: 0.5}) #학습시 keeps은 0.5로 한다.
    if i % 100 == 0:
        check = sess.run(acc, feed_dict={x:batch[0], y_: batch[1], keep: 1.0})
        print("step %d, training accuracy %.2f" % (i, check))
 
# model accuracy with MNIST test set
images = mnist.test.images # 10,000개 이미지
labels = mnist.test.labels
final = sess.run(acc, feed_dict={x: images, y_: labels, keep: 1.0})
print("final test accuracy %g" % final)
