import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

# traninst set : 55,000개
# 인공신경망 구성 (model creation) synapse : 7840개
x = tf.placeholder(tf.float32, [None, 784]) # input neuron : 784개(28 * 28)
W = tf.Variable(tf.zeros([784, 10])) # synapse : 7,840개
b = tf.Variable(tf.zeros([10])) # bias
y = tf.nn.softmax(tf.matmul(x, W) + b) # 추측값

# loss function and optimizer
ans = tf.placeholder(tf.float32, [None, 10]) # output neuron 10개 (0 ~ 9)
loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(y), 1))
opt = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# session creation
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def test():
    # 0에서 시작해서 1개만, 그러므로 0. 0부터 784개 픽셀.
    x_train = mnist.test.images[0:1, 0:784]
    answer = sess.run(y, feed_dict = {x: x_train}) # y : 예측값. input은 x.
    print('\ny vector is', answer)
    print('my guess is', answer.argmax()) # argmax는 제일 큰 값이 몇번째 있는지 확인
    # argmax는 배열에서 가장 큰 숫자가 몇번째 있는지 구한다. 가장 큰 숫자를 구하지 않는다.

train_tot = 1000
batch_size = 100

test()
for i in range(train_tot):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    error, _ = sess.run([loss, opt], feed_dict = {x: batch_xs, ans: batch_ys})
    if i % 100 == 0:
        print('batch', i, 'error = %.3f' % error)
test()

correct = tf.equal(tf.argmax(y, 1), tf.argmax(ans, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
images = mnist.test.images
labels = mnist.test.labels
print('\nmodel accuracy:', sess.run(accuracy, feed_dict={x: images, ans: labels}))
