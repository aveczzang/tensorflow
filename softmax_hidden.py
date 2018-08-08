import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

# test set : 10000개
# traninst set : 55,000개
# 인공신경망 구성 (model creation) synapse : 7840개
x = tf.placeholder(tf.float32, [None, 784]) # input neuron : 784개(28 * 28)
W1 = tf.Variable(tf.random_normal([784, 300])) # synapse : 7,840개
b1 = tf.Variable(tf.zeros([300])) # bias
y1 = tf.nn.softmax(tf.matmul(x, W1) + b1) # 추측값 

W2 = tf.Variable(tf.random_normal([300, 10])) # synapse : 7,840개
b2 = tf.Variable(tf.zeros([10])) # bias

# random_normal을 사용하게 되면, 평균이 줄어든다. 정확도가 떨어짐.
# W = tf.Variable(tf.random_normal([784, 10])) # synapse : 7,840개
# b = tf.Variable(tf.random_normal([10])) # bias
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2) # 추측값

# loss function and optimizer
ans = tf.placeholder(tf.float32, [None, 10]) # output neuron 10개 (0 ~ 9)
#loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(y), 1))
loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(y2), 1))
#opt = tf.train.GradientDescentOptimizer(0.2).minimize(loss) # running rate
# opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss) # running rate
opt = tf.train.AdamOptimizer().minimize(loss) # running rate

# session creation
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def test():
    # 0에서 시작해서 1개만, 그러므로 0. 0부터 784개 픽셀.
    #x_train = mnist.test.images[0:1, 0:784]
    x_train = mnist.test.images[0:1]
    #answer = sess.run(y, feed_dict = {x: x_train}) # y : 예측값. input은 x.
    answer = sess.run(y2, feed_dict = {x: x_train}) # y : 예측값. input은 x.
    print('\ny vector is', answer)
    print('my guess is', answer.argmax()) # argmax는 제일 큰 값이 몇번째 있는지 확인
    # argmax는 배열에서 가장 큰 숫자가 몇번째 있는지 구한다. 가장 큰 숫자를 구하지 않는다.

train_tot = 10000 # running rate을 높이면 train_tot 를 낮춘다.
batch_size = 100
#batch_size = 50

test()
for i in range(train_tot): # 1000번 반복 수업. 100개씩 batch job
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)

    # opt는 _로 사용하지 않음. loss만 에러로 사용.
    error, _ = sess.run([loss, opt], feed_dict = \
        {x: batch_xs, ans: batch_ys})
    if i % 100 == 0: # 100번째 마다 출력
        print('batch', i, 'error = %.3f' % error)
test()

# 10,000개 test set으로 테스트
# y는 추측값, asn는 정답, tf.argmax(y, 1) 에서 1은 세로 축(label). 0번째 축은 10000개중 첫번째 의미.
#correct = tf.equal(tf.argmax(y, 1), tf.argmax(ans, 1)) # 가장 큰 값이 가지고 있는 첨자 확인
correct = tf.equal(tf.argmax(y2, 1), tf.argmax(ans, 1)) # 가장 큰 값이 가지고 있는 첨자 확인
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
images = mnist.test.images
labels = mnist.test.labels
print('\nmodel accuracy:', sess.run(accuracy, feed_dict={x: images, ans: labels}))
