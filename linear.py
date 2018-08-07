import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

a = tf.constant([[1.0, 1.0], [2.0, 2.0]])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.reduce_mean(a, 0))) # 0번째 축을 reduce_mean
print(sess.run(tf.reduce_mean(a, 1))) # 1번째 축을 reduce_mean
print(sess.run(tf.reduce_mean(a))) # 0번째 축을 reduce_mean


x = tf.placeholder(tf.float32) # input
y = tf.placeholder(tf.float32) # output

model = W * x + b # 인공신경망 구성

cost = tf.reduce_mean(tf.square(model - y))

# W와 b를 cost를 minimize로 변경해주고, GradientDescent(기울기로 보정) 한다.
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.05) #learning_rate = 감마
#opt = tf.train.GradientDescentOptimizer(learning_rate = 0.0005) #learning_rate = 감마

train = opt.minimize(cost)

# 반복학습의 수
train_tot = 100
#train_tot = 1000
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

# x_tr = [1, 2, 3]
# y_tr = [1, 2, 3]

x_tr = [1, 2, 3]
y_tr = [1, 2, 30]


#입력데이터를 줄이면, 결과가 나빠진다. 정확도가 떨어진다.
# x_tr = [1, 2]
# y_tr = [1, 2]
#입력데이터를 늘리면, learning_rate = 0.0005를 조정해야 함.
# x_tr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y_tr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# feed_dict를 활용해서 big data를 input/output placeholder에 채워주는 명령어
# train은 _로 해서 무시.
# W와 b가 보정이 완료된다.
for i in range(train_tot):
    error, _ = sess.run([cost, train], feed_dict = {x: x_tr, y: y_tr})
    print(i, 'error = %.3f' % error, 'W = %.3f' %
        sess.run(W), 'b = %.3f' % sess.run(b))

# 학습 후 모델을 테스트 한다.
test = 5
#test = 500000
guess = sess.run(model, feed_dict={x: test})
print('\ntest = ', test, 'guess = %.3f' % guess)

#error가 줄어드는 것은 GradientDescent를 적용해서 그렇다.