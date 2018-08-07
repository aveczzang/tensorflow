import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#T, F = 1., -1.
T = 1.0
F = 1.0
bias = 1.0

# 문제지 input 2개
train_in = [[T, T, bias], [T, F, bias], [F, T, bias], [F, F, bias]]
# 정답지 AND function
train_out = [[T], [F], [F], [F]]
# 정답지 OR function
#train_out = [[T], [T], [T], [F]]

#문제지 input 3개
# train_in = [[T, T, T, bias], [T, T, F, bias], [T, F, T, bias], [T, F, F, bias],
#             [F, T, T, bias], [F, T, F, bias], [F, F, T, bias], [F, F, F, bias]]
#정답지 AND function
# train_out = [[T], [F], [F], [F], [F], [F], [F], [F]]

# train_in = [[T, T, bias],  [T, F, bias],  [F, T, bias],  [F, F, bias]]
# train_out = [[F], [T], [T], [F]]

#weight 보정
w = tf.Variable(tf.random_normal([3, 1])) # 3개인 이유(w1, w2, w3)
#w = tf.Variable(tf.random_normal([4, 1])) # 4개인 이유(w1, w2, w3, w4)

#step(1) = 1, step(-0.3) = -1, step(0) = 1
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)

output = step(tf.matmul(train_in, w))
error = tf.subtract(train_out, output)
mse = tf.reduce_mean(tf.square(error)) #mse : mean square error

# train_in : 4행 3열 tensor, error : 4행 1열
# train_in  * error가 이뤄지지 않으므로 train_in(4*3)을 transpose하면 3*4가 된다.
# (3*4) * (4*1) = 3 * 1 tensor
delta = tf.matmul(train_in, error, transpose_a = True)
train = tf.assign(w, tf.add(w, delta))

#tensorflow Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0
epoch, max_epochs = 0, 100

def test():
    print('\nweight/bias\n', sess.run(w))
    print('output\n', sess.run(output))
    print('mse : ', sess.run(mse), '\n')

#main session
test()
# epoch : 현재 수업 수, err는 현재 에러, target은 원하는 error값.
# error가 없을 때까지 반복
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch : ', epoch, 'mse : ', err)

test()