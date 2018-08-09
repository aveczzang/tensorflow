#기본 코드 추가
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
print(char_arr)

# 'a':0, 'b':1, 'c':2 ...
num_dic = {n: i for i, n in enumerate(char_arr)}
print(num_dic)

seq_data = ['body', 'dial', 'open', 'rank', 'need',
            'wise', 'item', 'jury', 'path', 'ease']
n_input = n_class = 26 # 마지막 글자를 분류 가능한 수. 입력 가능한 글자 
n_stage = 3

#input encoder
def make_batch(seq_data):
    input_batch = [] #3개씩 ont hot encoding된 데이터 10개포함.(10개 seq_data 단어)
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0:-1]] # 앞의 문자 3개를 num_dic 전환
        target = num_dic[seq[3]] #마지막 글자를 num_dic 전환
        input_batch.append(np.eye(26)[input]) # one hot encoding 첫 3개 문자 된다.
        target_batch.append(target)
    print(input_batch)
    print(target_batch)
    return input_batch, target_batch

make_batch(seq_data)

# parameter 설정
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

#one hot encoding 데이터가 input. n_stage : 3, n_input : 26개
X = tf.placeholder(tf.float32, [None, n_stage, n_input]) #weight와 곱해져서 float32 타입으로 한다.
Y = tf.placeholder(tf.int32, [None]) # 정답. 마지막 단어
W = tf.Variable(tf.random_normal([n_hidden, n_class])) #
b = tf.Variable(tf.random_normal([n_class])) # n_class : 26

# RNN Cells and deep RNN Network
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

#output의 3가지 종류 : batch_size, n_stage, n_hidden 중에서
# transpose하게되면 n_stage, batch_size, n_hidden으로 변경
# output[-1]로 n_stage 제거
# 3개의 입력중 LSTM을 통해 마지막 것이 활용가치가 높아 선택하기 위해 처리.
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

#logits=model : 추측값, labels=Y : 정답
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)
for epoch in range(total_epoch):
    _, error = sess.run([optimizer, loss], feed_dict={X: input_batch, Y: target_batch})
    print('epoch: %04d' % epoch, 'error = %.4f' %error)
print()

# RNN model
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
input_batch, target_batch = make_batch(seq_data) # 단어 10개 encoding
guess = sess.run(prediction, feed_dict={X: input_batch, Y: target_batch}) # ANN에 넣어주는 방법

for i, seq in enumerate(seq_data):
    print(seq[0:3], char_arr[guess[i]])
