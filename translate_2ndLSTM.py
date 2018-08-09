import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나현무놀이금소녀부키스여릎사랑피행']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)
n_class = n_input = dic_len

# seq0 : wood, seq1 : 나무
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑'],
            ['cash', '현금'], ['tour', '여행'],
            ['skin', '피부'], ['knee', '무릎']]

# one-hot encoding function
def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch

learning_rate = 0.01
n_hidden = 128
#n_hidden = 3
total_epoch = 100
#total_epoch = 10000

# Seq2Seq RNN의 encoder, decoder
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])

# output : [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5) # 50%
    #enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.1) # 10% 결과가 좋지 않다.

    enc_cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([enc_cell, enc_cell2])
    
    outputs, enc_states = tf.nn.dynamic_rnn(multi_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5) # 50%
    #dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.1) # 10% 결과가 좋지 않다.
    
    dec_cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([dec_cell, dec_cell2])

    outputs, dec_states = tf.nn.dynamic_rnn(multi_cell, dec_input, initial_state=enc_states, dtype=tf.float32)



model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})
    print('epoch: %04d' % epoch, 'cost: %.4f' % loss)

def translate(word):
    seq = [word, 'PP']

    input_batch, output_batch, target_batch = make_batch([seq])
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction, feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})

    decoded = [char_arr[i] for i in result[0]]
    translated = decoded[0] + decoded[1]
    return translated

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))
print('wide ->', translate('wide'))
print('gate ->', translate('gate'))
print()
print('cash ->', translate('cash'))
print('cart ->', translate('cart'))
print('true ->', translate('true'))
