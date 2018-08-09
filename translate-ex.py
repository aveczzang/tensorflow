import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

print('char_arr:\n', char_arr)
print('num_dic:\n', num_dic)
print('number of letters in our alphabet:', dic_len, '\n')

# seq0 : wood, seq1 : 나무
seq_data = [['wood', '나무'], ['love', '사랑']]

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        print('word = ', seq)
        print('input:', input)
        print('output:', output)
        print('target', target)

        input_one = np.eye(dic_len)[input]
        output_one = np.eye(dic_len)[output]

        print('input_one\n', input_one)
        print('output_one\n', output_one)
        print()

        input_batch.append(input_one)
        output_batch.append(output_one)
        target_batch.append(target)
    print('=== final result ===')
    print('input_batch:\n', input_batch)
    print('output_batch:\n', output_batch)
    print('target_batch:\n', target_batch)

    return input_batch, output_batch, target_batch
sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_batch(seq_data)


