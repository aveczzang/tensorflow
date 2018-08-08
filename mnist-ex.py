from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)
images = mnist.test.images
labels = mnist.test.labels

# original data
images = images.reshape([-1, 28, 28])  # 2차원 tensor
print(images.shape)
print(images[0])

# label
print(labels.shape)
print(labels[0])
print(labels[1])
