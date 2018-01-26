import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from VAE.vanilla_vae.vae_model import VariationalAutoencoder

np.random.seed(0)
tf.set_random_seed(0)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    logs_path='./summary/'
    summary_writer=tf.summary.FileWriter(logdir=logs_path)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            cost,merger_summary_op = vae.partial_fit(batch_xs)
            summary_writer.add_summary(merger_summary_op,epoch*total_batch+i)
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

network_architecture = dict(
        n_hidden_recog_1=500, # 1st layer encoder neurons
        n_hidden_recog_2=500, # 2nd layer encoder neurons
        n_hidden_gener_1=500, # 1st layer decoder neurons
        n_hidden_gener_2=500, # 2nd layer decoder neurons
        n_input=784, # MNIST data input
        n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=75)

print("tensorboard --logdir=PycharmProjects/generative-models/VAE/vanilla_vae/summary")

x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize=(8, 12))
for i in range(3):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()

