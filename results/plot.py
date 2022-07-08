from matplotlib import pyplot as plt

from trial_1 import mnist_ffn_scores, mnist_cnn_scores

mnist_ffn_loss = [scores[0] for scores in mnist_ffn_scores]
mnist_ffn_acc = [scores[1] for scores in mnist_ffn_scores]

mnist_cnn_loss = [scores[0] for scores in mnist_cnn_scores]
mnist_cnn_acc = [scores[1] for scores in mnist_cnn_scores]

x = [i for i in range(len(mnist_ffn_loss))]

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, mnist_ffn_loss, label='mnist_ffn_loss')
axs[0].plot(x, mnist_cnn_loss, label='mnist_cnn_loss')
axs[1].plot(x, mnist_ffn_acc, label='mnist_ffn_acc')
axs[1].plot(x, mnist_cnn_acc, label='mnist_cnn_acc')

axs[0].legend()
axs[0].set_xlabel('global update index')
axs[0].set_ylabel('loss')
axs[1].legend()
axs[1].set_xlabel('global update index')
axs[1].set_ylabel('accuracy')

plt.show()