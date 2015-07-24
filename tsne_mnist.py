from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from tsne import bh_sne
import matplotlib.pyplot as plt
import numpy as np

mnist = load_digits()

mnist_tsne = bh_sne(mnist.data, perplexity = 30, theta = 0.15)
mnist_pca = PCA().fit_transform(mnist.data)

plt.figure(figsize=(15, 5))
plt.subplot(121)

plt.scatter(mnist_pca[:, 0], mnist_pca[:, 1], c = mnist.target)
plt.subplot(122)
plt.scatter(mnist_tsne[:, 0], mnist_tsne[:, 1], c = mnist.target)
plt.show()
