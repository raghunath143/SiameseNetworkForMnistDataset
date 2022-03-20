#!/usr/bin/env python
# coding: utf-8

# # Create a Siamese Network with Triplet Loss in Keras

# # Task 1: Understanding the Approach

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from pca_plotter import PCAPlotter

print('TensorFlow version:', tf.__version__)


# ## Understanding the Approach
# 
# This appraoch is taken from the popular [FaceNet](https://arxiv.org/abs/1503.03832) paper.
# 
# We have a CNN model called `EmbeddingModel`:
# 
# ![CNN](assets/CNN.png)
# 
# We use three images for each training example:
# 1. `person1_image1.jpg` (Anchor Example, represented below in green)
# 2. `person1_image2.jpg` (Positive Example, in blue)
# 3. `person2_image1.jpg` (Negative Example, in red).
# 
# ![Embeddings](assets/embeddings.png)
# 
# 
# ## Siamese Network
# 
# All the three images of an example pass through the model, and we get the three Embeddings: One for the Anchor Example, one for the Positive Example, and one for the Negative Example.
# 
# ![Siamese Network](assets/siamese.png)
# 
# The three instances of the `EmbeddingModel` shown above are not different instances. It's the same, shared model instance - i.e. the parameters are shared, and are updated for all the three paths simultaneously.

# # Task 2: Importing the Data

# In[3]:


(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)


# In[4]:


x_train = np.reshape(x_train, (60000, 784))/255.
x_test = np.reshape(x_test, (10000, 784))/255.
print(x_test.shape)


# # Task 3: Plotting Examples

# In[5]:


def plot_triplet(triplet):
    plt.figure(figsize=(6,2))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        plt.imshow(np.reshape(triplet[i],(28,28)),cmap='binary')
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[6]:


plot_triplet([x_train[0],x_train[1],x_train[2]])


# # Task 4: A Batch of Triplets

# In[11]:


def create_batch(batch_size):
    anchors = np.zeros((batch_size, 784))
    positives = np.zeros((batch_size, 784))
    negatives = np.zeros((batch_size, 784))
    
    for i in range(0, batch_size):
        index = random.randint(0,60000 -1)
        anc = x_train[index]
        y = y_train[index]
        
        indices_for_pos = np.squeeze(np.where(y_train == y))
        indices_for_neg = np.squeeze(np.where(y_train != y))
        
        pos = x_train[indices_for_pos[random.randint(0, len(indices_for_pos)-1)]]
        neg = x_train[indices_for_neg[random.randint(0, len(indices_for_neg)-1)]]
        
        anchors[i] = anc
        positives[i] = pos
        negatives[i] = neg
        
    return [anchors, positives, negatives] 


# In[12]:


triplet = create_batch(1)
plot_triplet(triplet)


# # Task 5: Embedding Model

# In[15]:


emb_dim = 64

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape = (784,)),
    tf.keras.layers.Dense(emb_dim, activation ='sigmoid')
])


# In[16]:


embedding_model.summary()


# In[17]:


example = x_train[0]
example_emb = embedding_model.predict(np.expand_dims(example, axis=0))[0]
print(example_emb)


# # Task 6: Siamese Network

# In[18]:


in_anc = tf.keras.layers.Input(shape=(784,))
in_pos = tf.keras.layers.Input(shape=(784,))
in_neg = tf.keras.layers.Input(shape=(784,))

em_anc = embedding_model(in_anc)
em_pos = embedding_model(in_pos)
em_neg = embedding_model(in_neg)

out = tf.keras.layers.concatenate([em_anc, em_pos, em_neg], axis = 1)

net = tf.keras.models.Model(
 [in_anc, in_pos, in_neg],
 out
)
net.summary()


# # Task 7: Triplet Loss
# 
# A loss function that tries to pull the Embeddings of Anchor and Positive Examples closer, and tries to push the Embeddings of Anchor and Negative Examples away from each other.
# 
# Root mean square difference between Anchor and Positive examples in a batch of N images is:
# $
# \begin{equation}
# d_p = \sqrt{\frac{\sum_{i=0}^{N-1}(f(a_i) - f(p_i))^2}{N}}
# \end{equation}
# $
# 
# Root mean square difference between Anchor and Negative examples in a batch of N images is:
# $
# \begin{equation}
# d_n = \sqrt{\frac{\sum_{i=0}^{N-1}(f(a_i) - f(n_i))^2}{N}}
# \end{equation}
# $
# 
# For each example, we want:
# $
# \begin{equation}
# d_p \leq d_n
# \end{equation}
# $
# 
# Therefore,
# $
# \begin{equation}
# d_p - d_n \leq 0
# \end{equation}
# $
# 
# This condition is quite easily satisfied during the training.
# 
# We will make it non-trivial by adding a margin (alpha):
# $
# \begin{equation}
# d_p - d_n + \alpha \leq 0
# \end{equation}
# $
# 
# Given the condition above, the Triplet Loss L is defined as:
# $
# \begin{equation}
# L = max(d_p - d_n + \alpha, 0)
# \end{equation}
# $

# In[19]:


def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:2*emb_dim], y_pred[:, 2*emb_dim:]
        dp = tf.reduce_mean(tf.square(anc - pos), axis =1)
        dn = tf.reduce_mean(tf.square(anc - neg), axis =1)
        return tf.maximum(dp-dn+alpha, 0.)
    return loss


# # Task 8: Data Generator

# In[20]:


def data_generator(batch_size, emd_dim):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_dim))
        yield x, y


# # Task 9: Model Training

# In[21]:


batch_size = 1024
epochs = 10
steps_per_epoc = int(60000/batch_size)

net.compile(loss = triplet_loss(alpha=0.2,emb_dim=emb_dim), optimizer='adam')

X, Y = x_test[:1000], y_test[:1000]


# In[ ]:


_=net.fit(
    data_generator(batch_size, emb_dim),
    epochs=epochs, steps_per_epoch = steps_per_epoc,
    verbose = False,
    callbacks=[
        PCAPlotter(plt, embedding_model, X, Y)
    ]
)


# In[ ]:




