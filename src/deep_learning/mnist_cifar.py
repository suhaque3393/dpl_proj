# Use the following Neural network models to classify both MNIST and CIFAR-10.

# 1. MLP with the hidden lay numbers ranging from 1 to 5.
# 2. LeNet
# 3. VGG
# 4. ResNet 


import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# MNIST DATASET

import numpy as np
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pickle

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

print("X_train_full : ",X_train_full.shape)
print("y_train_full : ",y_train_full.shape)
print("X_test : ",X_test.shape)
print("y_test : ",y_test.shape)

# MLP with the hidden lay numbers ranging from 1 to 5.

# Number of hidden layers = 1

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)
print(len(model.layers))

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 2

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 3

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 4

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(100, activation="relu", name="layer4"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy",optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 5

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(100, activation="relu", name="layer4"),
    keras.layers.Dense(100, activation="relu", name="layer5"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Hyper parameter sweep, with 5 hidden layers


mlp_acc_epoch = []
mlp_acc_lr = []
mlp_acc_dense = []

for i in range(1,5):
  for j in range(1,3):
    for k in range(1,5):
      mlp_dense = 10*i
      mlp_lr = 10**-j
      mlp_epoch = 10*k
      print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(mlp_dense, mlp_lr, mlp_epoch))
      model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu", name="layer1"),
        keras.layers.Dense(100, activation="relu", name="layer2"),
        keras.layers.Dense(100, activation="relu", name="layer3"),
        keras.layers.Dense(100, activation="relu", name="layer4"),
        keras.layers.Dense(100, activation="relu", name="layer5"),
        keras.layers.Dense(mlp_dense, activation="softmax")
      ])

      model.summary()
      keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

      opt = keras.optimizers.SGD(learning_rate=mlp_lr)
      model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

      history = model.fit(X_train, y_train, epochs=mlp_epoch, validation_data=(X_valid, y_valid))
      pd.DataFrame(history.history).plot(figsize=(8, 5))
      plt.grid(True)
      plt.gca().set_ylim(0, 1)
      plt.show()
      acc = model.evaluate(X_test, y_test)
      mlp_acc_epoch.append(acc[1])
    plt.plot([10,20,30,40],mlp_acc_epoch, label="accuracy for fixed dense layers and learning rate")
    plt.title("Dense: %d" %mlp_dense + ", Learning Rate: %12.8f" %mlp_lr)
    plt.xlabel('epochs')
    plt.ylabel('accuray')
    plt.show()
    mlp_acc_epoch.clear() 

# LeNet

lenet_acc_epoch = []
lenet_acc_lr = []
lenet_acc_dense = []

X_train_ln = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_ln = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_ln = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

lenet_dense = 10
lenet_lr = 10**-1
lenet_epoch = 10
print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(lenet_dense, lenet_lr, lenet_epoch))
model = keras.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=120, activation='relu'))
model.add(keras.layers.Dense(units=84, activation='relu'))
model.add(keras.layers.Dense(units=lenet_dense, activation = 'softmax'))

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=lenet_lr) #hyper parameter helped with high accuracy results
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train_ln, y_train, epochs=lenet_epoch, validation_data=(X_valid_ln, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test_ln, y_test)
acc[1]

# hyperparameter sweep

lenet_acc_epoch = []
lenet_acc_lr = []
lenet_acc_dense = []

X_train_ln = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_ln = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_ln = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

for i in range(1,5):
  for j in range(1,3):
    for k in range(1,5):
      lenet_dense = 10*i
      lenet_lr = 10**-j
      lenet_epoch = 10*k
      print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(lenet_dense, lenet_lr, lenet_epoch))
      model = keras.Sequential()
      model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)))
      model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
      model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

      model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
      model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(units=120, activation='relu'))
      model.add(keras.layers.Dense(units=84, activation='relu'))
      model.add(keras.layers.Dense(units=lenet_dense, activation = 'softmax'))

      model.summary()
      keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

      opt = keras.optimizers.SGD(learning_rate=lenet_lr) #hyper parameter helped with high accuracy results
      model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

      history = model.fit(X_train_ln, y_train, epochs=lenet_epoch, validation_data=(X_valid_ln, y_valid))
      pd.DataFrame(history.history).plot(figsize=(8, 5))
      plt.grid(True)
      plt.gca().set_ylim(0, 1)
      plt.show()
      acc = model.evaluate(X_test_ln, y_test)
      lenet_acc_epoch.append(acc[1])
    plt.plot([10,20,30,40],lenet_acc_epoch, label="accuracy for fixed dense layers and learning rate")
    plt.title("Dense: %d" %lenet_dense + ", Learning Rate: %12.8f" %lenet_lr)
    plt.xlabel('epochs')
    plt.ylabel('accuray')
    plt.show()
    lenet_acc_epoch.clear()


# VGG

vgg_acc_epoch = []
vgg_acc_lr = []
vgg_acc_dense = []

X_train_vg = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_vg = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_vg = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

vgg_dense = 10*1
vgg_lr = 10**-1
vgg_epoch = 10*1
print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(vgg_dense, vgg_lr, vgg_epoch))
model = keras.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=4096,activation="relu"))
model.add(keras.layers.Dense(units=4096,activation="relu"))
model.add(keras.layers.Dense(units=vgg_dense, activation="softmax"))

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=vgg_lr) #hyper parameter helped with high accuracy results
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train_vg, y_train, epochs=vgg_epoch, validation_data=(X_valid_vg, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test_vg, y_test)
acc[1]

# hyperparameter sweep

vgg_acc_epoch = []
vgg_acc_lr = []
vgg_acc_dense = []

X_train_vg = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_vg = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_vg = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

for i in range(1,5):
  for j in range(1,3):
    for k in range(1,5):
      vgg_dense = 10*i
      vgg_lr = 10**-j
      vgg_epoch = 10*k
      print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(vgg_dense, vgg_lr, vgg_epoch))
      model = keras.Sequential()
      model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)))
      model.add(keras.layers.Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
      model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

      model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

      model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

      model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

      model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
      model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(units=4096,activation="relu"))
      model.add(keras.layers.Dense(units=4096,activation="relu"))
      model.add(keras.layers.Dense(units=vgg_dense, activation="softmax"))

      model.summary()
      keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

      opt = keras.optimizers.SGD(learning_rate=vgg_lr) #hyper parameter helped with high accuracy results
      model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

      history = model.fit(X_train_vg, y_train, epochs=vgg_epoch, validation_data=(X_valid_vg, y_valid))
      pd.DataFrame(history.history).plot(figsize=(8, 5))
      plt.grid(True)
      plt.gca().set_ylim(0, 1)
      plt.show()

    vgg_acc_epoch.append(acc[1])
    plt.plot([10,20,30,40],vgg_acc_epoch, label="accuracy for fixed dense layers and learning rate")
    plt.title("Dense: %d" %vgg_dense + ", Learning Rate: %12.8f" %vgg_lr)
    plt.xlabel('epochs')
    plt.ylabel('accuray')
    plt.show()
    vgg_acc_epoch.clear()


#Resnet

rn_acc_epoch = []
rn_acc_lr = []
rn_acc_dense = []

X_train_rn = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_rn = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_rn = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation="relu",**kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers = [
        keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        self.activation,
        keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
        keras.layers.BatchNormalization()]
    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
          keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
          keras.layers.BatchNormalization()]

  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z)
    skip_Z = inputs
    for layer in self. skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)

rn_dense = 10*1
rn_lr = 10**-1
rn_epoch = 10*1
print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(rn_dense, rn_lr, rn_epoch))
model = keras.models.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[28,28,1], padding="same",use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
  strides = 1 if filters == prev_filters else 2
  model.add(ResidualUnit(filters, strides=strides))
  prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(rn_dense,activation="softmax"))

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=rn_lr) #hyper parameter helped with high accuracy results
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train_rn, y_train, epochs=rn_epoch, validation_data=(X_valid_rn, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test_rn, y_test)
acc[1]

# hyperparameter sweep

rn_acc_epoch = []
rn_acc_lr = []
rn_acc_dense = []

X_train_rn = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_rn = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_rn = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation="relu",**kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers = [
        keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        self.activation,
        keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
        keras.layers.BatchNormalization()]
    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
          keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
          keras.layers.BatchNormalization()]

  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z)
    skip_Z = inputs
    for layer in self. skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)


for i in range(1,5):
  for j in range(1,3):
    for k in range(1,5):
      rn_dense = 10*i
      rn_lr = 10**-j
      rn_epoch = 10*k
      print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(rn_dense, rn_lr, rn_epoch))
      model = keras.models.Sequential()
      model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(28,28,1)))
      model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[28,28,1], padding="same",use_bias=False))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Activation("relu"))
      model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
      prev_filters = 64
      for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
      model.add(keras.layers.GlobalAvgPool2D())
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(rn_dense,activation="softmax"))

      model.summary()
      keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

      opt = keras.optimizers.SGD(learning_rate=rn_lr) #hyper parameter helped with high accuracy results
      model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

      history = model.fit(X_train_rn, y_train, epochs=rn_epoch, validation_data=(X_valid_rn, y_valid))
      pd.DataFrame(history.history).plot(figsize=(8, 5))
      plt.grid(True)
      plt.gca().set_ylim(0, 1)
      plt.show()
      acc = model.evaluate(X_test_ln, y_test)
      rn_acc_epoch.append(acc[1])
    plt.plot([10,20,30,40],rn_acc_epoch, label="accuracy for fixed dense layers and learning rate")
    plt.title("Dense: %d" %rn_dense + ", Learning Rate: %12.8f" %rn_lr)
    plt.xlabel('epochs')
    plt.ylabel('accuray')
    plt.show()
    rn_acc_epoch.clear()

# CIFAR-10

import numpy as np
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import cifar10
# load dataset
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

print("X_train_full : ",X_train_full.shape)
print("y_train_full : ",y_train_full.shape)
print("X_test : ",X_test.shape)
print("y_test : ",y_test.shape)

# MLP with the hidden lay numbers ranging from 1 to 5.

# Number of hidden layers = 1

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)
print(len(model.layers))

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 2

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 3

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 4

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(100, activation="relu", name="layer4"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy",optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Number of hidden layers = 5

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(100, activation="relu", name="layer4"),
    keras.layers.Dense(100, activation="relu", name="layer5"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()
acc = model.evaluate(X_test, y_test)
acc[1]

# Hyper parameter sweep, with 5 hidden layers

mlp_acc_epoch = []
mlp_acc_lr = []
mlp_acc_dense = []

for j in range(1,5):
  mlp_dense = 10
  mlp_lr = 10**-j
  mlp_epoch = 100
  print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(mlp_dense, mlp_lr, mlp_epoch))
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu", name="layer1"),
    keras.layers.Dense(100, activation="relu", name="layer2"),
    keras.layers.Dense(100, activation="relu", name="layer3"),
    keras.layers.Dense(100, activation="relu", name="layer4"),
    keras.layers.Dense(100, activation="relu", name="layer5"),
    keras.layers.Dense(mlp_dense, activation="softmax")
  ])

  model.summary()
  keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

  opt = keras.optimizers.SGD(learning_rate=mlp_lr)
  model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

  history = model.fit(X_train, y_train, epochs=mlp_epoch, validation_data=(X_valid, y_valid))
  pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.show()
  acc = model.evaluate(X_test, y_test)
  mlp_acc_lr.append(acc[1])
  
plt.plot([1,2,3,4],mlp_acc_lr, label="accuracy for fixed dense layers and epoch")
plt.title("Dense: %d" %mlp_dense + ", Epoch: %d" %mlp_epoch)
plt.xlabel('learning rate/10^-n')
plt.ylabel('accuray')
plt.show()
mlp_acc_lr.clear()

# LeNet

lenet_acc_epoch = []
lenet_acc_lr = []
lenet_acc_dense = []

X_train_ln = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test_ln = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_valid_ln = X_valid.reshape(X_valid.shape[0], 32, 32, 3)

lenet_dense = 10
lenet_lr = 10**-1
lenet_epoch = 10
print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(lenet_dense, lenet_lr, lenet_epoch))
model = keras.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(32,32,3)))
model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=120, activation='relu'))
model.add(keras.layers.Dense(units=84, activation='relu'))
model.add(keras.layers.Dense(units=lenet_dense, activation = 'softmax'))

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=lenet_lr) #hyper parameter helped with high accuracy results
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train_ln, y_train, epochs=lenet_epoch, validation_data=(X_valid_ln, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test_ln, y_test)
acc[1]

# hyperparameter sweep

lenet_acc_epoch = []
lenet_acc_lr = []
lenet_acc_dense = []

X_train_ln = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test_ln = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_valid_ln = X_valid.reshape(X_valid.shape[0], 32, 32, 3)

for j in range(1,5):
  lenet_dense = 10
  lenet_lr = 10**-j
  lenet_epoch = 100
  print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(lenet_dense, lenet_lr, lenet_epoch))
  model = keras.Sequential()
  model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(32,32,3)))
  model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
  model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

  model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
  model.add(keras.layers.AveragePooling2D(strides=2,pool_size=(2,2)))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(units=120, activation='relu'))
  model.add(keras.layers.Dense(units=84, activation='relu'))
  model.add(keras.layers.Dense(units=lenet_dense, activation = 'softmax'))

  model.summary()
  keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

  opt = keras.optimizers.SGD(learning_rate=lenet_lr) #hyper parameter helped with high accuracy results
  model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

  history = model.fit(X_train_ln, y_train, epochs=lenet_epoch, validation_data=(X_valid_ln, y_valid))
  pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.show()
  acc = model.evaluate(X_test_ln, y_test)
  lenet_acc_lr.append(acc[1])

plt.plot([1,2,3,4],lenet_acc_lr, label="accuracy for fixed dense layers and epochs")
plt.title("Dense: %d" %lenet_dense + ", Epoch: %d" %lenet_epoch)
plt.xlabel('learning rate/10^-n')
plt.ylabel('accuray')
plt.show()
lenet_acc_lr.clear()


# VGG

vgg_acc_epoch = []
vgg_acc_lr = []
vgg_acc_dense = []

X_train_vg = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test_vg = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_valid_vg = X_valid.reshape(X_valid.shape[0], 32, 32, 3)

vgg_dense = 10
vgg_lr = 10**-1
vgg_epoch = 10
print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(vgg_dense, vgg_lr, vgg_epoch))
model = keras.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(32,32,3)))
model.add(keras.layers.Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=4096,activation="relu"))
model.add(keras.layers.Dense(units=4096,activation="relu"))
model.add(keras.layers.Dense(units=vgg_dense, activation="softmax"))

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=vgg_lr) #hyper parameter helped with high accuracy results
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train_vg, y_train, epochs=vgg_epoch, validation_data=(X_valid_vg, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test_vg, y_test)
acc[1]

# hyperparameter sweep

vgg_acc_epoch = []
vgg_acc_lr = []
vgg_acc_dense = []

X_train_vg = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test_vg = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_valid_vg = X_valid.reshape(X_valid.shape[0], 32, 32, 3)

for j in range(1,4):
  vgg_dense = 10
  vgg_lr = 10**-j
  vgg_epoch = 100
  print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(vgg_dense, vgg_lr, vgg_epoch))
  model = keras.Sequential()
  model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(32,32,3)))
  model.add(keras.layers.Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(units=4096,activation="relu"))
  model.add(keras.layers.Dense(units=4096,activation="relu"))
  model.add(keras.layers.Dense(units=vgg_dense, activation="softmax"))

  model.summary()
  keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

  opt = keras.optimizers.SGD(learning_rate=vgg_lr) #hyper parameter helped with high accuracy results
  model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

  history = model.fit(X_train_vg, y_train, epochs=vgg_epoch, validation_data=(X_valid_vg, y_valid))
  pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.show()
  acc = model.evaluate(X_test_vg, y_test)
  vgg_acc_lr.append(acc[1])

plt.plot([1,2,3],vgg_acc_lr, label="accuracy for fixed dense layers and learning rate")
plt.title("Dense: %d" %vgg_ + ", Epoch: %d" %vgg_epoch)
plt.xlabel('learning rate/10^-n')
plt.ylabel('accuray')
plt.show()
vgg_acc_lr.clear()


#Resnet

rn_acc_epoch = []
rn_acc_lr = []
rn_acc_dense = []

X_train_rn = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test_rn = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_valid_rn = X_valid.reshape(X_valid.shape[0], 32, 32, 3)

class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation="relu",**kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers = [
        keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        self.activation,
        keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
        keras.layers.BatchNormalization()]
    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
          keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
          keras.layers.BatchNormalization()]

  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z)
    skip_Z = inputs
    for layer in self. skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)

rn_dense = 10
rn_lr = 10**-1
rn_epoch = 10
print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(rn_dense, rn_lr, rn_epoch))
model = keras.models.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[32, 32, 3], padding="same",use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
  strides = 1 if filters == prev_filters else 2
  model.add(ResidualUnit(filters, strides=strides))
  prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(rn_dense,activation="softmax"))

model.summary()
keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

opt = keras.optimizers.SGD(learning_rate=rn_lr) #hyper parameter helped with high accuracy results
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(X_train_rn, y_train, epochs=rn_epoch, validation_data=(X_valid_rn, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
acc = model.evaluate(X_test_rn, y_test)
acc[1]

# hyperparameter sweep

rn_acc_epoch = []
rn_acc_lr = []
rn_acc_dense = []

X_train_rn = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test_rn = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_valid_rn = X_valid.reshape(X_valid.shape[0], 32, 32, 3)

class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation="relu",**kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers = [
        keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        self.activation,
        keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
        keras.layers.BatchNormalization()]
    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
          keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
          keras.layers.BatchNormalization()]

  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z)
    skip_Z = inputs
    for layer in self. skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)


# for i in range(1,5):
for j in range(1,5):
  rn_dense = 10
  rn_lr = 10**-j
  rn_epoch = 100
  print("Dense : %2d, learning rate: %5.12f, epochs: %2d" %(rn_dense, rn_lr, rn_epoch))
  model = keras.models.Sequential()
  model.add(keras.layers.ZeroPadding2D(padding=2, input_shape=(32, 32, 3)))
  model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[32, 32, 3], padding="same",use_bias=False))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
  prev_filters = 64
  for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
  model.add(keras.layers.GlobalAvgPool2D())
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(rn_dense,activation="softmax"))

  model.summary()
  keras.utils.plot_model(model, "my_mnist_model.png", show_shapes=True)

  opt = keras.optimizers.SGD(learning_rate=rn_lr) #hyper parameter helped with high accuracy results
  model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

  history = model.fit(X_train_rn, y_train, epochs=rn_epoch, validation_data=(X_valid_rn, y_valid))
  pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1)
  plt.show()
  acc = model.evaluate(X_test_rn, y_test)
  rn_acc_lr.append(acc[1])

plt.plot([1,2,3,4],rn_acc_lr, label="accuracy for fixed dense layers and learning rate")
plt.title("Dense: %d" %rn_dense + ", Epoch: %d" %rn_epoch)
plt.xlabel('learning rate/10^-n')
plt.ylabel('accuray')
plt.show()
rn_acc_lr.clear()