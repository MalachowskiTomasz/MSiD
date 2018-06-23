from __future__ import print_function

import pickle as pkl

import keras
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 10})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

np.set_printoptions(threshold=np.nan)

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
import keras.optimizers

from scipy.ndimage.filters import convolve
# from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 36
epochs = 1000


def hog(image):
    nwin_x = 5
    nwin_y = 5
    B = 7
    (L, C) = np.shape(image)
    H = np.zeros(shape=(nwin_x * nwin_y * B, 1))
    m = np.sqrt(L / 2.0)
    if C is 1:
        raise NotImplementedError
    step_x = np.floor(C / (nwin_x + 1))
    step_y = np.floor(L / (nwin_y + 1))
    cont = 0
    hx = np.array([[1, 0, -1]])
    hy = np.array([[-1], [0], [1]])
    grad_xr = convolve(image, hx, mode='constant', cval=0.0)
    grad_yu = convolve(image, hy, mode='constant', cval=0.0)
    angles = np.arctan2(grad_yu, grad_xr)
    magnit = np.sqrt((grad_yu ** 2 + grad_xr ** 2))
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont += 1
            angles2 = angles[int(n * step_y):int((n + 2) * step_y),
                      int(m * step_x):int((m + 2) * step_x)]
            magnit2 = magnit[int(n * step_y):int((n + 2) * step_y),
                      int(m * step_x):int((m + 2) * step_x)]
            v_angles = angles2.ravel()
            v_magnit = magnit2.ravel()
            bin = 0
            H2 = np.zeros(shape=(B, 1))

            for ang_lim in np.arange(start=-np.pi + 2 * np.pi / B,
                                     stop=np.pi + 2 * np.pi / B,
                                     step=2 * np.pi / B):
                check = v_angles < ang_lim
                v_angles = (v_angles * (~check)) + (check) * 100
                H2[bin] += np.sum(v_magnit * check)
                bin += 1

            H2 = H2 / (np.linalg.norm(H2) + 0.01)
            H[(cont - 1) * B:cont * B] = H2
    return np.squeeze(H.T)


f = open("train.pkl", 'rb')
x_train, y_train = pkl.load(f)


f1 = open("hogged.pkl", 'rb')
x_train = pkl.load(f1)
f1.close()
x_train = np.asarray(x_train)
x_train = x_train.squeeze()
y_train = y_train.transpose()
y_train = y_train[0]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

name = "Bmodel2.h5"
layers = [
    Dense(1000, activation="relu", input_shape=(x_train[0].shape[0],)),
    Dense(500, activation="relu"),
    Dense(100, activation="relu"),
    Dense(36, activation="softmax")
]

f = open(name + "test.pkl", 'wb')
pkl.dump((x_test, y_test), f)
f.close()

model = Sequential()
for layer in layers:
    model.add(layer)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(name, save_best_only=True),
    ReduceLROnPlateau(monitor='val_acc',
                      patience=3,
                      verbose=1,
                      factor=0.5,
                      min_lr=0.00001)
]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test),
#                     callbacks=callbacks)

model.save("model3.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
