
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, layers
from scipy.io import wavfile
from scipy import signal
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc


if __name__ == '__main__':

    PATH = 'C:\\AI\\Juganu\\dataset'
    labels = ['juganu', 'not_juganu']

    X = []
    Y = []
    corrupted_list = []
    s_b_corrupted_list = []
    for l in labels:
        label_path = os.path.join(PATH, l)
        files = os.listdir(label_path)
        for index, f in enumerate(files):
            sample_rate, audio = wavfile.read(os.path.join(label_path, f))
            if len(audio) > sample_rate*0.5:
                spectrogram = mfcc(audio[0:int(sample_rate*(0.5))], sample_rate)
                spec = spectrogram.T
                if spec.shape == (13, 49):
                    X.append(spec)
                    Y.append(l)
                else:
                    s_b_corrupted_list.append(os.path.join(label_path, f))

            else:
                corrupted_list.append(os.path.join(label_path, f))

    y_arr = np.zeros((len(Y), 2), dtype=np.float)
    for index, l in enumerate(Y):
        if l == 'juganu':
            y_arr[index][0] = 1.
        else:
            y_arr[index][1] = 1

    x__ = np.array(X)
    y__ = np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(x__, y_arr, test_size=0.1, random_state=42)

    with tf.Session() as sess0:
        assert not tf.executing_eagerly()
        model = Sequential()

        model.add(layers.Dense(32, input_shape=X_train.shape[1:], activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit(x=X_train, y=y_train, epochs=10, verbose=1, validation_split=0.1, shuffle=True, batch_size=128)

        model_evaluation = model.evaluate(x=X_test, y=y_test, batch_size=None, verbose=1)

        prediction = model.predict(X_test, batch_size=128, verbose=1)

        # Save the model
        model.save('../output/juganu_speech_recognition.model')
        sess0.close()

