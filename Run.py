import numpy as np
import os
import queue
from tensorflow import keras
from scipy.io import wavfile
from python_speech_features import mfcc

import speech_recognition as sr

# r = sr.Recognizer()
# with sr.Microphone() as source:                # use the default microphone as the audio source
#     audio = r.listen(source)
#     try:
#         print(f'You said {r.recognize_google(audio)}')
#     except:
#         print("error")

PATH = 'C:\\AI\\Juganu\\dataset'

labels = ['juganu', 'not_juganu']
model = keras.models.load_model('output/juganu_speech_recognition.model')


def audio_for_test():
    X = queue.Queue()
    corrupted_list = []
    s_b_corrupted_list = []
    for l in labels:
        label_path = os.path.join(PATH, l)
        files = os.listdir(label_path)
        for index, f in enumerate(files):
            sample_rate, audio = wavfile.read(os.path.join(label_path, f))
            if len(audio) > sample_rate * 0.5:

                spectrogram = mfcc(audio[0:int(sample_rate * (0.5))], sample_rate)
                spec = spectrogram.T
                if spec.shape == (13, 49):
                    X.put((f, l, spec))

    return X


if __name__ =='__main__':

    X = audio_for_test()
    while not X.empty():
        (f, l, spec) = X.get()

        p = model.predict(np.array( [spec,]))
        prediction = np.argmax(p)
        if prediction==0:

            print(f'{f} = {l}, the result is: Juganu')
        else:
            print(f'{f} = {l}, the result is: Not Juganu')

    # TODO:
    #  final test is to use the user micerophon
