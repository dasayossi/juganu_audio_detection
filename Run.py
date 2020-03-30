import os
import numpy as np
from tensorflow import keras
from scipy.io import wavfile
from python_speech_features import mfcc
import speech_recognition as sr
from time import sleep
import subprocess
import wave


labels = ['juganu', 'not_juganu']
PATH = 'C:\\AI\\Juganu\\dataset'
sounds_folder = 'sounds'


def empty_sounds_folder():
    if not os.path.isdir(sounds_folder):
        os.mkdir(sounds_folder)
    else:
        filelist = [f for f in os.listdir(sounds_folder) if f.endswith(".wav")]
        for f in filelist:
            os.remove(os.path.join(sounds_folder, f))

def get_center(sound_len, max_index, sample_rate):
    delta = int(sample_rate * 0.4)
    if max_index - delta < 0:
        return delta
    if max_index + delta > sound_len:
        return sound_len - delta

    return max_index


if __name__ == '__main__':

    model = keras.models.load_model('output/juganu_speech_recognition.model', compile=True)

    r = sr.Recognizer()
    with sr.Microphone() as source:  # use the default microphone as the audio source
        counter = 0
        empty_sounds_folder()
        while True:
            print("Please say any word...")
            audio = r.listen(source)
            sound_path = os.path.join(sounds_folder, f'sound_{counter}.wav')
            sound16_path = os.path.join(sounds_folder, f'sound16_{counter}.wav')
            obj = wave.open(sound_path, 'w')
            obj.setnchannels(1)  # mono
            obj.setsampwidth(2)
            obj.setframerate(audio.sample_rate)
            obj.writeframesraw(audio.get_wav_data())
            obj.close()
            sleep(0.1)  # To ensure the writing
            subprocess.run(
                f'"ffmpeg/ffmpeg.exe" -y -i '
                f'{sound_path} -ar 16000 {sound16_path}',
                shell=True)
            sleep(0.1)  # To ensure the conversion
            sample_rate, sound = wavfile.read(sound16_path)
            counter += 1
            sound_len = len(sound)
            if sound_len > sample_rate*0.5:

                max_index = np.argmax(sound)
                center = get_center(sound_len, max_index, sample_rate)
                spectrogram = mfcc(np.array(sound[center - int(sample_rate * 0.25):center + int(sample_rate * 0.25)]), sample_rate)
                spec = spectrogram.T

                # Get Prediction
                p = model.predict(np.array([spec, ]))
                prediction = np.argmax(p)
                if prediction == 0:

                    print('You said Juganu')
                else:
                    print("You didn't say Juganu")
            else:
                print("Corrupted record")
