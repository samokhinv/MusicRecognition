import requests
import os
import tempfile
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft


CHUNK_SIZE = 1000


#TODO согласовать с telegram
def getWav(link):
    r = requests.get(link, allow_redirects=True)
    f = tempfile.NamedTemporaryFile()
    open(f, 'w').write(r.content)
    f.close()
    return f


def getFFT(filename):
    """from the wavfile we create an array that contains
        info about all frequences in chunks
        """
    sr, data = wavfile.read(filename)
    if data.shape[1] == 2:
        data = data.sum(axis=1) / 2
    chunks_number = data.len()//CHUNK_SIZE

    results = []
    for i in range(chunks_number):
        chunk = []
        for j in range(CHUNK_SIZE):
            chunk.append(complex(data[i*CHUNK_SIZE + j], 0))
        results.append(fft(chunk))
    return results


#TODO visualisation

def getSpan(borders, item):
    for i in range(borders.len() - 1):
        if item > borders[i] and item < borders[i + 1]:
            return i

def getKeyPoints(fft_results):
    """leave frequencies with highest amplitudes
        in some specific spans"""
    freqs = [40, 80, 120, 200, 300, 400, 500, 700] #TODO choose spans to study
    highscores = [0]*(len(freqs) - 1)
    key_freqs = [0]*(len(freqs) - 1)
    for i in range(freqs[0], freqs[-1]+1):
        index = getSpan(freqs, i)
        A = np.log(fft_results[i].abs() + 1)
        if A > highscores[index]:
            highscores[index] = A
            key_freqs[index] = i
    return key_freqs

#TODO сложна
def creeate_hash()







