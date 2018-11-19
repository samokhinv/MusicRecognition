import os, sox, scipy
import numpy as np
import matplotlib.pyplot as plt

def convert_to_wav(fn):
    base_fn, ext = os.path.splitext(fn)
    tfm = sox.Transformer()
    tfm.remix()
    tfm.set_output_format(encoding="floating-point")
    wav_fn = base_fn + ".wav"
    tfm.build(fn, wav_fn)
    print("{0} converted to wav and saved as {1}".format(fn, wav_fn))
    return wav_fn

def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn);
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    scipy.save(data_fn, fft_features)
    print("FFT of {0} written to {1}".format(fn, data_fn))

import glob


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]

def read_fft(genre_list, base_dir="genres"):
    X = []
    y = []

    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)

        for fn in file_list:
            fft_features = scipy.load(fn)
            X.append(fft_features[:1000])
            y.append(label)

    return np.array(X), np.array(y)

import sklearn
import sys



def plot_confusion_matrix(y_test, y_pred, title):
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cm = cm/cm.sum(axis=1)[:,None]
    plt.clf()
    plt.matshow(cm, fignum=False)
    ax = plt.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    ax.set_xticklabels(genre_list)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.grid(False)

    plt.show()


def classifier(X, y):
    print(X.shape, y.shape)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    clf = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    clf.fit(x_train, y_train)
    test(clf, x_test, y_test)
    return clf


def test(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, "confusion matrix")
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    for i,label in enumerate(genre_list):
        y_label_test = np.equal(y_test, i)
        proba = clf.predict_proba(X_test)
        proba_label = proba[:,i:i+1]
        print(proba_label.T)
        fpr, tpr, roc_thres = sklearn.metrics.roc_curve(y_label_test, proba_label)
        axes= plt.plot(fpr, tpr)
    plt.show()
    test_mfc = create_mfc(convert_to_wav("../PycharmProjects/Shazam/test.mp3"))
    print(test_mfc.shape)


from librosa.feature import mfcc


def write_mfc(ceps, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print("Cepstrum of {0} written to {1}".format(fn, data_fn))


def create_mfc(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ceps = mfcc(y=X, sr=sample_rate, n_mfcc=20).mean(axis=1)
    write_mfc(ceps.T, fn)
    return ceps


def read_ceps(genre_list, base_dir="genres"):
    X, y = [], []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            X.append(ceps)
            y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    if sys.argv[1] == '-fft':
        #for genre in genre_list:
        #    genre_dir = os.path.join("genres", genre, "*.au")
        #    file_list = glob.glob(genre_dir)
        #    print(file_list)
        #    for fn in file_list:
        #        create_fft(convert_to_wav(fn))
        X, y = read_fft(genre_list)
        classifier(X, y)
    if sys.argv[1] == '-mfc':
        for genre in genre_list:
            genre_dir = os.path.join("genres", genre, "*.wav")
            file_list = glob.glob(genre_dir)
            for fn in file_list:
                create_mfc(fn)
        X, y = read_ceps(genre_list)
        classifier(X, y)

