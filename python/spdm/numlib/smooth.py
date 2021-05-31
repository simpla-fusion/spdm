
import scipy.signal


def smooth(y, s=None):
    b, a = scipy.signal.butter(1, 0.05)
    if  s == -1:
        pass
    elif s is not None:
        y[s] = scipy.signal.filtfilt(b, a, y[s])
    else:
        y = scipy.signal.filtfilt(b, a, y)

    return y
