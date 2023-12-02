from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

piano = np.loadtxt("/Users/ziabkinfamily/Downloads/piano.txt", skiprows=0, unpack=True)
trumpet = np.loadtxt("/Users/ziabkinfamily/Downloads/trumpet.txt", skiprows=0, unpack=True)
fpiano = np.abs(fft(piano))
ftrumpet = np.abs(fft(trumpet))

f0 = 44100
x = fftfreq(len(piano), 1/f0)

plt.figure()
plt.plot(x[:10000], fpiano[:10000])
plt.title("Piano")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.figure()
plt.plot(x[:10000], ftrumpet[:10000])
plt.title("Trumpet")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")