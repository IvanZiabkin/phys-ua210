#Problem 1
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from timeit import default_timer as t

hdu_list = astropy.io.fits.open('/Users/ziabkinfamily/Downloads/specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

#Part A
plt.plot(logwave, flux[50, :])
ind = np.argmax(flux[50, :])
print(10**logwave[ind] *0.1) #find peak wavelength
plt.plot(logwave, flux[1080, :])
plt.plot(logwave, flux[2300, :])
plt.title("Spectra for Images 51, 1081, 2301")
plt.xlabel("Log10(Lambda)")
plt.ylabel("Spectrum")

#Parts B, C
norm_constant = np.zeros((1,9713))
means = np.zeros((1,9713))
for i in range(9713):
    row = flux[i, :]
    nc = np.sum(row)
    row = row/nc
    norm_constant[0,i] = nc
    m = np.mean(row)
    means[0,i]=m
    row = row-m
    flux[i, :] = row #flux is now our R matrix

#Part D
start = t()
C = np.transpose(flux).dot(flux)
plt.figure()
evals, evecs = np.linalg.eig(C)
for i in range(5):
    plt.plot(logwave, np.abs(evecs[i, :]))
    plt.title("Covariance Matrix Method (abs)")
    plt.xlabel("Log10(Lambda)")
    plt.ylabel("Eigenvector")
end = t()
print(end-start)

plt.figure()
for i in range(5):
    plt.plot(logwave, evecs[i, :])
    plt.title("Covariance Matrix Method")
    plt.xlabel("Log10(Lambda)")
    plt.ylabel("Eigenvector")

#Part E
start = t()
plt.figure()
(U, w, VT) = np.linalg.svd(flux)
V = np.transpose(VT)
for i in range(5):
    plt.plot(logwave, np.abs(V[i, :]))
    plt.title("SVD Method")
    plt.xlabel("Log10(Lambda)")
    plt.ylabel("Eigenvector")
end = t()
print(end-start)

#Part F
condition = np.max(w)/np.min(w[np.nonzero(w)])
print(condition)


#Part G
plt.figure()
Z = np.matmul(flux, evecs)
for i in [50, 1080, 2300]:
    artificial = (np.matmul(Z, np.transpose(evecs))+means[0,i])*norm_constant[0,i]
    art_plot = artificial[i, :]
    plt.plot(logwave, art_plot)
    plt.title("Perfectly Reconstructed Spectra: 51, 1081, 2301")
    plt.xlabel("Log10(Lambda)")
    plt.ylabel("Artificial Spectrum")


#Part H
plt.figure()
plt.plot(Z[0, :], Z[1, :])
plt.title("Coefficients c0 vs c1")
plt.xlabel("c0")
plt.ylabel("c1")
plt.figure()
plt.plot(Z[0, :], Z[2, :])
plt.title("Coefficients c0 vs c2")
plt.xlabel("c0")
plt.ylabel("c2")

#Part I
plt.figure()
iterations = 0
N = np.array([1, 5, 20, 100, 500, 2000, 3000])
r = np.zeros((7))
for i in N:
    fillZ = np.ones(4000-i+1)
    Z[:, (i-1):4000] = fillZ
    artificial = (np.matmul(Z, np.transpose(evecs))+means[0,0])*norm_constant[0,0]
    r[iterations] = np.mean(((flux-artificial)/(artificial))**2)
plt.plot(N, r)
plt.title("Fractional Squared Residuals, Set 1")
plt.xlabel("N")
plt.ylabel("Fractional Squared Residual")