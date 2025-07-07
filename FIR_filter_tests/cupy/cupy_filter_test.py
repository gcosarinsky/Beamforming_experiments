import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from importlib import reload
import os
import time
import numpy as np
from scipy import signal
from FIR_filter_tests import hilbert_coef as hilb
import cupy as cp
import cupyx.scipy.ndimage as ndi

cfg = {
    'npzfile': r'C:\Users\ggc\PROYECTOS\UTIMAG\utimag\AutofocusApp\tests\alu_8_taladros.npz',
    'n_elementos': 128,
    'taps': 62,
    'fs': 40,
    'f1': 0.5,
    'f2': 7,
}

bandpass_coef = cp.asarray(
    signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']], pass_zero=False))
hilb_coef = cp.asarray(hilb.coef)

temp = np.load(cfg['npzfile'])
matrix = cp.asarray(temp['matrix'])
cfg['n_samples'] = matrix.shape[-1]

matrix_filt = ndi.convolve1d(matrix, bandpass_coef, axis=-1, mode='constant')
matrix_imag = ndi.convolve1d(matrix_filt, hilb_coef, axis=-1, mode='constant')

i = 0
fig, ax = plt.subplots()
ax.plot(matrix[i, 0, :].get(), label='raw')
ax.plot(matrix_filt[i, 0, :].get(), label='filt')
ax.plot(matrix_imag[i, 0, :].get(), label='imag')


def fun(n=10):
    t_copy = []
    t_filter = []
    random_data = (1000 * np.random.random((n,) + matrix.shape)).astype(np.int16)

    print('start')
    for i in range(n):
        # Medir tiempo de copia
        t0 = time.perf_counter()
        gpu_data = cp.asarray(random_data[i])
        cp.cuda.runtime.deviceSynchronize()  # Sincronizar GPU
        t_copy.append(time.perf_counter() - t0)

        # Medir tiempo de ejecución de los filtros
        t0 = time.perf_counter()
        filtered_data = ndi.convolve1d(gpu_data, bandpass_coef, axis=-1, mode='constant')
        imag_data = ndi.convolve1d(filtered_data, hilb_coef, axis=-1, mode='constant')
        cp.cuda.runtime.deviceSynchronize()  # Sincronizar GPU
        t_filter.append(time.perf_counter() - t0)

        # print(f"Iteración {i + 1}/{n}", end="\r", flush=True)

    print('end')
    return 1000 * np.around(t_copy, 2), 1000 * np.around(t_filter, 2)