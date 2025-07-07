import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from importlib import reload
import os
import time
import numpy as np
from scipy import signal
import FIR_filter_tests.hilbert_coef as hilb
import cupy as cp


cfg = {
    'npzfile': r'C:\Users\ggc\PROYECTOS\UTIMAG\utimag\AutofocusApp\tests\alu_8_taladros.npz',
    'n_elementos': 128,
    'taps': 62,
    'fs': 40,
    'f1': 0.5,
    'f2': 7,
}

bandpass_coef_cpu = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                  pass_zero=False)
bandpass_coef = cp.asarray(bandpass_coef_cpu, dtype=cp.float32)

hilb_coef = cp.asarray(hilb.coef, dtype=cp.float32)

temp = np.load(cfg['npzfile'])
matrix = cp.asarray(temp['matrix'])
cfg['n_samples'] = matrix.shape[-1]

macros = "\n".join([f"#define {key.upper()} {value}" for key, value in cfg.items() if
                    key in ['taps', 'n_ch', 'n_elementos', 'n_samples']])
aux = r'C:\Users\ggc\PROYECTOS\UTIMAG\utimag\OpenCL_Cuda_experimentos\FIR_filter_tests\cupy\\'
code_file = aux + 'filt_kernel.cu'
with open(code_file, encoding='utf-8') as f:
    code = f.read()
#
# module = cp.RawModule(path=code_file)
module = cp.RawModule(code=macros + '\n' + code)
# filt_kernel_2 = module.get_function('_Z13filt_kernel_2PKsPKfPs')
filt_kernel_2 = module.get_function('filt_kernel_2')

# Configurar grid y bloques
block_size = (16, 16)
grid_size = ((cfg['n_elementos'] + block_size[0] - 1) // block_size[0],
             (cfg['n_elementos'] + block_size[1] - 1) // block_size[1])

# Memoria compartida
shared_mem = (cfg['taps'] + 1) * cp.dtype(cp.float32).itemsize

# Llamar al kernel
matrix_filt = cp.zeros((cfg['n_elementos'], cfg['n_elementos'], cfg['n_samples']), dtype=cp.int16)
matrix_imag = cp.zeros_like(matrix_filt)

filt_kernel_2(grid_size, block_size, (matrix, bandpass_coef, matrix_filt), shared_mem=shared_mem)
filt_kernel_2(grid_size, block_size, (matrix_filt, hilb_coef, matrix_imag), shared_mem=shared_mem)

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
        a = cp.asarray(random_data[i])
        cp.cuda.runtime.deviceSynchronize()  # Sincronizar GPU
        t_copy.append(time.perf_counter() - t0)

        # Medir tiempo de ejecución de los kernels
        t0 = time.perf_counter()
        filt_kernel_2(grid_size, block_size, (a, bandpass_coef, matrix_filt))
        filt_kernel_2(grid_size, block_size, (matrix_filt, hilb_coef, matrix_imag))
        cp.cuda.runtime.deviceSynchronize()  # Sincronizar GPU
        t_filter.append(time.perf_counter() - t0)

    print('end')
    return np.around(1000 * np.array(t_copy), 2), np.around(1000 * np.array(t_filter), 2)