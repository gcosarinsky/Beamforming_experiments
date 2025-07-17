import time
import numpy as np
import cupy as cp
from scipy.io import loadmat
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bf_tools import KernelParameters
import FIR_filter_tests.hilbert_coef as hilb

data_path = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\MUST/matlab/pruebas/pwi_acq_25angles.mat'
data = loadmat(data_path)
matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)
angles = data['angles']

cfg = {
    'fs': 62.5,
    'c1': 6.3,
    'pitch': 0.5,
    'n_ch': 128,
    'n_elementos': 128,
    'n_angles': angles.size,
    'n_batch': 1,
    'f1': 2.,
    'f2': 8.,
    'taps': 62,
    'bfd': 1,
    'x_step': 0.2,
    'z_step': 0.2,
    'x0_roi': -20.,
    'z0_roi': 1.,
    'nx': 224,
    'nz': 256,
    'n_samples': matrix.shape[-1],
    'matrix_shape': matrix.shape,
    't_start': 0.
}

cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2

# Generar coeficientes del filtro FIR
bandpass_coef_cpu = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                                  pass_zero=False)
bandpass_coef = cp.asarray(bandpass_coef_cpu, dtype=cp.float32)
hilb_coef = cp.asarray(hilb.coef, dtype=cp.float32)

# Crear objeto KernelParameters
params = KernelParameters(cfg)
int_params = cp.asarray(params.get_int_array(), dtype=cp.int32)
float_params = cp.asarray(params.get_float_array(), dtype=cp.float32)

# Cargar código CUDA
codepath = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\Beamforming\Cuda\pwi_params_array\\'
with open(codepath + r'pwi_kernels.cu', encoding='utf-8') as f:
       code = f.read()
with open(codepath + r'pwi_batch_kernels.cu', encoding='utf-8') as f:
       code += f.read()

module = cp.RawModule(code=code, options=('--use_fast_math',))
filt_kernel = module.get_function('filt_batch_kernel')
pwi_kernel = module.get_function('pwi_batch_1pix_per_thread')

# replicar la matriz de datos para crear un batch
matrix_batch = np.repeat(np.expand_dims(matrix, axis=0), cfg['n_batch'], axis=0)
# arrays gpu
matrix_batch_gpu = cp.asarray(matrix_batch, dtype=cp.int16)
matrix_filt_gpu = cp.zeros_like(matrix_batch_gpu)
matrix_imag_gpu = cp.zeros_like(matrix_batch_gpu)
img_gpu = cp.zeros((cfg['n_batch'], cfg['nx'], cfg['nz']), dtype=cp.float32)
img_imag_gpu = cp.zeros_like(img_gpu)

nblock = 128
n_ascans = matrix.shape[0] * matrix.shape[1]
grid_size = ((n_ascans + nblock - 1) // nblock,)
block_size = (nblock,)

filt_kernel(grid_size, block_size, (int_params, matrix_batch_gpu, bandpass_coef, matrix_filt_gpu))
filt_kernel(grid_size, block_size, (int_params, matrix_filt_gpu, hilb_coef, matrix_filt_gpu))
# matrix_filt = cp.asnumpy(matrix_filt_gpu)
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(matrix_filt[0, 0, 0, :])
# ax[1].plot(matrix_filt[1, 0, 0, :])


grid_size_img = (cfg['nz'] // 32, cfg['nx'] // 32)
shared_mem_size = 32 * 32 * cfg['n_batch'] * 4 * 2
pwi_kernel(grid_size_img, (32, 32), (int_params, float_params, cp.asarray(angles, dtype=cp.float32),
                                     matrix_filt_gpu, matrix_imag_gpu, img_gpu, img_imag_gpu),
           shared_mem=shared_mem_size)

img = cp.asnumpy(img_gpu)
img_imag_gpu = cp.asnumpy(img_imag_gpu)
img_abs = np.sqrt(img**2 + img_imag_gpu**2)

fig, ax = plt.subplots()
ax.imshow(img_abs[0, ...])
plt.show()