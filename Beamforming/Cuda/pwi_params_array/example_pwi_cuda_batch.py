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
    'n_batch': 10,
    'block_size_img': (32, 16),
    'filter_n_block': 128,
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

# assert cfg['n_batch'] <= 4, "Con n_batch mayor a 4 falla, no se por qué..."

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

# Compilar el código CUDA
module = cp.RawModule(code=code, options=('--use_fast_math',))
filt_kernel = module.get_function('filt_batch_kernel')
pwi_kernel = module.get_function('pwi_batch_1pix_per_thread')

# replicar la matriz de datos para crear un batch
matrix_batch = np.repeat(np.expand_dims(matrix, axis=0), cfg['n_batch'], axis=0)
# arrays gpu
matrix_batch_gpu = cp.asarray(matrix_batch, dtype=cp.int16)
cp.cuda.Device().synchronize()
matrix_filt_gpu = cp.zeros_like(matrix_batch_gpu)
matrix_imag_gpu = cp.zeros_like(matrix_batch_gpu)
img_gpu = cp.zeros((cfg['n_batch'], cfg['nz'], cfg['nx']), dtype=cp.float32)
img_imag_gpu = cp.zeros_like(img_gpu)
# cp.cuda.Device().synchronize()

nblock = cfg['filter_n_block']
n_ascans = matrix.shape[0] * matrix.shape[1]
grid_size = ((n_ascans + nblock - 1) // nblock,)
block_size_filter = (nblock,)

filt_kernel(grid_size, block_size_filter, (int_params, matrix_batch_gpu, bandpass_coef, matrix_filt_gpu))
# cp.cuda.Device().synchronize()
filt_kernel(grid_size, block_size_filter, (int_params, matrix_filt_gpu, hilb_coef, matrix_imag_gpu))
# cp.cuda.Device().synchronize()

block_size_img = cfg['block_size_img']
grid_size_img = (cfg['nz'] // block_size_img[0], cfg['nx'] // block_size_img[1])
shared_mem_size = block_size_img[0] * block_size_img[1] * cfg['n_batch'] * 4 * 2
assert shared_mem_size <= 48 * 1024, "Shared memory size exceeds the limit of 48KB per block."
pwi_kernel(grid_size_img, block_size_img, (int_params, float_params, cp.asarray(angles, dtype=cp.float32),
                                           matrix_filt_gpu, matrix_imag_gpu, img_gpu, img_imag_gpu),
           shared_mem=shared_mem_size)
# cp.cuda.Device().synchronize()

img = cp.asnumpy(img_gpu) #/ 1E-6
# cp.cuda.Device().synchronize()
img_imag = cp.asnumpy(img_imag_gpu) #/ 1E-6
# cp.cuda.Device().synchronize()
img_abs = np.abs(img + 1j * img_imag)

i = 5
matrix_filt = cp.asnumpy(matrix_filt_gpu)
matrix_imag = cp.asnumpy(matrix_imag_gpu)
fig, ax = plt.subplots(1, 2)
ax[0].plot(matrix_filt[0, 0, 0, :])
ax[1].plot(matrix_filt[i, 0, 0, :])
ax[0].plot(matrix_imag[0, 0, 0, :])
ax[1].plot(matrix_imag[i, 0, 0, :])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
scale = 1E5
ax[0].imshow(img_abs[i, ...]/scale)
ax[1].imshow(img[i, ...]/scale)
ax[2].imshow(img_imag[i, ...]/scale)
plt.show()


def speed_test(n=20, block_size_1=cfg['filter_n_block'], block_size_2=cfg['block_size_img']):
    nb = cfg['n_batch']
    print('generating random data')
    q = np.random.randint(low=np.iinfo(np.int16).min, high=np.iinfo(np.int16).max,
                          size=(n*nb,) + cfg['matrix_shape'],
                          dtype=np.int16)
    print('start beamforming')
    t0 = time.perf_counter()
    for i in range(0, n*nb, nb):
        matrix_batch_gpu = cp.asarray(q[i:(i+nb), ...], dtype=cp.int16)
        filt_kernel(grid_size, block_size_filter, (int_params, matrix_batch_gpu, bandpass_coef, matrix_filt_gpu))
        # cp.cuda.Device().synchronize()
        filt_kernel(grid_size, block_size_filter, (int_params, matrix_filt_gpu, hilb_coef, matrix_imag_gpu))
        # cp.cuda.Device().synchronize()
        pwi_kernel(grid_size_img, block_size_img, (int_params, float_params, cp.asarray(angles, dtype=cp.float32),
                                           matrix_filt_gpu, matrix_imag_gpu, img_gpu, img_imag_gpu),
                   shared_mem=shared_mem_size)
        pwi_kernel(grid_size_img, block_size_img, (int_params, float_params, cp.asarray(angles, dtype=cp.float32),
                                           matrix_filt_gpu, matrix_imag_gpu, img_gpu, img_imag_gpu),
                   shared_mem=shared_mem_size)
        # cp.cuda.Device().synchronize()

        img = cp.asnumpy(img_gpu)  # / 1E-6
        # cp.cuda.Device().synchronize()
        img_imag = cp.asnumpy(img_imag_gpu)  # / 1E-6
        # cp.cuda.Device().synchronize()
        img_abs = np.abs(img + 1j * img_imag)

    t = time.perf_counter() - t0
    print(f'end, time: {t:.3f} s')
    print(f'average time per batch: {t / n:.3f} s')
    print(f'FPS: {nb*n / t:.2f}')