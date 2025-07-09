import numpy as np
import scipy.signal as signal
import cupy as cp
from scipy.io import loadmat
from helper_funcs import KernelParameters
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Cargar adquisición
data_path = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\MUST/matlab/pruebas/pwi_acq_25angles.mat'
data = loadmat(data_path)
matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)
angles = data['angles']

# Configuración de parámetros
cfg = {
    'fs': 62.5, 'c1': 6.3, 'pitch': 0.5, 'n_elementos': 128, 'n_angles': angles.size,
    'f1': 2., 'f2': 8., 'taps': 62, 'bfd': 10, 'x_step': 0.2, 'z_step': 0.2,
    'x0_roi': -20., 'z0_roi': 1., 'nx': 224, 'nz': 224, 'n_samples': matrix.shape[-1],'t_start': 0.}
cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2
cfg['matrix_shape'] = (cfg['n_angles'], cfg['n_elementos'], cfg['n_samples'])

# Generar coeficientes del filtro FIR
bandpass_coef = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                                  pass_zero=False)
bandpass_coef_gpu = cp.asarray(bandpass_coef, dtype=cp.float32)

# Crear objeto KernelParameters
params = KernelParameters(cfg)
int_params = cp.asarray(params.get_int_array(), dtype=cp.int32)

# Cargar código CUDA
codepath = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\Beamforming\Cuda\pwi_params_array\\'
with open(codepath + r'pwi_kernels.cu', encoding='utf-8') as f:
    code = f.read()

module = cp.RawModule(code=code)
filt_kernel = module.get_function('filt_kernel')

# Configurar señales
matrix_gpu = cp.asarray(matrix, dtype=cp.int16)
matrix_filt_gpu = cp.zeros_like(matrix_gpu)

# Configurar grid y bloques
nblock = 128
n_ascans = matrix.shape[0] * matrix.shape[1]
grid_size = ((n_ascans + nblock - 1) // nblock,)
block_size = (nblock,)

# Ejecutar el kernel
filt_kernel(grid_size, block_size, (int_params, matrix_gpu, bandpass_coef_gpu, matrix_filt_gpu))
cp.cuda.Device().synchronize()

# Copiar resultados a la CPU
matrix_filt = cp.asnumpy(matrix_filt_gpu)

# Plotear resultados
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(matrix[0, 0, :], label='Señal original')
ax.plot(matrix_filt[0, 0, :], label='Filtrado de la primera señal')