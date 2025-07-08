import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import utimag.utils as utils

import FIR_filter_tests.hilbert_coef as hilb

# Configuración
cfg = {
    'n_elementos': 128,
    'n_angles': 25,
    'taps': 63,
    'fs': 62.5,
    'f1': 2,
    'f2': 8,
}

# Generar coeficientes del filtro FIR
bandpass_coef_cpu = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']], pass_zero=False)
hilb_coef_cpu = hilb.coef

# Cargar datos de entrada
data_path = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\MUST/matlab/pruebas/pwi_acq_25angles.mat'
data = loadmat(data_path)
matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)
cfg['n_samples'] = matrix.shape[-1]

# Configurar OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Cargar el kernel
kernel_path = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\Beamforming\OpenCL\filtro_fir_conv_transient.cl'
with open(kernel_path, 'r') as f:
    kernel_code = f.read()

mac = utils.parameters_macros(cfg, utils.FLOAT_LIST, utils.INT_LIST)
program = cl.Program(context, mac + kernel_code).build()

# Crear buffers
matrix_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
bandpass_coef_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bandpass_coef_cpu.astype(np.float32))
hilb_coef_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hilb_coef_cpu.astype(np.float32))
matrix_filt_gpu = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix.nbytes)
matrix_imag_gpu = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix.nbytes)

# Configurar dimensiones del kernel
global_size = (cfg['n_angles'], cfg['n_elementos'])
local_size = None

# Ejecutar el kernel para el filtro FIR
filt_kernel_2 = program.filt_kernel_2
filt_kernel_2(queue, global_size, local_size, matrix_gpu, bandpass_coef_gpu, matrix_filt_gpu)
filt_kernel_2(queue, global_size, local_size, matrix_filt_gpu, hilb_coef_gpu, matrix_imag_gpu)

# Leer resultados
matrix_filt = np.empty_like(matrix)
matrix_imag = np.empty_like(matrix)
cl.enqueue_copy(queue, matrix_filt, matrix_filt_gpu)
cl.enqueue_copy(queue, matrix_imag, matrix_imag_gpu)

# Visualizar resultados
i = 0
fig, ax = plt.subplots()
ax.plot(matrix[i, 0, :], label='raw')
ax.plot(matrix_filt[i, 0, :], label='filt')
ax.plot(matrix_imag[i, 0, :], label='imag')
plt.legend()
plt.show()