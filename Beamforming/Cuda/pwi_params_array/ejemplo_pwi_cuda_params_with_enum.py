import time

import numpy as np
import cupy as cp
from scipy.io import loadmat
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from helper_funcs import KernelParameters


def return_pw_cuda_beamformer(cfg):
    """
    Configura el beamformer utilizando los nuevos kernels CUDA.
    """
    # Generar coeficientes del filtro FIR
    bandpass_coef_cpu = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                                      pass_zero=False)
    bandpass_coef = cp.asarray(bandpass_coef_cpu, dtype=cp.float32)

    # Crear objeto KernelParameters
    params = KernelParameters(cfg)
    int_params = cp.asarray(params.get_int_array(), dtype=cp.int32)
    float_params = cp.asarray(params.get_float_array(), dtype=cp.float32)

    # Cargar código CUDA
    codepath = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\Beamforming\Cuda\pwi_params_array\\'
    with open(codepath + r'pwi_kernels.cu', encoding='utf-8') as f:
        code = f.read()

    module = cp.RawModule(code=code)
    filt_kernel = module.get_function('filt_kernel')
    pwi_kernel = module.get_function('pwi')

    def beamformer(matrix):
        """
        Aplica el beamformer a los datos de entrada.
        """
        matrix_gpu = cp.asarray(matrix, dtype=cp.int16)
        matrix_filt_gpu = cp.zeros_like(matrix_gpu)
        matrix_imag_gpu = cp.zeros_like(matrix_gpu)
        img_gpu = cp.zeros((cfg['nz'], cfg['nx']), dtype=cp.float32)
        img_imag_gpu = cp.zeros_like(img_gpu)
        cohe_gpu = cp.zeros_like(img_gpu)

        # Aplicar filtro FIR
        nblock = 128
        n_ascans = matrix.shape[0] * matrix.shape[1]
        grid_size = ((n_ascans + nblock - 1) // nblock,)
        block_size = (nblock,)
        filt_kernel(grid_size, block_size, (int_params, matrix_gpu, bandpass_coef, matrix_filt_gpu))

        # Aplicar transformada de Hilbert
        filt_kernel(grid_size, block_size, (int_params, matrix_filt_gpu, bandpass_coef, matrix_imag_gpu))

        # Aplicar beamforming
        grid_size_img = (cfg['nz'] // 32, cfg['nx'] // 32)
        pwi_kernel(grid_size_img, (32, 32), (int_params, float_params, cp.asarray(cfg['angles'], dtype=cp.float32),
                                             matrix_filt_gpu, matrix_imag_gpu, img_gpu, img_imag_gpu, cohe_gpu))

        # Copiar resultados de la GPU a la CPU
        cp.cuda.Device().synchronize()
        img = cp.asnumpy(img_gpu)
        img_imag = cp.asnumpy(img_imag_gpu)
        cohe = cp.asnumpy(cohe_gpu)
        return img, img_imag, cohe

    return beamformer


def measure_time(beamformer, cfg, n, repeat_data=False):
    """
    Mide el tiempo de ejecución del beamformer.
    """
    if repeat_data:
        q = np.random.randint(low=np.iinfo(np.int16).min, high=np.iinfo(np.int16).max, size=cfg['matrix_shape'],
                              dtype=np.int16)
        print('start beamforming')
        t0 = time.perf_counter()
        for i in range(n):
            beamformer(q)
        t = time.perf_counter() - t0
        print('end')

    else:
        q = np.random.randint(low=np.iinfo(np.int16).min, high=np.iinfo(np.int16).max, size=(n,) + cfg['matrix_shape'],
                              dtype=np.int16)
        print('start beamforming')
        t0 = time.perf_counter()
        for i in range(n):
            beamformer(q[i, ...])
        t = time.perf_counter() - t0
        print('end')

    return n / t

if __name__ == '__main__':
    # Cargar adquisición
    data_path = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\MUST/matlab/pruebas/pwi_acq_25angles.mat'
    data = loadmat(data_path)
    matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)
    angles = data['angles']

    cfg = {'fs': 62.5, 'c1': 6.3, 'pitch': 0.5, 'n_elementos': 128, 'n_angles': angles.size, 'f1': 2., 'f2': 8.,
           'taps': 62, 'bfd': 10, 'x_step': 0.2, 'z_step': 0.2, 'x0_roi': -20., 'z0_roi': 1., 'nx': 224, 'nz': 224,
           'n_samples': matrix.shape[-1], 'angles': angles.flatten(), 't_start': 0.}
    cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2
    cfg['matrix_shape'] = (cfg['n_angles'], cfg['n_elementos'], cfg['n_samples'])

    bffun = return_pw_cuda_beamformer(cfg)
    img, img_imag, cohe = bffun(matrix)
    img_abs = np.abs(img + 1j * img_imag)

    fig, ax = plt.subplots()
    ax.imshow(np.abs(img))
    plt.show()