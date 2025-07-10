from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyopencl as cl
import utimag.utils as utils
from scipy import signal
import time


def return_pw_cl_beamformer(cfg):

    # ret_ini = (2*cfg['x_0']/cfg['c1']) * np.sin(cfg['angles']) * (cfg['angles'] < 0)

    bandpass_coef = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                                  pass_zero=False)  # .astype(np.float32)

    img_shape = (cfg['nz'], cfg['nx'])
    mac = utils.parameters_macros(cfg, utils.FLOAT_LIST + ['bfd'], utils.INT_LIST + ['n_angles'])
    codepath = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\Beamforming\OpenCL\\'
    with open(codepath + r'pwi_kernels.cl', encoding='utf-8') as f:
        code = f.read()
    with open(codepath + r'filtro_fir_conv_transient.cl', encoding='utf-8') as f:
        code += f.read()

    ctx, queue, mf = utils.init_gpu()
    prg = cl.Program(ctx, mac + code).build()

    buf = {'matrix': utils.DualBuffer(queue, None, cfg['matrix_shape'], data_type=np.int16),
           'matrix_imag': utils.DualBuffer(queue, None, cfg['matrix_shape'], data_type=np.int16),
           'matrix_filt': utils.DualBuffer(queue, None, cfg['matrix_shape'], data_type=np.int16),
           'img': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
           'img_imag': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
           'img_abs': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
           'cohe': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
           'bandpass_coef': utils.DualBuffer(queue, bandpass_coef, None, data_type=np.float32),
           'hilb_coef': utils.DualBuffer(queue, utils.HILB_COEF, None, data_type=np.float32),
           'angles': utils.DualBuffer(queue, cfg['angles'], None, data_type=np.float32),
           # 'ret_ini': utils.DualBuffer(queue, ret_ini, None, data_type=np.float32)}
           }

    def beamformer(matrix):
        buf['matrix'].c2g(matrix)
        knl1 = prg.filt_kernel_2
        knl2 = prg.pwi
        args = utils.return_knl_args(buf, ['matrix', 'bandpass_coef', 'matrix_filt'])
        knl1(queue, (cfg['n_angles'], cfg['n_elementos']), None, *args).wait()
        args = utils.return_knl_args(buf, ['matrix_filt', 'hilb_coef', 'matrix_imag'])
        knl1(queue, (cfg['n_angles'], cfg['n_elementos']), None, *args).wait()

        args = utils.return_knl_args(buf,
                                     ['matrix_filt', 'matrix_imag', 'img', 'img_imag', 'cohe', 'angles'])
        knl2(queue, img_shape, None, *args).wait()
        img = buf['img'].g2c()
        img_imag = buf['img_imag'].g2c()
        cohe = buf['cohe'].g2c()
        return img, img_imag, cohe

    return beamformer, buf


def measure_time(beamformer, cfg, n, repeat_data=False):

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
        # generate n random matrixes to apply the beamformer, changing the input data each time
        print('creating random data')
        q = np.random.randint(low=np.iinfo(np.int16).min, high=np.iinfo(np.int16).max, size=(n, ) + cfg['matrix_shape'],
                              dtype=np.int16)

        print('start beamforming')
        t0 = time.perf_counter()
        for i in range(n):
            beamformer(q[i, ...])
        t = time.perf_counter() - t0
        print('end')

    return n/t


if __name__ == '__main__':

    # load acquisition
    data_path = r'C:\Users\ggc\PROYECTOS\Beamforming_experiments\MUST/matlab/pruebas/pwi_acq_25angles.mat'
    data = loadmat(data_path)
    matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)
    angles = data['angles']

    #dadadadad
    #matrix[1:, ...] = 0
    matrix[0:12, ...] = 0
    matrix[13:, ...] = 0

    cfg = {'fs': 62.5, 'c1': 6.3, 'pitch': 0.5, 'n_elementos': 128, 'n_angles': angles.size, 'f1': 2, 'f2': 8,
           'taps': 62, 'bfd': 1, 'x_step': 0.2, 'z_step': 0.2, 'x0_roi': -20, 'z0_roi': 1, 'nx': 200, 'nz': 200,
           'n_samples': matrix.shape[-1], 'angles': angles, 't_start': 0}
    cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2
    cfg['matrix_shape'] = (cfg['n_angles'], cfg['n_elementos'], cfg['n_samples'])

    bffun, buf = return_pw_cl_beamformer(cfg)
    img, img_imag, cohe = bffun(matrix)
    img_abs = np.abs(img + 1j*img_imag)

    fig, ax = plt.subplots()
    # ax.imshow(img_abs / img_abs.max())
    ax.imshow(np.abs(img))
