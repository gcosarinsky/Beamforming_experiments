from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyopencl as cl
import utimag.utils as utils
import helper_funcs as hf
from scipy import signal, io
import time

"""
23/4/2025
Este script usa el kernel pwi_0, que devuelve la imagen nomral, y la de coherencia. Se pùede comparar con lo de
fUS_Montaldo\DATOS_NERF_CSIC
"""

data_path = r'C:\Users\ggc\PROYECTOS\fUS_Montaldo\DATOS_NERF_CSIC\parameters\data_rata.mat'
data = loadmat(data_path)
p = hf.convert_to_dict(data['p'])
rf = data['rf'].T
bfcomp = data['bfcomp']

cfg = {}
for k in hf.param_names.keys():
    cfg[hf.param_names[k]] = p[k]

# cfg['t_start'] = p['Rt0'][0][0]
cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2
cfg['taps'] = 62
cfg['bfd'] = 1
cfg['x0_roi'] = -cfg['x_0']
cfg['z0_roi'] = 4
cfg['f1'] = 10
cfg['f2'] = 25

bandpass_coef = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                  pass_zero=False) #.astype(np.float32)

n_burst = rf.shape[0]
matrix_shape = rf.shape[1:]
img_shape = (cfg['nz'], cfg['nx'])
mac = utils.parameters_macros(cfg, utils.FLOAT_LIST + ['bfd'], utils.INT_LIST + ['n_angles'])
with open('../Beamforming/OpenCL/pwi_kernels_montaldo.cl') as f:
    code = f.read()
filt_code_path = r'../Beamforming/OpenCL/filtro_fir_conv_transient.cl'
with open(filt_code_path) as f:
    code += f.read()

ctx, queue, mf = utils.init_gpu()
prg = cl.Program(ctx, mac + code).build()

buf = {'matrix': utils.DualBuffer(queue, None, matrix_shape, data_type=np.int16),
       'matrix_imag': utils.DualBuffer(queue, None, matrix_shape, data_type=np.int16),
       'matrix_filt': utils.DualBuffer(queue, None, matrix_shape, data_type=np.int16),
       'img': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'img_imag': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'img_abs': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'cohe': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'bandpass_coef': utils.DualBuffer(queue, bandpass_coef, None, data_type=np.float32),
       'hilb_coef': utils.DualBuffer(queue, utils.HILB_COEF, None, data_type=np.float32),
       'angles': utils.DualBuffer(queue, p['Cax'][0], None, data_type=np.float32),
       'ret_ini': utils.DualBuffer(queue, p['Rt0'][0], None, data_type=np.float32)}

img = np.zeros(img_shape + (n_burst,), dtype=np.float32)
img_imag = np.zeros_like(img)
cohe = np.zeros_like(img)
knl_args = [utils.return_knl_args(buf, ['matrix', 'bandpass_coef', 'matrix_filt']),
        utils.return_knl_args(buf, ['matrix_filt', 'hilb_coef', 'matrix_imag']),
        utils.return_knl_args(buf, ['matrix_filt', 'matrix_imag', 'img', 'img_imag', 'cohe', 'angles', 'ret_ini'])]

print('start bf')
t0 = time.perf_counter()
for i in range(n_burst):
    buf['matrix'].c2g(rf[i, ...])
    # filters
    prg.filt_kernel_2(queue, (cfg['n_angles'], cfg['n_elementos']), None, *knl_args[0]).wait()
    args = utils.return_knl_args(buf, ['matrix_filt', 'hilb_coef', 'matrix_imag'])
    prg.filt_kernel_2(queue, (cfg['n_angles'], cfg['n_elementos']), None, *knl_args[1]).wait()
    # matrix_filt = buf['matrix_filt'].g2c()
    # matrix_imag = buf['matrix_imag'].g2c()

    prg.pwi_0(queue, img_shape, None, *knl_args[2]).wait()
    img[..., i] = buf['img'].g2c()
    img_imag[..., i] = buf['img_imag'].g2c()
    cohe[..., i] = buf['cohe'].g2c()
    print(i, end="\r", flush=True)

print('end bf')
print('ellapsed time: ', time.perf_counter() - t0)
print('nan?: ', np.any(np.isnan(img)))

# # plot RF data imag
# fig4, ax4 = plt.subplots()
# ax4.plot(matrix_filt[0, 50, :], label='filt')
# ax4.plot(matrix_imag[0, 50, :], label='imag')
# ax4.legend()

i = 9
img_abs = np.abs(img[..., i] + 1j*img_imag[..., i])
fig6, ax6 = plt.subplots(1, 2)
ax6[0].imshow(20*np.log10(img_abs/np.nanmax(img_abs)), cmap='gray', vmin=-50, vmax=0)
ax6[1].imshow(cohe[..., i], cmap='gray')

io.savemat('rata_bf_ggc.mat', {'bf1': img, 'bf2': img_imag, 'cohe': cohe})