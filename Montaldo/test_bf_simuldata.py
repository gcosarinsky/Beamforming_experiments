from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyopencl as cl
import utimag.utils as utils
import helper_funcs as hf

data_path = r'C:\Users\ggc\PROYECTOS\fUS_Montaldo\sitau2_openfus_v7\sitau2_openfus\fusExample\Beamforming\\'
sim_data = loadmat(data_path + 'simulated_data')

p = hf.convert_to_dict(sim_data['p'])
rf = sim_data['RF']
bf = sim_data['bf']

# reshape first acquisiton, as (n_angles, n_elements, n_samples)
matrix = rf[:, :, 0:p['Cn']].T  # first acq

cfg = {}
for k in hf.param_names.keys():
    cfg[hf.param_names[k]] = p[k]

cfg['t_start'] = p['Rt0'][0][0]
cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2
cfg['taps'] = 62
cfg['bfd'] = 1
cfg['x0_roi'] = -cfg['x_0']
# cfg['x_step'] = 0.05
# cfg['z_step'] = 0.02
# cfg['nx'] = 256
# cfg['nz'] = 200

# some plots -----------------------------------------------------------------
# plot RF data
fig0, ax0 = plt.subplots(1, p['Cn'])
for i in range(p['Cn']):
    ax0[i].imshow(np.abs(matrix[i, :, :].T), vmax=1000)

# plot a vertical line of the bf image, ix=63, where the particle is present
fig1, ax1 = plt.subplots()
ax1.plot(bf[0, :, 63])  # real
ax1.plot(bf[1, :, 63])  # imag
ax1.plot(np.sqrt(bf[0, :, 63]**2 + bf[1, :, 63]**2))  # envelope

# plot image, real, imag and envelope
fig2, ax2 = plt.subplots(1, 3)
ax2[0].imshow(bf[0, :, :])
ax2[1].imshow(bf[1, :, :])
ax2[2].imshow(np.sqrt(bf[0, :, :]**2 + bf[1, :, :]**2))
# -------------------------------------------------------------------------------

img_shape = (cfg['nz'], cfg['nx'])
mac = utils.parameters_macros(cfg, utils.FLOAT_LIST + ['bfd'], utils.INT_LIST + ['n_angles'])
with open(r'../Beamforming/OpenCL/pwi_kernels_montaldo.cl') as f:
    code = f.read()
with open(r'../Beamforming/OpenCL/filtro_fir_conv_transient.cl') as f:
    code += f.read()

ctx, queue, mf = utils.init_gpu()
prg = cl.Program(ctx, mac + code).build()

buf = {'matrix': utils.DualBuffer(queue, matrix, None, data_type=np.int16),
       'matrix_imag': utils.DualBuffer(queue, None, matrix.shape, data_type=np.int16),
       'img': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'img_imag': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'img_abs': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'cohe': utils.DualBuffer(queue, None, img_shape, data_type=np.float32),
       'hilb_coef': utils.DualBuffer(queue, utils.HILB_COEF, None, data_type=np.float32),
       'angles': utils.DualBuffer(queue, p['Cax'][0], None, data_type=np.float32),
       'ret_ini': utils.DualBuffer(queue, p['Rt0'][0], None, data_type=np.float32)}

# hilbert transform
args = utils.return_knl_args(buf, ['matrix', 'hilb_coef', 'matrix_imag'])
prg.filt_kernel_2(queue, (cfg['n_angles'], cfg['n_elementos']), None, *args).wait()
matrix_imag = buf['matrix_imag'].g2c()

# # plot RF data imag
# fig3, ax3 = plt.subplots(1, p['Cn'])
# for i in range(p['Cn']):
#     ax3[i].imshow(np.abs(matrix_imag[i, :, :].T), vmax=1000)

# fig4, ax4 = plt.subplots()
# ax4.plot(matrix[0, 50, :])
# ax4.plot(matrix_imag[0, 50, :])

args = utils.return_knl_args(buf, ['matrix', 'matrix_imag', 'img', 'img_imag', 'cohe', 'angles', 'ret_ini'])
prg.pwi_0(queue, img_shape, None, *args).wait()
img = buf['img'].g2c()
img_imag = buf['img_imag'].g2c()
img_abs = np.abs(img + 1j*img_imag)
cohe = buf['cohe'].g2c()
#
# fig5, ax5 = plt.subplots(2, 2)
# ax5[0, 0].imshow(img/img.max())
# ax5[0, 1].imshow(img_imag/img_imag.max())
# ax5[1, 0].imshow(img_abs)
# ax5[1, 1].imshow(cohe)

fig6, ax6 = plt.subplots(1, 2)
ax6[0].imshow(np.sqrt(bf[0, :, :]**2 + bf[1, :, :]**2))
ax6[0].set_title('montaldo')
ax6[1].imshow(img_abs)
ax6[1].set_title('cosarinsky')