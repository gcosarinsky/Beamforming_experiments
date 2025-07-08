import smb_funcs as smb
import numpy as np
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt


def return_pw_tf_beamformer(cfg):
    bandpass_coef = scipy.signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                                  pass_zero=False).astype(np.float32)

    # Crear el kernel (filtro FIR)
    kernel = tf.constant(bandpass_coef.reshape(-1, 1, 1))  # (kernel_size, in_channels, out_channels)

    angles = data['angles'].flatten()

    angles = 180 * angles/np.pi #asdfdsfsdfsdfsdfsdfsdfsdfsdfsd

    n_ch, nx, nz, ns = [cfg[k] for k in ['n_elementos', 'nx', 'nz', 'n_samples']]
    pos_x = np.linspace(cfg['x0_roi'], nx*cfg['x_step'], nx)
    pos_z = np.linspace(cfg['z0_roi'], nz*cfg['z_step'], nz)
    pos_trans = cfg['pitch'] * np.linspace(-(n_ch - 1) / 2, (n_ch - 1) / 2, n_ch)
    # rx_delay = (2*cfg['x_0']/cfg['c1']) * np.sin(cfg['angles']) * (cfg['angles'] < 0)
    rx_delay = 0
    fnum = 1

    print('computing SMB matrix')
    smb_mat = smb.gen_compound_mat(pos_trans, pos_z, pos_x, nz, nx, ns, n_ch, cfg['fs'], cfg['c1'], rx_delay,
                                   angles, fnum)

    indices = np.vstack((smb_mat.row, smb_mat.col)).T  # shape (nnz, 2)
    values = smb_mat.data
    shape = smb_mat.shape
    smb_mat_tf = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

    def beamformer(matrix):
        matrix = matrix.astype(np.float32) # Convertir a float32 para el filtrado
        reshaped = matrix.reshape(-1, cfg['n_samples'], 1)
        matrix_tf = tf.convert_to_tensor(reshaped)
        matrix_filt = tf.nn.conv1d(matrix_tf, kernel, stride=1, padding='SAME')
        # matrix_filt[:, 0, :] = 0
        # matrix_filt[:, -1, :] = 0
        matrix_filt_one_column = tf.reshape(matrix_filt, (shape[1], 1))
        temp = tf.sparse.sparse_dense_matmul(smb_mat_tf, matrix_filt_one_column)
        img = tf.reshape(temp, (nz, nx))
        return img.numpy()

    return beamformer, smb_mat_tf


if __name__ == '__main__':

    # load acquisition
    data_path = r'../../MUST/matlab/pruebas/pwi_acq_25angles.mat'
    data = scipy.io.loadmat(data_path)
    matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)
    angles = data['angles'].flatten()

    #dadadadad
    matrix[1:, ...] = 0
    # matrix[0:12, ...] = 0
    # matrix[13:, ...] = 0

    cfg = {'fs': 62.5, 'c1': 6.3, 'pitch': 0.5, 'n_elementos': 128, 'n_angles': angles.size, 'f1': 2, 'f2': 8,
           'taps': 63, 'x_step': 0.2, 'z_step': 0.2, 'x0_roi': -20, 'z0_roi': 1, 'nx': 200, 'nz': 200,
           'n_samples': matrix.shape[-1], 'angles': angles}
    cfg['x_0'] = cfg['pitch'] * (cfg['n_elementos'] - 1) / 2
    cfg['matrix_shape'] = (cfg['n_angles'], cfg['n_elementos'], cfg['n_samples'])

    # n_ch, nx, nz, ns = [cfg[k] for k in ['n_elementos', 'nx', 'nz', 'n_samples']]
    # pos_x = np.linspace(cfg['x0_roi'], nx*cfg['x_step'], nx)
    # pos_z = np.linspace(cfg['z0_roi'], nz*cfg['z_step'], nz)
    # pos_trans = cfg['pitch'] * np.linspace(-(n_ch - 1) / 2, (n_ch - 1) / 2, n_ch)
    # rx_delay = 0
    # fnum = 1
    #
    # print('computing SMB matrix')
    # smb_mat = smb.gen_compound_mat(pos_trans, pos_z, pos_x, nz, nx, ns, n_ch, cfg['fs'], cfg['c1'], rx_delay,
    #                                angles, fnum)

    bffun, smb_mat_tf = return_pw_tf_beamformer(cfg)
    img = bffun(matrix)

    plt.imshow(np.abs(img))