import numpy as np
import scipy.io
from scipy.signal import hilbert
from scipy.sparse import coo_matrix
from numba import njit
import matplotlib.pyplot as plt

@njit
def gen_mat(pos_trans, pos_z, pos_x, Nz, Nx, Ns, Nc, fs, sos, rx_delay, ang, fnum):
    wave_source = pos_trans[-1] if ang < 0 else pos_trans[0]
    s_row = np.zeros(2 * Nz * Nx * Nc, dtype=np.int32)
    s_col = np.zeros(2 * Nz * Nx * Nc, dtype=np.int32)
    s_val = np.zeros(2 * Nz * Nx * Nc, dtype=np.float32)

    count = 0
    for z in range(Nz):
        a = pos_z[z] / (2 * fnum)
        for x in range(Nx):
            tx_d = pos_z[z] * np.cos(np.deg2rad(ang)) + (pos_x[x] - wave_source) * np.sin(np.deg2rad(ang))
            rx_d = np.sqrt(pos_z[z] ** 2 + (pos_x[x] - pos_trans) ** 2)
            total_time = rx_delay + (tx_d + rx_d) / sos
            best_samp = np.clip(fs * total_time, 1, Ns - 1)
            s_bot = np.floor(best_samp).astype(np.int32)
            s_interp = best_samp - s_bot

            for c in range(Nc):
                pixel_index = z + Nz * x
                sample_index_1 = s_bot[c] + Ns * c
                sample_index_2 = s_bot[c] + 1 + Ns * c

                s_row[count] = pixel_index
                s_col[count] = sample_index_1
                if abs(pos_trans[c] - pos_x[x]) < a:
                    s_val[count] = 1 - s_interp[c]
                count += 1

                s_row[count] = pixel_index
                s_col[count] = sample_index_2
                if abs(pos_trans[c] - pos_x[x]) < a:
                    s_val[count] = s_interp[c]
                count += 1

    nnz_mask = s_val[:count] != 0
    return s_row[:count][nnz_mask], s_col[:count][nnz_mask], s_val[:count][nnz_mask]


def vis_bmode(img, pos_z, pos_x, dyn_range=40):
    img = hilbert(img, axis=0)
    env = 20 * np.log10(np.abs(img) + 1e-10)
    img_max = np.max(env)
    plt.imshow(env, extent=[pos_x[0], pos_x[-1], pos_z[-1], pos_z[0]],
               cmap='gray', aspect='auto', vmin=img_max - dyn_range, vmax=img_max)
    plt.xlabel('Lateral position (m)')
    plt.ylabel('Axial position (m)')
    plt.title('B-mode image')
    plt.colorbar(label='dB')
    plt.show()


# ---- Load MATLAB data ----
mat_data = scipy.io.loadmat('../Data/1.mat')
rf_filt = mat_data['rf_filt']  # shape (Ns, Nc)

Ns, Nc = rf_filt.shape

# Transducer and image grid setup
pitch = 0.3048e-3
pos_trans = pitch * np.linspace(-(Nc - 1) / 2, (Nc - 1) / 2, Nc)

Nz = 2048
Nx = 256
pos_z = np.linspace(5e-3, 35e-3, Nz)
pos_x = np.linspace(-15e-3, 15e-3, Nx)

# Imaging parameters
ang = -1  # degrees
fs = 40e6
sos = 1540
rx_delay = -4.1e-6
fnum = 1.4

# ---- Generate sparse matrix (row, col, val) ----
row, col, val = gen_mat(pos_trans, pos_z, pos_x, Nz, Nx, Ns, Nc, fs, sos, rx_delay, ang, fnum)

# ---- Perform beamforming without saving the matrix ----
sp_mat = coo_matrix((val, (row, col)), shape=(Nz * Nx, Ns * Nc))
img_flat = sp_mat @ rf_filt.ravel(order='F')
img = img_flat.reshape((Nz, Nx), order='F')

# ---- Display the result ----
vis_bmode(img, pos_z, pos_x, dyn_range=40)
