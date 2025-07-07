import scipy.io
import numpy as np
from numba import njit
import math
from scipy.sparse import coo_matrix


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
            best_samp = np.clip(fs * total_time, 0, Ns - 1)
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

    nnz_mask = s_val != 0
    return s_row[nnz_mask], s_col[nnz_mask], s_val[nnz_mask]


if __name__ == '__main__':

    # Cargar matriz desde MATLAB
    mat_data = scipy.io.loadmat('../sp_matlab_2.mat')
    rows = mat_data['rows'].flatten() - 1  # MATLAB usa índices desde 1
    cols = mat_data['cols'].flatten() - 1
    vals = mat_data['vals'].flatten()
    nnz = vals != 0
    Nz = int(mat_data['Nz'][0][0])
    Nx = int(mat_data['Nx'][0][0])
    Ns = int(mat_data['Ns'][0][0])
    Nc = int(mat_data['Nc'][0][0])

    # Reconstrucción de parámetros
    pitch = 0.3048e-3
    pos_trans = pitch * np.linspace(-(Nc - 1) / 2, (Nc - 1) / 2, Nc)
    pos_z = np.linspace(5e-3, 35e-3, Nz)
    pos_x = np.linspace(-15e-3, 15e-3, Nx)
    fs = 40e6
    sos = 1540
    rx_delay = -4.1e-6
    ang = -1
    fnum = 1.4

    # Generar matriz en Python
    q = gen_mat(pos_trans, pos_z, pos_x, Nz, Nx, Ns, Nc, fs, sos, rx_delay, ang, fnum)
    sp_mat_py = coo_matrix((q[-1], (q[0], q[1])), shape=(Nz * Nx, Ns * Nc))

    # Matriz de MATLAB
    sp_mat_matlab = coo_matrix((vals[nnz], (rows[nnz], cols[nnz])), shape=(Nz * Nx, Ns * Nc))

    # Comparar
    l2_matlab = scipy.linalg.norm(sp_mat_matlab.data)
    l2_py = scipy.linalg.norm(sp_mat_py.data)
    diff = sp_mat_py - sp_mat_matlab
    error_l2 = scipy.linalg.norm(diff.data)
    error_l2_rel = error_l2 / l2_matlab
    nonzeros_diff = diff.count_nonzero()

    print("L2 error:", error_l2)
    print("L2 relative error :", error_l2_rel)
    print("N° de diferencias no nulas:", nonzeros_diff)
    print("¿Misma forma?:", sp_mat_py.shape == sp_mat_matlab.shape)

    print('print some values:')
    print(sp_mat_matlab.row[1000:1010])
    print(sp_mat_py.row[1000:1010])
    print(sp_mat_matlab.data[1000:1010])
    print(sp_mat_py.data[1000:1010])