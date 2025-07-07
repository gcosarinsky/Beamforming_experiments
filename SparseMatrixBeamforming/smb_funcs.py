import numpy as np
from scipy.signal import hilbert
import scipy
from numba import njit
import matplotlib.pyplot as plt


@njit
def gen_single_planewave_mat(pos_trans, pos_z, pos_x, Nz, Nx, Ns, Nc, fs, sos, rx_delay, ang, fnum):

    """
    (Chat-GPT generated docstring, from data in the matlab script gen_saprse_matrix.m of the BINN repo)

    Generates the sparse beamforming matrix for delay-and-sum reconstruction.

    This function calculates the row, column, and value arrays needed to construct
    a sparse matrix that maps acquired RF data to pixel positions in the image
    according to the transmit and receive geometry.

    Parameters
    ----------
    pos_trans : ndarray of shape (Nc,)
        Lateral positions of transducer elements (in meters).
    pos_z : ndarray of shape (Nz,)
        Axial pixel positions (in meters).
    pos_x : ndarray of shape (Nx,)
        Lateral pixel positions (in meters).
    Nz : int
        Number of axial pixels.
    Nx : int
        Number of lateral pixels.
    Ns : int
        Number of samples per channel in the RF data.
    Nc : int
        Number of transducer channels (elements).
    fs : float
        Sampling frequency of the RF data (in Hz).
    sos : float
        Speed of sound in the medium (in m/s).
    rx_delay : float
        Fixed time offset to align the received signals to time zero (in seconds).
    ang : float
        Steering angle of the transmit beam (in degrees).
    fnum : float
        F-number used to define the dynamic receive aperture.

    Returns
    -------
    s_row : ndarray of int
        Row indices of the nonzero values in the sparse matrix.
    s_col : ndarray of int
        Column indices of the nonzero values in the sparse matrix.
    s_val : ndarray of float
        Values corresponding to the weights (interpolation coefficients) in the sparse matrix.

    Notes
    -----
    - The sparse matrix maps RF data samples to image pixels using linear interpolation.
    - Only contributions within the dynamic aperture are retained (based on `fnum`).
    - Indexing assumes column-major order when reshaping RF data or image arrays.
    - The returned arrays are trimmed to exclude zero-value entries for memory efficiency.
    """

    wave_source = pos_trans[-1] if ang < 0 else pos_trans[0]
    s_row = np.zeros(2 * Nz * Nx * Nc, dtype=np.int32)
    s_col = np.zeros(2 * Nz * Nx * Nc, dtype=np.int32)
    s_val = np.zeros(2 * Nz * Nx * Nc, dtype=np.float32)

    count = 0
    for z in range(Nz):
        a = pos_z[z] / (2 * fnum)
        for x in range(Nx):
            tx_d = pos_z[z] * np.cos(np.deg2rad(ang)) + (pos_x[x] - wave_source) * np.sin(np.deg2rad(ang))
            # tx_d = pos_z[z] * np.cos(np.deg2rad(ang)) + (pos_x[x] + x_0) * np.sin(np.deg2rad(ang))
            rx_d = np.sqrt(pos_z[z] ** 2 + (pos_x[x] - pos_trans) ** 2)
            total_time = rx_delay + (tx_d + rx_d) / sos  # OJO el singo del retardo es opuesto al del T_START en opencl
            # total_time = -rx_delay + (tx_d + rx_d) / sos

            # best_samp = np.clip(fs * total_time, 0, Ns - 1)
            best_samp = fs * total_time
            s_bot = np.floor(best_samp).astype(np.int32)
            s_interp = best_samp - s_bot

            for c in range(Nc):
                # pixel_index = z + Nz * x
                pixel_index = x + Nx * z
                if 0 < s_bot[c] < Ns - 2:
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


def gen_compound_mat(pos_trans, pos_z, pos_x, Nz, Nx, Ns, Nc, fs, sos, rx_delay, angs, fnum):
    """
    Generates a compound beamforming matrix by stacking multiple plane-wave matrices.

    This function calls `gen_single_planewave_mat` for each angle in `angs` and horizontally
    concatenates the resulting sparse matrices to form a compound beamforming matrix.

    Parameters
    ----------
    angs : ndarray of float
        Array of transmit steering angles (in degrees) for plane wave compounding.

    Returns
    -------
    scipy.sparse.coo_matrix
        Compound sparse beamforming matrix of shape (Nz*Nx, Ns*Nc * len(angs)).

    See Also
    --------
    gen_single_planewave_mat : Generates the sparse matrix for a single transmit angle.
    """

    n_angles = angs.size
    mat_list = []
    for i in range(n_angles):
        print(f"\rAngle: {i + 1}/{n_angles}", end="", flush=True)
        rows, cols, vals = \
            gen_single_planewave_mat(pos_trans, pos_z, pos_x, Nz, Nx, Ns, Nc, fs, sos, rx_delay, angs[i], fnum)
        mat_list.append(scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(Nz * Nx, Ns * Nc)))

    return scipy.sparse.hstack(mat_list)


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