import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from scipy import signal

# Parámetros del filtro FIR
f1, f2 = 2, 8
taps = 63
fs = 62.5
bandpass_coef = signal.firwin(
    taps + 1,
    [2 * f1 / fs, 2 * f2 / fs],
    pass_zero=False
).astype(np.float32)

# Cargar la matriz desde el archivo .mat
data_path = r'/MUST/matlab/pruebas/pwi_acq_25angles.mat'
data = loadmat(data_path)
matrix = np.ascontiguousarray(data['a'].T, dtype=np.int16)  # (n_angles, n_ch, n_samples)

# Convertir a float32 para el filtrado
matrix = matrix.astype(np.float32)

# Obtener dimensiones
n_angles, n_ch, n_samples = matrix.shape

# Reordenar a (batch, time, channels) para tf.nn.conv1d
reshaped = matrix.reshape(-1, n_samples, 1)  # (n_angles * n_ch, n_samples, 1)

# Crear el kernel (filtro FIR)
kernel = tf.constant(bandpass_coef.reshape(-1, 1, 1))  # (kernel_size, in_channels, out_channels)

# Convertir a tensor de TensorFlow
signal_tf = tf.convert_to_tensor(reshaped)

# Aplicar convolución 1D (filtro FIR temporal)
filtered_tf = tf.nn.conv1d(signal_tf, kernel, stride=1, padding='SAME')

# Volver a forma original (n_angles, n_ch, n_samples)
filtered_matrix = tf.reshape(filtered_tf, [n_angles, n_ch, n_samples]).numpy()

fig, ax = plt.subplots()
i, j = 0, 100
ax.plot(matrix[i, j, :])
ax.plot(filtered_matrix[i, j, :])