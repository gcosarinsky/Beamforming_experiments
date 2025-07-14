import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def hilbert_filter_coeffs(N, window_type='hamming'):
    """
    Genera los coeficientes de un filtro Hilbert de longitud N (N impar) y les aplica una ventana.
    """
    if N % 2 == 0:
        raise ValueError("N debe ser impar")
    # Coeficientes ideales
    n = np.arange(N)
    h = np.zeros(N)
    center = N // 2
    for i in range(N):
        k = i - center
        if k == 0 or k % 2 == 0:
            h[i] = 0
        else:
            h[i] = 2 / (np.pi * k)
    # Aplica la ventana
    if window_type == 'hamming':
        window = np.hamming(N)
    elif window_type == 'blackman':
        window = np.blackman(N)
    elif window_type == 'hann':
        window = np.hann(N)
    else:
        raise ValueError("window_type desconocido")
    h_windowed = h * window
    return h_windowed


def apply_hilbert_filter(x, h):
    """
    Aplica el filtro Hilbert FIR a la señal x mediante convolución.
    """
    return np.convolve(x, h, mode='same')


def example():
    # Señal de prueba: seno
    fs = 62.5
    f0 = 15
    t = np.arange(0, 10/f0, 1/fs)
    x = np.sin(2 * np.pi * f0 * t)

    # Filtro Hilbert FIR
    N = 23  # longitud del filtro FIR, impar y suficientemente grande
    h = hilbert_filter_coeffs(N)
    x_hilb_fir = apply_hilbert_filter(x, h)

    # Filtro Hilbert usando scipy.signal.hilbert
    x_hilb_analytic = np.imag(hilbert(x))

    # Graficar resultados
    plt.figure(figsize=(10,6))
    plt.plot(t, x, label='Señal original')
    plt.plot(t, x_hilb_fir, label='Hilbert FIR (convolución)', linestyle='--')
    plt.plot(t, x_hilb_analytic, label='Hilbert Analítico (scipy)', linestyle=':')
    plt.xlim(0, t.max())
    plt.legend()
    plt.title('Comparación filtro Hilbert FIR vs scipy.signal.hilbert')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    example()