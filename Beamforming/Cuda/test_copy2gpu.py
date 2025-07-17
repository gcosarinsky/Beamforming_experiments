import cupy as cp
import numpy as np
import time

def medir_transferencia_host_a_gpu(tamaño, repeticiones=10):
    """
    Mide la velocidad de transferencia de datos del host (CPU) a la GPU.

    Args:
        tamaño: Número de elementos en el array.
        repeticiones: Número de veces que se realiza la transferencia.

    Returns:
        Velocidad promedio de transferencia en GB/s.
    """
    # Crear datos aleatorios en la CPU
    print("Creando datos aleatorios en la CPU...")
    datos_cpu = [np.random.randint(size=tamaño, low=-10000, high=10000, dtype=np.int16)
                 for i in range(repeticiones)]
    print("Datos creados.")

    # Sincronizar antes de comenzar la medición
    cp.cuda.Device().synchronize()

    # Medir el tiempo de transferencia
    print("Midiendo tiempo de transferencia de CPU a GPU...")
    t0 = time.perf_counter()
    for i in range(repeticiones):
        datos_gpu = cp.asarray(datos_cpu[i])
        cp.cuda.Device().synchronize()
    t_total = time.perf_counter() - t0

    # Calcular velocidad de transferencia
    tamaño_bytes = datos_cpu[0].nbytes
    velocidad_gb_s = (tamaño_bytes * repeticiones / t_total) / (1024 ** 3)
    t_single = 1000*t_total / repeticiones
    return velocidad_gb_s, t_single


if __name__ == "__main__":
    tamaño = 20 * 11 * 128 * 1000 # Número de elementos en el array
    repeticiones = 100  # Número de transferencias
    gbs, t_single = medir_transferencia_host_a_gpu(tamaño, repeticiones)
    print(f"Velocidad promedio de transferencia: {gbs:.2f} GB/s")
    print(f"Tiempo promedio por transferencia: {t_single:.1f} milisegundos")