import cupy as cp
import time

def medir_tiempo_svd(tamaño):
    """
    Mide el tiempo de ejecución del cálculo de SVD en la GPU.

    Args:
        tamaño: Tupla que define el tamaño de la matriz.

    Returns:
        Tiempo de ejecución en segundos.
    """
    # Crear una matriz aleatoria en la GPU
    matriz = cp.random.rand(*tamaño)

    # Sincronizar antes de comenzar la medición
    cp.cuda.Device().synchronize()

    # Medir el tiempo de ejecución del SVD
    print("Calculando SVD...")
    t0 = time.perf_counter()
    U, S, Vt = cp.linalg.svd(matriz, full_matrices=False, driver='cusolver')
    cp.cuda.Device().synchronize()
    t_total = time.perf_counter() - t0

    return t_total

if __name__ == "__main__":
    tamaño = (250, 10000)  # Tamaño de la matriz
    tiempo = medir_tiempo_svd(tamaño)
    print(f"Tiempo de ejecución del SVD: {tiempo:.2f} segundos")