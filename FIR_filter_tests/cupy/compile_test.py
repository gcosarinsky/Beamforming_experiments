import cupy as cp

# Definir el kernel CUDA
hello_world_kernel = r"""
extern "C" __global__
void hello_world_kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 42.0;  // Asignar un valor constante
}
"""

# Crear el módulo RawModule
module = cp.RawModule(code=hello_world_kernel)
kernel = module.get_function("hello_world_kernel")

# Configurar grid y bloques
block_size = 256
grid_size = 4

# Crear un arreglo en la GPU
n_elements = block_size * grid_size
data = cp.zeros(n_elements, dtype=cp.float32)

# Ejecutar el kernel
kernel((grid_size,), (block_size,), (data,))

# Verificar el resultado
print("Resultado:", data[:10].get())  # Imprimir los primeros 10 elementos