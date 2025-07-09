import numpy as np
import cupy as cp

# Declarar parámetros
float_params_host = np.array([62.5, 6.3, 0.5, 2.0, 8.0, 10.0, 0.2, 0.2, -20.0, 1.0, 0.0, 32.0], dtype=np.float32)
int_params_host = np.array([62, 128, 25, 224, 224, 1024], dtype=np.int32)

# Copiar parámetros a la GPU
float_params_gpu = cp.asarray(float_params_host)
int_params_gpu = cp.asarray(int_params_host)

# Código del kernel CUDA
with open('test_enum.cu', 'r') as f:
    code = f.read()

# Compilar el kernel
module = cp.RawModule(code=code)
print_params_kernel = module.get_function("print_params")

# Configurar grid y bloques
grid_size = (1,)
block_size = (1,)

# Ejecutar el kernel
print_params_kernel(grid_size, block_size, (float_params_gpu, int_params_gpu))
cp.cuda.Device().synchronize()