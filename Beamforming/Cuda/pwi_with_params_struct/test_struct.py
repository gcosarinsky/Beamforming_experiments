import numpy as np
import cupy as cp

# Configuración de parámetros
float_params_host = np.array([62.5, 6.3, 0.5, 2.0, 8.0, 10.0, 0.2, 0.2, -20.0, 1.0, 0.0, 32.0], dtype=np.float32)
int_params_host = np.array([62, 128, 25, 224, 224, 1024], dtype=np.int32)

# Copiar parámetros a la GPU
float_params_gpu = cp.asarray(float_params_host)
int_params_gpu = cp.asarray(int_params_host)

# Cargar y ejecutar el kernel
code = """
extern "C" __global__ void test_kernel(const float *float_params, const int *int_params) {
    int tid = threadIdx.x;

    if (tid == 0) {
        printf("Float Params:\\n");
        for (int i = 0; i < 12; i++) {
            printf("float_params[%d] = %f\\n", i, float_params[i]);
        }

        printf("Int Params:\\n");
        for (int i = 0; i < 6; i++) {
            printf("int_params[%d] = %d\\n", i, int_params[i]);
        }
    }
}
"""

module = cp.RawModule(code=code)
test_kernel = module.get_function("test_kernel")

# Configurar grid y bloques
grid_size = (1,)
block_size = (1,)

# Llamar al kernel
test_kernel(grid_size, block_size, (float_params_gpu, int_params_gpu))
cp.cuda.Device().synchronize()