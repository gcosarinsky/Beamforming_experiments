# test_script.py
import cupy as cp

# Configuración
cfg = {
    'nx': 200,
    'nz': 200,
}

# Crear memoria para la imagen
img_gpu = cp.zeros((cfg['nz'], cfg['nx']), dtype=cp.float32)

# Cargar el kernel
code = """
#define NX 200
#define NZ 200

extern "C" {
__global__ void test_kernel(float *img) {
    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= NZ || ix >= NX) return;

    unsigned short f_idx = iz * NX + ix;
    img[f_idx] = 1.0f;
}
}
"""
module = cp.RawModule(code=code)
test_kernel = module.get_function('test_kernel')

# Configurar grid y bloques
block_size = (16, 16)
grid_size = ((cfg['nz'] + block_size[0] - 1) // block_size[0],
             (cfg['nx'] + block_size[1] - 1) // block_size[1])

# Ejecutar el kernel
test_kernel(grid_size, block_size, (img_gpu,))

# Verificar resultados
img = cp.asnumpy(img_gpu)
print(img)