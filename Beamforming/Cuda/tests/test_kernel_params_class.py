import numpy as np
import cupy as cp
from helper_funcs import KernelParameters

def test_kernel_params():
    # Crear datos de prueba
    params_dict = {
        'fs': 62.5,
        'c1': 1540.0,
        'pitch': 0.3,
        'f1': 2.0,
        'f2': 8.0,
        'bfd': 1.0,
        'x_step': 0.2,
        'z_step': 0.2,
        'x0_roi': -20.0,
        'z0_roi': 0.0,
        't_start': 0.0,
        'x_0': 19.2,
        'taps': 63,
        'n_elementos': 128,
        'n_angles': 25,
        'nx': 200,
        'nz': 200,
        'n_samples': 1024
    }

    try:
        # Crear instancia de KernelParameters
        kp = KernelParameters(params_dict)

        # Obtener arrays de parámetros
        float_params = kp.get_float_array()
        int_params = kp.get_int_array()

        # Cargar código del kernel
        with open('..\\bf_params.cu', 'r') as f:
            kernel_code = f.read()

        # Agregar kernel de prueba al código
        test_kernel_code = """
        extern "C" __global__ void test_kernel(const float *float_params, const int *int_params, float *output) {
            kernel_params_t params = build_params(float_params, int_params);

            // Guardar todos los parámetros en el array de salida para verificación
            output[0] = params.fs;
            output[1] = params.c1;
            output[2] = params.pitch;
            output[3] = params.f1;
            output[4] = params.f2;
            output[5] = params.bfd;
            output[6] = params.x_step;
            output[7] = params.z_step;
            output[8] = params.x0_roi;
            output[9] = params.z0_roi;
            output[10] = params.t_start;
            output[11] = params.x_0;
            output[12] = params.taps;
            output[13] = params.n_elementos;
            output[14] = params.n_angles;
            output[15] = params.nx;
            output[16] = params.nz;
            output[17] = params.n_samples;
        }
        """

        # Compilar módulo
        module = cp.RawModule(code=kernel_code + test_kernel_code)
        test_kernel = module.get_function('test_kernel')

        # Preparar arrays en GPU
        float_params_gpu = cp.asarray(float_params, dtype=cp.float32)
        int_params_gpu = cp.asarray(int_params, dtype=cp.int32)
        output_gpu = cp.zeros(18, dtype=cp.float32)

        # Ejecutar kernel
        test_kernel((1,), (1,), (float_params_gpu, int_params_gpu, output_gpu))

        # Obtener resultados
        output = output_gpu.get()

        # Verificar resultados
        print("Verificación de parámetros:")
        print("\nParámetros float:")
        for i, name in enumerate(kp.float_names):
            print(f"{name}: {output[i]} (original: {float_params[i]})")

        print("\nParámetros int:")
        for i, name in enumerate(kp.int_names):
            print(f"{name}: {output[i+12]} (original: {int_params[i]})")

        # Verificar que todos los valores coincidan
        float_ok = np.allclose(output[:12], float_params)
        int_ok = np.allclose(output[12:], int_params)

        if float_ok and int_ok:
            print("\n✓ Todos los parámetros coinciden correctamente")
        else:
            print("\n✗ Error: algunos parámetros no coinciden")

    except Exception as e:
        print(f"Error durante la prueba: {str(e)}")


if __name__ == '__main__':
    test_kernel_params()